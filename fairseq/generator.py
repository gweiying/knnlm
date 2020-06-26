# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import time

from fairseq import utils
from fairseq.data import Dictionary


class SequenceGenerator(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']
        
        
        def batch_for_softmax(dec_out):
            """ batches decoder output
            
            Args:
                decoder_out: tuple where dec_out[0] is the logit output of the model
                dec_out[0]: (batch_size, token_size, dim)
                
            Output:
                batched decoder: (batch_size*token_size/softmax_batch) batches of (1, softmax_batch, dim)
            """
            first, rest = dec_out[0], dec_out[1:]
            batch_size, token_size, dim = first.shape
            if batch_size * token_size < self.softmax_batch:
                yield dec_out, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e,:],) + rest, False
                    s = e

        def gather_target_probs(probs, target):
            """ Indexes the target indices from probs 
            Args:
                probs: (1, softmax_batch, dim)
                target: (1, softmax_batch) shape
            Output:
                probs: (1, softmax_batch, 1)
            """
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs
        
        def combine_knn_and_vocab_probs(vocab_p, knn_p_k, knn_ind_k, coeff):
            """ Combines knn and vocab probs to form vector of top-k words selected by knn
            Args:
                vocab_p: (batch_size, token_size, dim)
                knn_p_k: (batch_size, token_size, k)
                knn_ind_k: (batch_size, token_size, k)
                coeff (float) 
            Output:
                curr_prob: (batch_size, token_size, k)
            """
            vocab_p_k = torch.gather(vocab_p, dim=2, index=knn_ind_k) # (batch_size, token_size, k)
            combine_probs = torch.stack([vocab_p_k, knn_p_k], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob # (batch_size, token_size, k)

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)
            
            ## batches model output for softmax
            batched = batch_for_softmax(decoder_out, orig_target)
            
            ## indexes probabilities of targets and outputs probs (1, batch_size*token_size, dim)
            probs, idx = None, 0
            for i, (bd, single_batch) in enumerate(batched):
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, \
                                    sample=sample).data #(1, softmax_batch, dim)
                if single_batch:
                    probs = curr_prob
                else:
                    if probs is None:
                        probs = curr_prob.new(dec_out[0].size(0)*dec_out[0].size(1), dec_out[0].size(2))
                    step = curr_prob.size(0) * curr_prob.size(1) # softmax_batch
                    end = step + idx
                    probs[idx:end,:] = curr_prob
                    idx = end
                
                print(probs)
            
            probs = probs.view(dec_out[0].shape) # (batch_size, token_size, dim)
            
            
            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                queries = bd[1][self.args.knn_keytype] # (token_size, batch_size, context_embed)
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob, knn_ind = dstore.get_knn_log_prob_k(queries) # (token_size, batch_size, k)
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2) # (batch_size, token_size, k)
                knn_ind = knn_ind.permute(1,0,2) # (batch_size, token_size, k)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()
                
                ## combines probs from knn and from language model
                probs = combine_knn_and_vocab_probs(probs, \
                            yhat_knn_prob, knn_ind, self.args.lmbda) # (batch_size, token_size, k)
                                              
                                             
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
                                              
        ## averaging across models
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        
        batch_size = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * batch_size
        for i in range(batch_size):
            # remove padding from ref
            avg_probs_i = avg_probs[i]
            knn_ind_i = knn_ind[i] # (token_size, k)
            values, indices = knn_max_i = torch.max(knn_ind_i, dim=-1) # token_size
            knn_top_i = utils.get_token_to_word_mapping(values, [self.pad, self.eos]) # token_size
            
            hypos.append([{
                'positional_scores': avg_probs_i,
                'knn_ind_i': knn_ind_i,
                'knn_top_i': knn_word_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])
        return hypos


