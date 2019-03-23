# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF
import numpy as np

class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print("build network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)


    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        nb_of_results = 3
        best_probs = {}
        best_indices = {}
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            #numpy_outs = outs.data.numpy()
            #sorted_probs = [-np.sort(-numpy_outs)[:,i].reshape(batch_size, seq_len) for i in range(2)]
            #best_indices = [mask.long()*torch.autograd.Variable(torch.from_numpy((-numpy_outs).argsort[:,i].reshape(batch_size, seq_len))) for i in range(2)]
            #probs = torch.nn.functional.log_softmax(outs, dim=1)
            #best_probs[0], best_indices[0] = torch.max(outs, 1)
            #for j in range(1,nb_of_results):
            #    for i in range((len(best_indices[0]))):
            #        outs.data[i, best_indices[j-1].data[i]] = -10000000000000
            #    best_probs[j], best_indices[j] = torch.max(outs,1)

            best_probs, best_indices = torch.topk(outs, k=nb_of_results, dim=1, sorted=True)

            #best_probs = torch.exp(best_probs)
            #normalization = torch.sum(best_probs, dim=1)

            #best_probs = torch.div(best_probs, normalization)

	    best_probs = torch.nn.functional.softmax(best_probs, dim=1)

            best_indices = [mask.long()*best_indices[:,i].contiguous().view(batch_size, seq_len) for i in range(nb_of_results)]
            best_probs = [best_probs[:, i].contiguous().view(batch_size, seq_len) for i in range(nb_of_results)]

            #for i in range(nb_of_results):
            #    best_indices[i] = mask.long()*best_indices[i].view(batch_size, seq_len)
            #    best_probs[i] = best_probs[i].view(batch_size, seq_len)

            #best_index = numpy_outs.argmax(axis=1)
            #best_prob = np.empty_like(best_index)
            #for i in range(len(best_index.tolist())):
            #    best_prob[i] = numpy_outs[i,best_index[i]]
            #    numpy_outs[i, best_index[i]] = 0

            #best_index2 = numpy_outs.argmax(axis=1)
            #best_prob2 = np.empty_like(best_index2)
            #for i in range(len(best_index2.tolist())):
            #    best_prob2[i] = numpy_outs[i, best_index2[i]]

            #best_indices = [mask.long()*best_index.view(batch_size, seq_len), mask.long()*best_index2.view(batch_size, seq_len)]
            #best_probs = [best_prob.view(batch_size, seq_len).data.cpu().numpy(), best_prob2.view(batch_size, seq_len).data.cpu().numpy()]
            #_, tag_seq  = torch.max(outs, 1)
            #tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            #tag_seq = mask.long() * tag_seq
        return best_indices, best_probs


    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)


    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

