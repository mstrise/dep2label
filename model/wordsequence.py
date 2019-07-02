# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 14:50:58
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from wordrep import WordRep


# https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print(
            "build word sequence feature extractor: %s..." %
            (data.word_feature_extractor))

        self.tasks = data.HP_tasks
        self.main_tasks = data.HP_main_tasks

        self.data = data

        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.status = data.status
        self.index_of_main_tasks = data.index_of_main_tasks

        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm_hidden = lstm_hidden

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(
                self.input_size,
                lstm_hidden,
                num_layers=self.lstm_layer,
                batch_first=True,
                bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(
                self.input_size,
                lstm_hidden,
                num_layers=self.lstm_layer,
                batch_first=True,
                bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print "CNN layer: ", self.cnn_layer
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = (kernel - 1) / 2
            for idx in range(self.cnn_layer):
                self.cnn_list.append(
                    nn.Conv1d(
                        data.HP_hidden_dim,
                        data.HP_hidden_dim,
                        kernel_size=kernel,
                        padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(
                    nn.BatchNorm1d(data.HP_hidden_dim))

        self.hidden2tagList = nn.ModuleList(
            [
                nn.Linear(
                    data.HP_hidden_dim,
                    data.label_alphabet_sizes[idtask]) for idtask in range(
                    data.HP_tasks)])

        if self.gpu:
            self.droplstm = self.droplstm.cuda()

            self.hidden2tagList.cuda()

            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda(
                    )

            else:
                self.lstm = self.lstm.cuda()

    def forward(
            self,
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
            inference):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover)

        if self.word_feature_extractor == "CNN":
            word_in = F.tanh(
                self.word2cnn(word_represent)).transpose(
                2, 1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2, 1).contiguous()
        else:

            packed_words = pack_padded_sequence(
                word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None

            lstm_out, (hidden_x, cell_x) = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)

            feature_out = self.droplstm(lstm_out.transpose(1, 0))

        if self.status == "train":

            outputs = [self.hidden2tagList[idtask](feature_out) for idtask in range(
                self.tasks) if not inference or idtask < self.main_tasks]
        else:

            self.main_tasks = self.index_of_main_tasks
            print("second")
            outputs = [self.hidden2tagList[idtask](feature_out) for idtask in range(
                self.tasks) if not inference or idtask in self.main_tasks]

        return outputs
