# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 14:10:30

import time
import sys
import argparse
import random
import torch
import gc
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import os
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from utils.data import Data
from cons2labels.utils import sequence_to_parenthesis
from cons2labels.evaluate import posprocess_labels
from dep2label import decodeDependencies
from cons2labels import encoding2multitask as rebuild

# Uncomment/Comment these lines to determine when and which GPU(s) to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed_num = 17
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

reload(sys)
sys.setdefaultencoding('UTF8')


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    # data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def load_gold_trees(data):
    file = open(data.gold_train_trees, 'r')
    trees = file.readlines()
    return trees


def decoded_to_tree(input):

    all_sentences = []
    all_preds = []
    raw_sentences = input.split("\n\n")
    for raw_sentence in raw_sentences:
        lines = raw_sentence.split("\n")
        if len(lines) != 1:
            sentence = [tuple(l.split("\t")[0:2]) for l in lines]
            preds_sentence = posprocess_labels(
                [l.split("\t")[2] for l in lines])
            all_sentences.append(sentence)
            all_preds.append(preds_sentence)
    parenthesized_trees = sequence_to_parenthesis(all_sentences, all_preds)
    return parenthesized_trees[0]


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
#    print "right_token", right_token
    return right_token, total_token


def recover_label(
        pred_variable,
        gold_variable,
        mask_variable,
        label_alphabet,
        word_recover,
        inference=False):
    """
    input:
        pred_variable (batch_size, sent_len): pred tag result
        gold_variable (batch_size, sent_len): gold result variable
        mask_variable (batch_size, sent_len): mask variable
    """

    if inference:

        pred_variable = pred_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        seq_len = pred_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []

        for idx in range(batch_size):
            pred = [
                label_alphabet.get_instance(
                    pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred_label.append(pred)
        return pred_label, None

    else:

        pred_variable = pred_variable[word_recover]
        gold_variable = gold_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = gold_variable.size(0)
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []

        for idx in range(batch_size):
            pred = [
                label_alphabet.get_instance(
                    pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [
                label_alphabet.get_instance(
                    gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
        return pred_label, gold_label


def recover_nbest_label(
        pred_variable,
        mask_variable,
        label_alphabet,
        word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    # print pred_variable.size()
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [
                label_alphabet.get_instance(
                    pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):

    lr = init_lr / (1 + decay_rate * epoch)
    print " Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, inference, nbest=None):

    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == "test":
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    # set model in eval model
    model.eval()
    # len(instances)#128 #For comparison against Vinyals et al. (2015)
    batch_size = 128
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    # D: Variable to collect the preds and gold prediction in multitask
    # learning
    pred_labels = {idtask: [] for idtask in range(data.HP_tasks)}
    gold_labels = {idtask: [] for idtask in range(data.HP_tasks)}

    nbest_pred_labels = {idtask: [] for idtask in range(data.HP_tasks)}
    nbest_pred_scores = {idtask: [] for idtask in range(data.HP_tasks)}

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
            instance, data.HP_gpu, inference, True)
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features,
                                                       batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask,
                                                       inference, nbest)

            tag_seq = []

            for idtask, task_nbest_tag_seq in enumerate(nbest_tag_seq):
                nbest_pred_result = recover_nbest_label(
                    task_nbest_tag_seq, mask, data.label_alphabet[idtask], batch_wordrecover)
                nbest_pred_labels[idtask] += nbest_pred_result
                nbest_pred_scores[idtask] += scores[idtask][batch_wordrecover].cpu(
                ).data.numpy().tolist()
                tag_seq.append(task_nbest_tag_seq[:, :, 0])

        else:
            tag_seq = model(
                batch_word,
                batch_features,
                batch_wordlen,
                batch_char,
                batch_charlen,
                batch_charrecover,
                mask,
                inference=inference)

        if not inference:

            for idtask, task_tag_seq in enumerate(tag_seq):
                pred_label, gold_label = recover_label(
                    task_tag_seq, batch_label[idtask], mask, data.label_alphabet[idtask], batch_wordrecover, inference=inference)
                pred_labels[idtask] += pred_label
                gold_labels[idtask] += gold_label
        else:

            if len(data.index_of_main_tasks) == data.HP_tasks:
                for idtask, task_tag_seq in enumerate(tag_seq):
                    pred_label, _ = recover_label(
                        task_tag_seq, None, mask, data.label_alphabet[idtask], batch_wordrecover, inference=inference)
                    # print("***", pred_label)
                    pred_labels[idtask] += pred_label
            else:
                # for now

                index_task = data.index_of_main_tasks[0]
                for idtask, task_tag_seq in enumerate(tag_seq):
                    pred_label, _ = recover_label(
                        task_tag_seq, None, mask, data.label_alphabet[index_task], batch_wordrecover, inference=inference)
                    #print("***", pred_label)
                    pred_labels[idtask] += pred_label
                    index_task += 1

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    tasks_results = []
    range_tasks = data.HP_tasks if not inference else len(
        data.index_of_main_tasks)
    for idtask in range(range_tasks):

        if not inference:
            acc, p, r, f = get_ner_fmeasure(
                gold_labels[idtask], pred_labels[idtask], data.tagScheme)
        else:
            acc, p, r, f = -1, -1, -1, -1

        if nbest:
            tasks_results.append(
                (speed, acc, p, r, f, nbest_pred_labels[idtask], nbest_pred_scores[idtask]))
            #tasks_results.append((speed,acc,p,r,f,nbest_pred_results, pred_scores))
        else:
            #    print "idtask", idtask
            #    print "pred_labels[idtask]", pred_labels[idtask]
            tasks_results.append(
                (speed, acc, p, r, f, pred_labels[idtask], nbest_pred_scores[idtask]))
    return tasks_results


def batchify_with_label(input_batch_list, gpu, inference, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """

    batch_size = len(input_batch_list)

    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    if not inference:
        labels = [sent[3] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros(
        (batch_size, max_seq_len)), volatile=volatile_flag).long()
    # Creating n label_seq_tensors, one for each task

    if not inference:
        label_seq_tensor = {
            idtask: autograd.Variable(
                torch.zeros(
                    (batch_size,
                     max_seq_len)),
                volatile=volatile_flag).long() for idtask in range(
                len(
                    labels[0]))}
    else:
        label_seq_tensor = None

    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(
            autograd.Variable(
                torch.zeros(
                    (batch_size,
                     max_seq_len)),
                volatile=volatile_flag).long())
    mask = autograd.Variable(
        torch.zeros(
            (batch_size,
             max_seq_len)),
        volatile=volatile_flag).byte()

    if not inference:

        for idx, (seq, label, seqlen) in enumerate(
                zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

            for idtask in label_seq_tensor:
                label_seq_tensor[idtask][idx,
                                         :seqlen] = torch.LongTensor(label[idtask])

            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(
                    features[idx][:, idy])

    else:

        for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(
                    features[idx][:, idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    if not inference:
        for idtask in label_seq_tensor:
            label_seq_tensor[idtask] = label_seq_tensor[idtask][word_perm_idx]

    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] *
                 (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(
        torch.zeros(
            (batch_size,
             max_seq_len,
             max_word_len)),
        volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(
        batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(
        batch_size * max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()

        if not inference:
            for idtask in label_seq_tensor:
                label_seq_tensor[idtask] = label_seq_tensor[idtask].cuda()

        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data):
    print "Training model..."
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)
    model = SeqModel(data)

    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=data.HP_lr,
            momentum=data.HP_momentum,
            weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=data.HP_lr,
            weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=data.HP_lr,
            weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=data.HP_lr,
            weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=data.HP_lr,
            weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(0)
    best_dev = -10
    best_dev_dep_score = -10
    best_dev_const_score = -10

    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0

        sample_loss = {idtask: 0 for idtask in range(data.HP_tasks)}
        right_token = {idtask: 0 for idtask in range(data.HP_tasks)}
        whole_token = {idtask: 0 for idtask in range(data.HP_tasks)}
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu, inference=False)
            instance_count += 1

            loss, losses, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_features,
                                                                  batch_wordlen, batch_char,
                                                                  batch_charlen, batch_charrecover,
                                                                  batch_label, mask, inference=False)
            for idtask in range(data.HP_tasks):
                right, whole = predict_check(
                    tag_seq[idtask], batch_label[idtask], mask)
                sample_loss[idtask] += losses[idtask].data[0]
                right_token[idtask] += right
                whole_token[idtask] += whole
                if end % 500 == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    print(
                        "     Instance: %s; Task %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
                        (end,
                         idtask,
                         temp_cost,
                         sample_loss[idtask],
                            right_token[idtask],
                            whole_token[idtask],
                            (right_token[idtask] +
                             0.) /
                            whole_token[idtask]))
                    if sample_loss[idtask] > 1e8 or str(sample_loss) == "nan":
                        print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
                        exit(0)
                    sys.stdout.flush()
                    sample_loss[idtask] = 0

            if end % 500 == 0:
                print "--------------------------------------------------------------------------"

            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        for idtask in range(data.HP_tasks):
            print(
                "     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
                (end,
                 temp_cost,
                 sample_loss[idtask],
                 right_token[idtask],
                 whole_token[idtask],
                 (right_token[idtask] +
                  0.) /
                    whole_token[idtask]))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(
            "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" %
            (idx, epoch_cost, train_num / epoch_cost, total_loss))
        print "totalloss:", total_loss
        if total_loss > 1e8 or str(total_loss) == "nan":
            print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
            exit(0)

        summary = evaluate(data, model, "dev", False, False)

        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        current_scores = []
        for idtask in xrange(0, data.HP_tasks):
            speed, acc, p, r, f, pred_labels, _ = summary[idtask]
            if data.seg:
                current_score = f
                current_scores.append(f)
                print(
                    "Task %d Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
                    (idtask, dev_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                current_scores.append(acc)
                print(
                    "Task %d Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" %
                    (idtask, dev_cost, speed, acc))

        pred_results_tasks = []
        pred_scores_tasks = []
        for idtask in xrange(data.HP_tasks):
            speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
            pred_results_tasks.append(pred_results)
            pred_scores_tasks.append(pred_scores_tasks)

        # EVALUATING ON DEV SET FOR CHOOSING THE BEST MODEL

        # MULTITASK LEARNING OF BOTH CONSTITUENCY AND DEPENDENCY PARSING
        if data.dependency_parsing and data.constituency_parsing:

            # CONSTITUENCY PARSING
            with tempfile.NamedTemporaryFile() as f_decode_mt:
                with tempfile.NamedTemporaryFile() as f_decode_st:

                    if len(data.index_of_main_tasks) > 1:
                        data.decode_dir = f_decode_mt.name
                        decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')
                        # transform between @ and {}
                        rebuild.rebuild_tree(data.decode_dir, decoded_st_dir)
                    else:
                        if data.decode_dir is None:
                            data.decode_dir = f_decode_st.name
                            decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    # evaluate the output comparing to the gold
                    command = [
                        "PYTHONPATH=" +
                        data.cons2labels,
                        "python",
                        data.evaluate,
                        " --input ",
                        decoded_st_dir,
                        " --gold ",
                        data.gold_dev_cons,
                        " --evalb ",
                        data.evalb,
                        ">",
                        f_decode_mt.name]

                    os.system(" ".join(command))
                    current_score_cons = float([l for l in f_decode_mt.read().split(
                        "\n") if l.startswith("Bracketing FMeasure")][0].split("=")[1])
                    print(current_score_cons)

            # DEPENDENCY PARSING
            with tempfile.NamedTemporaryFile() as f_decode_mt:
                with tempfile.NamedTemporaryFile() as f_decode_st:

                    if len(data.index_of_main_tasks) > 1:
                        data.decode_dir = f_decode_mt.name
                        decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    else:

                        print("else")
                        if data.decode_dir is None:
                            data.decode_dir = f_decode_st.name
                            decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    output_nn = open(data.decode_dir)

                    tmp = tempfile.NamedTemporaryFile().name

                    decodeDependencies.decode(output_nn, tmp, data.language)
                    current_score_depen = float(
                        decodeDependencies.evaluateDependencies(
                            data.gold_dev_dep, tmp))
                    print(current_score_depen)

        # SINGLE OR MULTITASK CONSTITUENCY PARSING
        elif data.constituency_parsing:

            with tempfile.NamedTemporaryFile() as f_decode_mt:
                with tempfile.NamedTemporaryFile() as f_decode_st:

                    if len(data.index_of_main_tasks) > 1:
                        data.decode_dir = f_decode_mt.name
                        decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')
                        # transform between @ and {}
                        rebuild.rebuild_tree(data.decode_dir, decoded_st_dir)
                    else:

                        if data.decode_dir is None:
                            data.decode_dir = f_decode_st.name
                            decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    command = [
                        "PYTHONPATH=" +
                        data.cons2labels,
                        "python",
                        data.evaluate,
                        " --input ",
                        decoded_st_dir,
                        " --gold ",
                        data.gold_dev_cons,
                        " --evalb ",
                        data.evalb,
                        ">",
                        f_decode_mt.name]

                    os.system(" ".join(command))
                    current_score = float([l for l in f_decode_mt.read().split(
                        "\n") if l.startswith("Bracketing FMeasure")][0].split("=")[1])
                    print "Current Score (from EVALB)", current_score, "Previous best dev (from EVALB)", best_dev

        # SINGLE OR MULTITASK DEPENDENCY PARSING
        elif data.dependency_parsing:

            with tempfile.NamedTemporaryFile() as f_decode_mt:
                with tempfile.NamedTemporaryFile() as f_decode_st:

                    # If we are learning multiple task we move it as a sequence
                    # labeling
                    if len(data.index_of_main_tasks) > 1:
                        data.decode_dir = f_decode_mt.name
                        decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    else:

                        if data.decode_dir is None:
                            data.decode_dir = f_decode_st.name
                            decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')

                    output_nn = open(data.decode_dir)
                    tmp = tempfile.NamedTemporaryFile().name

                    decodeDependencies.decode(output_nn, tmp, data.language)
                    current_score = decodeDependencies.evaluateDependencies(
                        data.gold_dev_dep, tmp)
                    print "Current Score (from LAS)", current_score, "Previous best dev (from LAS)", best_dev

        else:

            current_score = sum(current_scores) / len(current_scores)
            print "Current Score", current_score, "Previous best dev", best_dev

        """
        if data.choice_of_best_model=="strict_inclusive":
            print("strict inclusive")
            if current_score_depen>best_dev_dep_score and current_score_cons>best_dev_const_score:
                print("New string inclusive: LAS "+repr(current_score_depen)+" F1 "+repr(current_score_cons))
                print ("BOTH Exceed previous best acc score: LAS " + repr(best_dev_dep_score) + " F1 " + repr(best_dev_const_score))
                best_dev_dep_score = current_score_depen
                best_dev_const_score = current_score_cons
                model_name = data.model_dir + ".model"
                print "Overwritting model to", model_name
                torch.save(model.state_dict(), model_name)
            else:
                print("sofar the best: LAS " + repr(best_dev_dep_score) + " F1 " + repr(best_dev_const_score))

        elif data.choice_of_best_model=="harmonic_mean":
            print("harmonic mean")
            harmonic_mean = (2*current_score_cons*current_score_depen)/(current_score_cons+current_score_depen)
            if harmonic_mean > best_dev:
                print("New harmonic mean "+repr(harmonic_mean))
                print ("Exceed previous best harmonic mean score: "+ repr(best_dev)+" LAS "+repr(current_score_depen)+" F1 "+repr(current_score_cons))
                model_name = data.model_dir + ".model"
                print "Overwritting model to", model_name
                torch.save(model.state_dict(), model_name)
                best_dev=harmonic_mean
            else:
                print("sofar the best "+repr(best_dev))

        else:
            print("best single score")
            if current_score > best_dev:
                if data.seg:
                    print "Exceed previous best f score:", best_dev
                else:
                    print "Exceed previous best acc score:", best_dev

                model_name = data.model_dir +".model"
                print "Overwritting model to", model_name
                torch.save(model.state_dict(), model_name)

                best_dev = current_score
            else:
                print("sofar the best "+repr(best_dev))
        """

        # SAVE THE BEST MODEL

        # by default save model with highest harmonic mean when parsing both
        # dependency and constituency trees
        if data.dependency_parsing and data.constituency_parsing:
            harmonic_mean = (2 * current_score_cons * current_score_depen) / \
                (current_score_cons + current_score_depen)
            if harmonic_mean > best_dev:
                print("New harmonic mean " + repr(harmonic_mean))
                print (
                    "Exceed previous best harmonic mean score: " +
                    repr(best_dev) +
                    " LAS " +
                    repr(current_score_depen) +
                    " F1 " +
                    repr(current_score_cons))
                model_name = data.model_dir + ".model"
                print "Overwritting model to", model_name
                torch.save(model.state_dict(), model_name)
                best_dev = harmonic_mean
            else:
                print("sofar the best " + repr(best_dev))
        else:

            if current_score > best_dev:
                if data.seg:
                    print "Exceed previous best f score:", best_dev
                else:
                    print "Exceed previous best acc score:", best_dev

                model_name = data.model_dir + ".model"
                print "Overwritting model to", model_name
                torch.save(model.state_dict(), model_name)

                best_dev = current_score
            else:
                print("sofar the best " + repr(best_dev))

        summary = evaluate(data, model, "test", False)
        test_finish = time.time()
        test_cost = test_finish - dev_finish

        for idtask in xrange(0, data.HP_tasks):
            speed, acc, p, r, f, _, _ = summary[idtask]
            if data.seg:
                current_score = f
                print(
                    "Task %d Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
                    (idtask, test_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                print(
                    "Task %d Test: time: %.2fs speed: %.2fst/s; acc: %.4f" %
                    (idtask, test_cost, speed, acc))

        gc.collect()


def load_model_decode(data, name):
    print "Load Model from file: ", data.model_dir
    model = SeqModel(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()

    summary = evaluate(data, model, name, True, data.nbest)
    pred_results_tasks = []
    pred_scores_tasks = []
    range_tasks = len(data.index_of_main_tasks)

    for idtask in xrange(range_tasks):
        speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
        pred_results_tasks.append(pred_results)
        pred_scores_tasks.append(pred_scores)
    end_time = time.time()
    time_cost = end_time - start_time
    if data:
        print(
            "%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
            (name, time_cost, speed, acc, p, r, f))
    else:
        print(
            "%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" %
            (name, time_cost, speed, acc))

    return pred_results_tasks, pred_scores_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--config', help='Configuration File')
    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)

    status = data.status.lower()
    data.HP_gpu = torch.cuda.is_available()
    print "Seed num:", seed_num

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()

        train(data)

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)

        print data.raw_dir
        data.show_data_summary()
        data.generate_instance('raw')

        decode_results, pred_scores = load_model_decode(data, 'raw')

        if data.nbest:
            data.write_nbest_decoded_results(
                decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/finetune/decode)"
