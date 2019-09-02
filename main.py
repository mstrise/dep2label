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
from dep2label import decode_dependencies
from subprocess import PIPE,Popen
from StringIO import StringIO
#Uncomment/Comment these lines to determine when and which GPU(s) to use
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
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


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
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, inference=False):
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
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
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
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
        return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):

    lr = init_lr/(1+decay_rate*epoch)
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
    ## set model in eval model
    model.eval()
    batch_size = 128#len(instances)#128 #For comparison against Vinyals et al. (2015)
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
     
    pred_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    gold_labels = {idtask:[] for idtask in range(data.HP_tasks)}
     
    nbest_pred_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    nbest_pred_scores = {idtask:[] for idtask in range(data.HP_tasks)}

    if data.disjoint:
        treebank_indexes = {}
        for idxsample, sample in enumerate(instances):
            if sample[-1] not in treebank_indexes:
                treebank_indexes[sample[-1]] = []   
            treebank_indexes[sample[-1]].append(idxsample)    

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end > train_num:
            end = train_num
        instance = instances[start:end]

        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, inference, True)
        if nbest:
#            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, 
                                                       batch_wordlen, batch_char, 
                                                       batch_charlen, batch_charrecover, mask, 
                                                       inference, nbest)
            
            tag_seq = []
            
            for idtask, task_nbest_tag_seq in enumerate(nbest_tag_seq):   
                nbest_pred_result = recover_nbest_label(task_nbest_tag_seq, mask, data.label_alphabet[idtask], batch_wordrecover)
                nbest_pred_labels[idtask] += nbest_pred_result
                nbest_pred_scores[idtask] += scores[idtask][batch_wordrecover].cpu().data.numpy().tolist()
                tag_seq.append(task_nbest_tag_seq[:,:,0])
            
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask,
                            inference=inference)

        if not inference:
            for idtask, task_tag_seq in enumerate(tag_seq):
                pred_label, gold_label = recover_label(task_tag_seq, batch_label[idtask], mask, data.label_alphabet[idtask], 
                                                       batch_wordrecover, inference=inference)
                
                pred_labels[idtask]+=pred_label
                gold_labels[idtask]+=gold_label
        else:
            
            for idtask, task_tag_seq in enumerate(tag_seq):
                pred_label, _ = recover_label(task_tag_seq, None, mask, data.label_alphabet[idtask], 
                                                       batch_wordrecover, inference=inference)
                pred_labels[idtask]+=pred_label

    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
     
    # Evaluating the different tasks
    tasks_results = []
    range_tasks = data.HP_tasks if not inference else data.HP_main_tasks
    for idtask in range(range_tasks):

        valid_indexes=None
        if not inference:
            
            valid_gold_labels = [g for idx,g in enumerate(gold_labels[idtask])
                                 if not data.disjoint or idx in treebank_indexes[data.inv_dataset_ids[idtask]]]
            valid_pred_labels = [p for idx,p in enumerate(pred_labels[idtask])
                                 if not data.disjoint or idx in treebank_indexes[data.inv_dataset_ids[idtask]]]
            
            valid_indexes = [idx for idx,p in enumerate(pred_labels[idtask])
                             if not data.disjoint or idx in treebank_indexes[data.inv_dataset_ids[idtask]]]
            
            acc, p, r, f = get_ner_fmeasure(valid_gold_labels, valid_pred_labels, data.tagScheme)
        else:
            acc, p, r, f = -1, -1, -1,-1
        
        if nbest:
            raise NotImplementedError
        else:
            tasks_results.append((speed,acc,p,r,f,pred_labels[idtask],nbest_pred_scores[idtask], valid_indexes))

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
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    #label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    #Creating n label_seq_tensors, one for each task
    
    if not inference:
        label_seq_tensor = {idtask: autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long() 
                            for idtask in range(len(labels[0]))}
    else:
        label_seq_tensor = None

    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long())
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()

    if not inference:
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            
            for idtask in label_seq_tensor:
                label_seq_tensor[idtask][idx, :seqlen] = torch.LongTensor(label[idtask])
           
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
                
    else:
        for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                       
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    if not inference:
        for idtask in label_seq_tensor:
            label_seq_tensor[idtask] = label_seq_tensor[idtask][word_perm_idx]
    
    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):

            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
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
        #label_seq_tensor = label_seq_tensor.cuda()
        if not inference:
            for idtask in label_seq_tensor:
                label_seq_tensor[idtask] = label_seq_tensor[idtask].cuda()
            
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask#, dataset_origin
    #return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

 
def train(data):
    print "Training model..."
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    model = SeqModel(data)
    
    if data.pretrained_model is not None:
        model_dict = model.state_dict()
        
        #We load the weights for the layers that we have pretrained (e.g. for language modeling)
        pretrained_dict = torch.load(data.pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if data.pretrained_part == data.PRETRAINED_ALL or 
                           (data.pretrained_part == data.PRETRAINED_LSTMS and "hidden2tagList" not in k)}

        # We overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # We load the new state dict
        model.load_state_dict(model_dict)
    


    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        #optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(0)
    best_dev = -10
    range_valid_tasks = range(data.HP_tasks)
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
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
        
        #We get the indexes where are the samples of each (shuffled) treebank
        if data.disjoint:
            treebank_indexes = {}
            for idxsample, sample in enumerate(data.train_Ids):
                if sample[-1] not in treebank_indexes:
                    treebank_indexes[sample[-1]] = []   
                treebank_indexes[sample[-1]].append(idxsample)
        
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        if data.disjoint:
            tb_idxs = {tb:(0,batch_size) for tb in treebank_indexes}
            
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            if data.disjoint:
                eligible_treebanks = [t for t in treebank_indexes
                                      if tb_idxs[t][0] < len(treebank_indexes[t]) and idx < data.ignore_after_epoch[t] ]
                if eligible_treebanks == []: break
                
                tb = random.choice(eligible_treebanks)
                range_valid_tasks = data.dataset_ids[tb]
            
                idx_init, idx_end = tb_idxs[tb]
                train_idxs = treebank_indexes[tb][idx_init:idx_end]
                instance = [data.train_Ids[idx_ins] for idx_ins in train_idxs] #data.train_Ids[train_idxs]
                tb_idxs[tb] = (idx_end, idx_end+batch_size)
            else:  
                instance = data.train_Ids[start:end]
                
            if not instance:
                continue
          
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, inference=False)
            instance_count += 1

            loss, losses, tag_seq = model.neg_log_likelihood_loss(batch_word,batch_features, 
                                                                  batch_wordlen, batch_char, 
                                                                  batch_charlen, batch_charrecover, 
                                                                  batch_label, mask, range_valid_tasks, 
                                                                  inference=False)

            log=True
            for idtask in range_valid_tasks:
                right, whole = predict_check(tag_seq[idtask], batch_label[idtask], mask)
                sample_loss[idtask]+= losses[idtask].data[0]
                right_token[idtask]+=right
                whole_token[idtask]+=whole

            if end%500 == 0 and log:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                log = False
                if sample_loss[idtask] > 1e8 or str(sample_loss) == "nan":
                    print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
                    exit(0)
                sys.stdout.flush()
    
                for aux_idtask in range(data.HP_tasks):
    
                    if whole_token[aux_idtask] == 0:
                        print ("Task %d (no samples found)"%(aux_idtask))
                    else:
                        if data.inv_dataset_ids[aux_idtask] in eligible_treebanks:
                            print("Task %d %s Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(aux_idtask,data.inv_dataset_ids[aux_idtask],end, temp_cost, sample_loss[aux_idtask], right_token[aux_idtask], whole_token[aux_idtask],(right_token[aux_idtask]+0.)/whole_token[aux_idtask]))
                        else:
                            print("Task %d %s does not contain more samples; loss: %4f"%(aux_idtask,data.inv_dataset_ids[aux_idtask], 
                                                                                      losses[aux_idtask].data[0]))
                    sample_loss[aux_idtask] = 0                 

            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()
                                  
        temp_time = time.time()
        temp_cost = temp_time - temp_start
 
        for aux_idtask in range(data.HP_tasks):
    
            if whole_token[aux_idtask] == 0:
                print ("Task %d (no samples found)"%(aux_idtask))
            else:
                name_tb = data.inv_dataset_ids[aux_idtask]
                print("Task %d %s Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(aux_idtask,name_tb,len(treebank_indexes[name_tb]), temp_cost, sample_loss[aux_idtask], right_token[aux_idtask], whole_token[aux_idtask],(right_token[aux_idtask]+0.)/whole_token[aux_idtask]))

            sample_loss[aux_idtask] = 0
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx,epoch_cost, train_num/epoch_cost, total_loss))

        if total_loss > 1e8 or str(total_loss) == "nan":
            print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
            exit(0)

        summary = evaluate(data,model, "dev", False, False)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_scores = []
        for idtask in xrange(0, data.HP_tasks):
            speed,acc,p,r,f,pred_labels,_,valid_indexes = summary[idtask]
            if data.seg:
                current_score = f
                current_scores.append(f)
                print("Task %d Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(idtask, dev_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                current_scores.append(acc)
                print("Task %d Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(idtask, dev_cost, speed, acc))

        pred_results_tasks = []
        pred_scores_tasks = []
        
        pred_las_tasks = []
        valid_indexes = None
        for idtask in xrange(data.HP_tasks):
            speed, acc, p, r, f, pred_results, pred_scores, pred_indexes = summary[idtask]
            pred_results_tasks.append(pred_results)
            pred_scores_tasks.append(pred_scores_tasks)
            
            if idtask in data.task_metric and data.task_metric[idtask] == "LAS":
                pred_las_tasks.append(pred_results)
                valid_indexes = pred_indexes

        with tempfile.NamedTemporaryFile() as f_decode_mt:
            with tempfile.NamedTemporaryFile() as f_decode_st:

                # If we are learning multiple task we move it as a sequence labeling
                if data.HP_main_tasks > 1:
                    data.decode_dir = f_decode_mt.name
                    decoded_st_dir = f_decode_st.name
                    data.write_decoded_results(pred_las_tasks, 'dev', indexes=valid_indexes)
                    split_char = "{}"
                else:
                    if data.decode_dir is None:
                        data.decode_dir = f_decode_st.name
                        decoded_st_dir =  f_decode_st.name
                    data.write_decoded_results(pred_las_tasks, 'dev', indexes=valid_indexes)
                    split_char = "@"
                output_nn = open(data.decode_dir)
                tmp = tempfile.NamedTemporaryFile().name

                if data.offset:
                    decode_dependencies.decode_combined_tasks(output_nn, tmp, split_char)
                else:
                    print("decoding single task")
                    decode_dependencies.decode(output_nn, tmp, split_char)
                current_score = decode_dependencies.evaluate_dependencies(data.gold_dev_dep, tmp)
                print "Current Score (from LAS)", current_score, "Previous best dev (from LAS)", best_dev

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

        if data.HP_tasks_weight_decays:
            print "Updating the weights using linear weight decay. ",
            print "The old weights were", data.HP_tasks_weights,
            data.HP_tasks_weights =[max(weight-decay,0) 
                                    for weight,decay in zip(data.HP_tasks_weights, data.HP_tasks_weight_decays)]
            print ". The new weights are", data.HP_tasks_weights
            model.set_tasks_weights(data.HP_tasks_weights)
        gc.collect()


def load_model_decode(data, name):
    print "Load Model from file: ", data.model_dir
    model = SeqModel(data)
    model.load_state_dict(torch.load(data.load_model_dir))
    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    summary = evaluate(data, model, name, True, data.nbest)
    pred_results_tasks = []
    pred_scores_tasks = []
    range_tasks = data.HP_main_tasks
    
    #for idtask in xrange(data.HP_tasks):
    for idtask in xrange(range_tasks):
        speed, acc, p, r, f, pred_results, pred_scores, valid_indexes = summary[idtask]
        pred_results_tasks.append(pred_results)
        pred_scores_tasks.append(pred_scores)
    end_time = time.time()
    time_cost = end_time - start_time
    if data:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
        
    return pred_results_tasks, pred_scores_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--config',  help='Configuration File' )
    
    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)
    status = data.status.lower()
    data.HP_gpu = torch.cuda.is_available()
    print "Seed num:",seed_num
    
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
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/finetune/decode)"




