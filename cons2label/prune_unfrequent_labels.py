from argparse import ArgumentParser
from collections import Counter
from utils import flat_list, split_label, is_int, to_absolute_scale
import codecs
import os
import itertools
from cProfile import label

"""
python prune_unfrequent_labels.py \
--input /home/david/Escritorio/CPH_research/dataset_versions/seq_lu/ptb-train.seq_lu.3R.-2 \
--output /home/david/Escritorio/CPH_research/dataset_versions/seq_lu/ptb-train.seq_lu.3R.-2 \
--threshold 10
"""


#TODO: Should we weight more the first labels of the sequence?
def most_similar_common_label(label, label_counter, threshold):
    
    common_labels = [l for l in label_counter
                     if label_counter[l] > threshold]
    
    uncollapsed_symbols = label.split("+")
#    print "Predicting alternative for:",uncollapsed_symbols
    for r in range(len(uncollapsed_symbols),0,-1):
        for alternative in list(itertools.combinations(uncollapsed_symbols, r)):
#            print "trying alternative", alternative, "+".join(alternative) ,label_counter["+".join(alternative)]
            if label_counter["+".join(alternative)] > threshold:
#                 print "label", label
#                 print "selected alternative", alternative
#                 print 
#                 raw_input("NEXT")
                return "+".join(alternative)      
    #Should not reach here 
    return label        


def most_similar_common_level(level, level_counter, threshold,
                              sentence_abs_levels, index):
    
    if is_int(level): 
        level = int(level)
        common_levels = [int(l) for l in level_counter
                         if is_int(l) and level_counter[l] > threshold]
        
        root_common_levels = [int(l.replace("ROOT","")) if l != "ROOT" else 1 
                              for l in level_counter
                              if "ROOT" in l and level_counter[l] > threshold]
        
        if level > 0:
            diffs = sorted([(cl, level - cl) for cl in common_levels
                            if cl > 0], 
                           key = lambda t: t[1],
                           reverse=False)
            return diffs[0][0]

        else:
            diffs = sorted([(cl, level - cl) for cl in common_levels
                            if cl < 0],
                           key = lambda t: t[1],
                           reverse = True)        
            #Computing the differences to see whether the closest level
            #can be encoded using the top-down encoding
            diffs_from_root = sorted([(str(rcl)+"ROOT",int(sentence_abs_levels[index]) - rcl) 
                                      for rcl in root_common_levels],
                                      key= lambda t: t[1],
                                      reverse=False) 
    
            if diffs[0][1] == diffs_from_root[0][1]:
                return diffs_from_root[0][0]
            else:
                return min([diffs[0], diffs_from_root[0]],
                           key= lambda t: abs(t[1]))[0]
            
    return level

def prune_label(label, label_counter, threshold):
    
    if label_counter[label] > threshold:
        return label
    else:
#        print "pruning uncommon label", label, label_counter[label]
        return most_similar_common_label(label, label_counter, threshold)


#TODO: Need to take into account XROOT levels
def prune_level(level, level_counter, threshold, sentence_abs_levels, index):
    
    try:
        int(level)
    except ValueError:
    #    print "AAAAA", level
        return level
        
    if level_counter[level] > threshold:
    #    print "BBBBB", level
        return level
    else:
      #  print "ENTRA", level, level_counter[level], type(level_counter[level])
        return str(most_similar_common_level(level, level_counter, threshold, sentence_abs_levels, index))
    

def update_log(label, new_label, log):
    if new_label != label:
        if label not in log: log[label] = {}
        if new_label not in log[label]: 
            log[label][new_label] = 0
         
        log[label][new_label]+=1
        
    
if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", dest="input", 
                            help="Path to the original encoding used in Constituent Parsing as Sequence Labeling", 
                            default=None)
    arg_parser.add_argument("--output", dest="output", 
                            help="Path to the output encoding, formatted as multitask learning", default=None)
    arg_parser.add_argument("--threshold_level", dest="threshold_level", type=int, default=100,
                            help="Prune levels that occur less than a threshold")
    arg_parser.add_argument("--threshold_label", dest="threshold_label", type=int, default=10,
                            help="Prune labels that occur less than a threshold")
    arg_parser.add_argument("--threshold_unary", dest="threshold_unary", type=int, default=10,
                            help="Prune unaries that occur less than a threshold")
    args = arg_parser.parse_args()
    
    f_out = codecs.open(args.output,"w")
    
    with codecs.open(args.input) as f_input:
    
        
        sentences = [sentence.split("\n") for sentence in f_input.read().split("\n\n")
                     if sentence != ""]
        
        
        raw_labels = [split_label(e.split("\t")[2]) for e in flat_list(sentences)]
        
        levels, labels, unaries = [],[],[]
        for label in raw_labels:
            
            levels.append(label[0])
            labels.append(label[1])
            unaries.append(label[2])
            
        counter_levels = Counter(levels)
        counter_labels = Counter(labels)
        counter_unaries = Counter(unaries)        


        log_changes_level = {}
        log_changes_label = {}
        log_changes_unary = {}
        for sentence in sentences:
            sentence_levels = to_absolute_scale([split_label(e.split("\t")[2])[0] 
                                                for e in sentence])
            
            for idline, line in enumerate(sentence):
                word, postag = line.split("\t")[0], line.split("\t")[1]
                level, label, unary  = split_label(line.split("\t")[2])
              
                new_level = prune_level(level, counter_levels, args.threshold_level, sentence_levels, idline) 
                update_log(level,new_level, log_changes_level)

                new_label = prune_label(label, counter_labels, args.threshold_label)
                update_log(label,new_label, log_changes_label)
                
                new_unary = prune_label(unary, counter_unaries, args.threshold_unary)
                update_log(unary, new_unary, log_changes_unary)

             #   print new_level, new_label, new_unary
                
                pruned_label = [new_level]
                if new_label != "-EMPTY-":
                    pruned_label.append(new_label)
                if new_unary != "-EMPTY-":
                    pruned_label.append(new_unary)
                #pruned_label += new_unary if new_unary != "-EMPTY-" else ""
                
                f_out.write("\t".join([word, postag, "_".join(pruned_label)])+"\n")
            f_out.write("\n")
            
        print log_changes_level
        print
        print log_changes_label
        print
        print log_changes_unary

    