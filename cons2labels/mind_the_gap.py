"""
Evaluates the sequentialized output produced by a sequence labeling constituent parsing.
The aim is to show which levels or constituents the system is struggling more with.

python mind_the_gap.py \
--input /home/david/Escritorio/dataset/ptb/ptb-train.seq_lu \
--gold /home/david/Escritorio/dataset/ptb/ptb-train.seq_lu
"""


from utils import flat_list, split_label, is_int
from argparse import ArgumentParser
from utils import sequence_to_parenthesis
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import codecs
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", dest="input", 
                            help="Path to the predicted output of the SL constituent parser", 
                            default=None)
    arg_parser.add_argument("--gold", dest="gold",
                            help="Path to the gold sequences of the dataset")
    
    args = arg_parser.parse_args()
    
    with codecs.open(args.input) as f_input:
        with codecs.open(args.gold) as f_gold:
            
            pred_sentences = [s.split("\n") for s in f_input.read().split("\n\n")]
            gold_sentences = [s.split("\n") for s in f_gold.read().split("\n\n")]
            
            pred_sequences = flat_list([[line.split("\t")[-1] for line in s
                                         if line != ""] for s in pred_sentences])
            gold_sequences = flat_list([[line.split("\t")[-1] for line in s
                                         if line != ""] for s in gold_sentences])
            
            pred_levels, gold_levels = [],[]
            pred_labels, gold_labels = [],[]
            pred_unaries, gold_unaries = [],[]
            
            assert (len(pred_sequences) == len(gold_sequences))
            
            for p, g in zip(pred_sequences, gold_sequences):

                plevel, plabel, punary = split_label(p)
                glevel, glabel, gunary = split_label(g)

                pred_levels.append(plevel)
                pred_labels.append(plabel)
                pred_unaries.append(punary)
                
                gold_levels.append(glevel)
                gold_labels.append(glabel)
                gold_unaries.append(gunary)
            
            print "Level report"
            level_report =classification_report(pred_levels,gold_levels)
            print level_report
            level_scores = []
            for l in level_report.split("\n")[2:-3]:
                score_label = filter(lambda x: x!='', l.split(" "))
                print score_label[0], score_label[3]
                level_scores.append((score_label[0], score_label[3]))
       #     print [(l.split("\t")[0],l.split("\t")[1]) for l in level_report.split("\n")[1:]]
            print "Label report"
            label_report = classification_report(pred_labels,gold_labels)
            print label_report
            print "Leaf unary chains report"
            unaries_report = classification_report(pred_unaries,gold_unaries)
            print unaries_report

            sns.set(style="whitegrid")
            
            rs = np.random.RandomState(365)
            values = rs.randn(365, 4).cumsum(axis=0)
            
            
            level_scores = sorted([(int(level),float(score)) for level,score in level_scores
                                   if is_int(level)])
            
            x = np.array([level for level, score in level_scores])
            y = np.array([score for level, score in level_scores])
            
       #     print x, len(x)
       #     print y, len(y)
            

            sns.set(style="white", palette="muted", color_codes=True)
            rs = np.random.RandomState(1)
            
            # Set up the matplotlib figure
            f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
            sns.despine(left=True)
            
            # Generate a random univariate dataset
            d = rs.normal(size=19)
# 
#             print d.shape, type(d)
#             print d

       #     print "dates", type(dates), len(dates)
           # data = pd.DataFrame(y, x, columns=map(str, x))
           # data = data.rolling(7).mean()
            
   #         print data
            #sns.lineplot(data=data, palette="tab10", linewidth=2.5)      
            print d      
            ax = sns.barplot(x=x, y=y, color=sns.xkcd_rgb["steel blue"])
            ax.set_ylabel("F-score")
            ax.set_xlabel("Level ($n_i$)")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    #        ax.set_xticklabels(x)
            plt.show()

#            print confusion_matrix(pred_levels,gold_levels)

            # Compute confusion matrix
#             class_names = []
#             for g in gold_levels:
#                 if g not in class_names: class_names.append(g)
#             cnf_matrix = confusion_matrix(gold_levels, pred_levels)
#             np.set_printoptions(precision=0)
# 
#             plt.figure()
#             plot_confusion_matrix(cnf_matrix, classes=class_names, 
#                                   normalize=True,
#                                   title='Normalized confusion matrix')
#             plt.show()            
            
            
            #print pred_sentences[0][0].split("\t")[-1]
            #pred_labels = flat_list([s.split("\n")[-1] for s in pred_sentences])
            #print pred_labels[0:100]
