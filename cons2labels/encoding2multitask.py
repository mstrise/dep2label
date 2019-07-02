from utils import rebuild_input_sentence
import codecs
import sys

def multitag_to_tag(multitag, multitask_char, split_char):

    multitag_split = multitag.split(multitask_char)[0:3]

    if multitag_split[1] in ["-BOS-","-EOS-","NONE"]:
        return multitag_split[1]
    
    if multitag_split[2] != "-EMPTY-":
        return split_char.join(multitag_split)
    else:
        return split_char.join(multitag_split[0:2])


def rebuild_tree(input, output):

    split_char ="@"
    multitask_char ="{}"
    sentence = []
    
    reload(sys)
    sys.setdefaultencoding('UTF8')

    labels = []

    with codecs.open(input) as f_input:
        lines = f_input.readlines()

    f_output = codecs.open(output, "w")
    for l in lines:
        if l != "\n":
            word, postag, label_raw = l.strip().split("\t")[0], "\t".join(l.strip().split("\t")[1:-1]), \
                                      l.strip().split("\t")[-1]

            label = multitag_to_tag(label_raw, multitask_char,
                                    split_char)  # The tasks that we care about are just the first three ones.
            sentence.append(l)
            labels.append(label)

        else:
            for token, label in zip(rebuild_input_sentence(sentence), labels):
                f_output.write("\t".join(token) + "\t" + label + "\n")
            sentence = []
            labels = []
            f_output.write("\n")


