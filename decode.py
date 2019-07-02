'''

'''

from argparse import ArgumentParser
from cons2labels.utils import sequence_to_parenthesis, get_enriched_labels_for_retagger, flat_list, \
    rebuild_input_sentence
import codecs
import os
import time
import sys
import uuid
from dep2label import decodeDependencies
from cons2labels import encoding2multitask as r
from utils.data import Data
STATUS_TEST = "test"
STATUS_TRAIN = "train"


def postprocess_labels(preds):
    for i in range(1, len(preds) - 2):
        if preds[i] in ["-BOS-", "-EOS-"] or preds[i].startswith("NONE"):
            preds[i] = "ROOT_S"
    if len(preds) != 3 and not preds[-2].startswith("NONE"): preds[-2] = "NONE"
    if preds[-1] != "-EOS-": preds[-1] = "-EOS-"
    if len(preds) == 3 and preds[1] == "ROOT":
        preds[1] = "NONE"

    return preds


if __name__ == '__main__':

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test", help="Path to the input test file as sequences", required=True)
    arg_parser.add_argument("--gold_constituency", dest="gold_constituency",
                            help="Path to the gold file in CoNNL-X format with constituency trees")
    arg_parser.add_argument("--gold_dependency", dest="gold_dependency",
                            help="Path to the gold file in CoNNL-X format with dependency trees")
    arg_parser.add_argument("--model", dest="model", help="Path to the model", required=True)
    arg_parser.add_argument("--status", dest="status", help="[train|test]", required=True)
    arg_parser.add_argument("--gpu", dest="gpu", help="[True|False]", default="False", required=False)
    arg_parser.add_argument("--output_constituency", help="output path for parsed constituency tree",dest="output_constituency")
    arg_parser.add_argument("--output_dependency", help="output path for parsed dependency tree", dest="output_dependency")
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository", required=True)
    arg_parser.add_argument("--multitask", dest="multitask", default=False, action="store_true")


    args = arg_parser.parse_args()
    reload(sys)
    sys.setdefaultencoding('UTF8')
    path_raw_dir = args.test
    path_name = args.model
    path_output = "/tmp/" + path_name.split("/")[-1] + ".output"
    path_tagger_log = "/tmp/" + path_name.split("/")[-1] + ".tagger.log"
    path_dset = path_name + ".dset"
    path_model = path_name + ".model"
    data = Data()
    data.load(path_dset)

    conf_str = """
    ### Decode ###
    status=decode
    """
    conf_str += "raw_dir=" + path_raw_dir + "\n"
    conf_str += "decode_dir=" + path_output + "\n"
    conf_str += "dset_dir=" + path_dset + "\n"
    conf_str += "load_model_dir=" + path_model + "\n"
    conf_str += "gpu=" + args.gpu + "\n"

    decode_fid = str(uuid.uuid4())
    decode_conf_file = codecs.open("/tmp/" + decode_fid, "w")
    decode_conf_file.write(conf_str)

    os.system("python " + args.ncrfpp + "/main.py --config " + decode_conf_file.name + " > " + path_tagger_log)

    log_lines = codecs.open(path_tagger_log).readlines()
    time_prediction = float([l for l in log_lines
                             if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:", "").replace("s", ""))

    # decode output from nn and build trees

    if data.dependency_parsing:
        output_content = codecs.open(path_output)
        start = time.time()
        decodeDependencies.decode(output_content, args.output_dependency, data.language)
        time_dependency = time.time() - start
        decodeDependencies.evaluateDependencies(args.gold_dependency, args.output_dependency)
        gold_depen = codecs.open(args.gold_dependency, "r")
        total_nb = 0
        for line in gold_depen:
            if line == "\n":
                total_nb += 1

    if data.constituency_parsing:
        gold_trees = codecs.open(args.gold_constituency).readlines()
        add_root_brackets = False
        if gold_trees[0].startswith("( ("):
            add_root_brackets = True

        if args.status.lower() == STATUS_TEST:
            # Reading stuff for evaluation
            sentences = []
            gold_labels = []
            for s in codecs.open(path_raw_dir).read().split("\n\n"):
                sentence = []
                for element in s.split("\n"):
                    if element == "": break
                    word, postag, label = element.strip().split("\t")[0], "\t".join(element.strip().split("\t")[1:-1]), \
                                          element.strip().split("\t")[-1]
                    sentence.append((word, postag))
                    gold_labels.append(label)
                if sentence != []: sentences.append(sentence)

            unary_preds = None
            end_merge_retags_time = 0
            time_mt2st = 0
        init_time_mt2st = time.time()

        if args.multitask:
            r.rebuild_tree(path_output, path_output + "2st")
            path_output = path_output + "2st"

        output_content = codecs.open(path_output)

        output_content = codecs.open(path_output).read()

        sentences = [rebuild_input_sentence(sentence.split("\n"))
                     for sentence in output_content.split("\n\n")
                     if sentence != ""]

        # I updated the output of the NCRFpp to be a tsv file
        init_posprocess_time = time.time()
        preds = [postprocess_labels([line.split("\t")[-1] if not args.multitask
                                    else line.split("\t")[-1]
                                     for line in sentence.split("\n")])
                 for sentence in output_content.split("\n\n")
                 if sentence != ""]

        end_posprocess_time = time.time() - init_posprocess_time

        init_parenthesized_time = time.time()
        parenthesized_trees = sequence_to_parenthesis(sentences, preds)  # ,None,None,None)

        if add_root_brackets:
            parenthesized_trees = ["( " + line + ")" for line in parenthesized_trees]

        end_parenthesized_time = time.time() - init_parenthesized_time

        tmpfile = codecs.open(args.output_constituency, "w")
        tmpfile.write("\n".join(parenthesized_trees) + "\n")

        # We read the time that it took to process the samples from the NCRF++ log file.
        log_lines = codecs.open(path_tagger_log).readlines()
        raw_time = float([l for l in log_lines
                          if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:", "").replace("s", ""))
        raw_unary_time = 0
        os.system(" ".join([data.evalb, args.gold_constituency, tmpfile.name]))
        os.remove("/tmp/" + decode_fid)

        #print (raw_time, raw_unary_time, end_posprocess_time, end_parenthesized_time, end_merge_retags_time)
        total_time = raw_time + raw_unary_time + end_posprocess_time + end_parenthesized_time + end_merge_retags_time
        with codecs.open(args.output_constituency + ".seq_lu", "w") as f_out_seq_lu:
            for (sentence, sentence_preds) in zip(sentences, preds):
                for ((w, pos), l) in zip(sentence, sentence_preds):
                    f_out_seq_lu.write("\t".join([w, pos, l]) + "\n")
                f_out_seq_lu.write("\n")

    if data.constituency_parsing and data.dependency_parsing:
        print("CONSTITUENCY AND DEPENDENCY PARSING")
        total = total_time + time_dependency
        total_dependency = time_dependency + time_prediction
        print("Sent/sec (dependency parsing) " + repr(round(len(gold_trees) / total_dependency,2)))
        print("Sent/sec (constituency parsing) " + repr(round(len(gold_trees) / total_time,2)))
        print("TOTAL TIME " + repr(total))
    elif data.constituency_parsing:
        print("CONSTITUENCY PARSING")
        print("Sent/sec " + repr(round(len(gold_trees) / total_time,2)))
        print("TOTAL TIME " + repr(round((total_time),2)))
    else:
        print("DEPENDENCY PARSING")
        total = time_dependency + time_prediction
        print("Sent/sec " + repr(round(total_nb / total,2)))
        print("TOTAL TIME " + repr(round(total,2)))
