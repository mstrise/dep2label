import random
import numpy as np
import codecs
random.seed(1)

# conllu file with gaze features
# 11	beautiful	_	ADJ	JJ	_	12	amod	_	667.0	222.33333333333331	3.0	1.0	243.0	667.0	2.0	1.0	1.0	0.0	140.0	0.0

# index of gaze features
# 9: total fixation duration (percentile, bin 20)
# 10: mean fixation duration (percentile, bin 20)
# 11: nb fixations
# 12: fixation probability
# 13: first fixation duration (percentile, bin 20)
# 14: first pass duration (percentile, bin 20)
# 15: nb re-fixations
# 16: re-read probability
# 17: w-1 fixation probability
# 18: w+1 fixation probability
# 19: w-1 fixation duration
# 20: w+1 fixation duration

def split_dataset(path, train_ratio=0.8):
    f_train = open("train.conllu", "w")
    f_dev = open("dev.conllu", "w")
    f_test = open("test.conllu", "w")
    dev_ratio = 0.9

    with open(path) as f:
        concat_tokens = ""
        sentence_conllu = []
        word_line_conllu_dict = {}
        for line in f:
            if not line == "\n":
                columns = line.strip().split("\t")
                concat_tokens = " ".join([concat_tokens, columns[1]])
                sentence_conllu.append(line)
            else:
                if not concat_tokens in word_line_conllu_dict:
                    list_lines_conllu = [sentence_conllu]
                    word_line_conllu_dict.update({concat_tokens: list_lines_conllu})
                else:
                    list_m = word_line_conllu_dict[concat_tokens]
                    list_m.append(sentence_conllu)
                concat_tokens = ""
                sentence_conllu = []

    train_first = True
    dev_first = True
    test_first = True

    for concat_tokens in sorted(word_line_conllu_dict.keys()):
        number = random.random()
        # sentences go to the training file
        if number <= train_ratio:
            if train_first:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_train)
                train_first = False
            else:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_train)
        # sentences go to the dev file
        elif train_ratio < number >= dev_ratio:
            if dev_first:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_dev)
                dev_first = False
            else:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_dev)
        # sentences go to the test file
        else:
            if test_first:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_test)
                test_first = False
            else:
                line_conllu_list = word_line_conllu_dict[concat_tokens]
                convert_from_dict_to_labels(concat_tokens, line_conllu_list, f_test)


def convert_from_dict_to_labels(words, lines, f):
    nb_copies = len(lines)
    for copy in range(nb_copies):
        words_list = words.strip().split(" ")
        m = lines[copy]
        for index, word in enumerate(words_list):
            line = m[index]
            f.write(line)
        f.write("\n")


def calculate_procentile(file, converted_file, index_feat):
    original_data = open(file, "r")
    converted_data = open(converted_file, "w")
    words = []
    times = []

    for line in original_data:
        if not line == "\n":
            feats_conll = line.strip().split("\t")
            token = feats_conll[1]
            pos = feats_conll[4]
            discrete = float(feats_conll[index_feat])
            words.append(token + "\t" + pos)
            times.append(discrete)
        else:
            converted_data.write("# TREEBANK=DUNDEE")
            converted_data.write("\n")
            bos = "-BOS-"
            full_label_BOS = str(bos + "\t" + bos + "\t" + bos)

            converted_data.write(full_label_BOS)
            converted_data.write("\n")

            a = np.percentile(times, 20)
            b = np.percentile(times, 40)
            c = np.percentile(times, 60)
            d = np.percentile(times, 80)

            for index, word in enumerate(words):
                time = times[index]
                if time <= a:
                    label = "0-20"
                    converted_data.write(word + "\t" + label)
                    converted_data.write("\n")
                elif a < time <= b:
                    label = "20-40"
                    converted_data.write(word + "\t" + label)
                    converted_data.write("\n")
                elif b < time <= c:
                    label = "40-60"
                    converted_data.write(word + "\t" + label)
                    converted_data.write("\n")
                elif c < time <= d:
                    label = "60-80"
                    converted_data.write(word + "\t" + label)
                    converted_data.write("\n")
                else:
                    label = "80-100"
                    converted_data.write(word + "\t" + label)
                    converted_data.write("\n")

            eos = "-EOS-"
            full_label_EOS = str(eos + "\t" + eos + "\t" + eos)
            converted_data.write(full_label_EOS)
            converted_data.write("\n\n")
            words = []
            times = []


def extract_gaze_feats(inputfile, output, index):
    original_data = open(inputfile, "r")
    converted_data = open(output, "w")
    words = []
    feats = []

    for line in original_data:
        if not line == "\n":
            columns = line.strip().split("\t")
            feat = columns[index]
            pos = columns[4]
            words.append(columns[1] + "\t" + pos)
            feats.append(int(float(feat)))

        else:
            converted_data.write("# TREEBANK=DUNDEE")
            converted_data.write("\n")
            bos = "-BOS-"
            full_labelBOS = str(bos + "\t" + bos + "\t" + bos)
            converted_data.write(full_labelBOS)
            converted_data.write("\n")
            eos = "-EOS-"
            full_labelEOS = str(eos + "\t" + eos + "\t" + eos)

            for i in range(len(words)):
                f = feats[i]
                w = words[i]
                converted_data.write(w + "\t" + repr(f))
                converted_data.write("\n")

            converted_data.write(full_labelEOS)
            converted_data.write("\n\n")

            words = []
            feats = []


def merge_multiple_labels(dependencies, gaze_feat1, gaze_feat2, gaze_feat3, file_merged_feats):
    file_merged = codecs.open(file_merged_feats, "w")

    with codecs.open(dependencies, "r") as file1, codecs.open(gaze_feat1, "r") as file2, codecs.open(gaze_feat2,
                                                                                                     "r") as file3 \
            , codecs.open(gaze_feat3, "r") as file4:

        sentence = []
        for line_dep, fe1, fe2, fe3 in zip(file1, file2, file3, file4):
            if line_dep.startswith("#"):
                file_merged.write(line_dep)
            elif not line_dep == "\n":
                dep_label = line_dep.strip("\n")
                feat1 = fe1.strip().split("\t")
                feat2 = fe2.strip().split("\t")
                feat3 = fe3.strip().split("\t")
                # feat4 = fe4.strip().split("\t")
                label_depend = feat1[2]
                f2 = feat2[2]
                f3 = feat3[2]
                # f4 = feat4[2]
                joint_labels = "{}".join((dep_label, label_depend, f2, f3))
                sentence.append(joint_labels)
            else:

                for s in sentence:
                    file_merged.write(s)
                    file_merged.write("\n")

                file_merged.write("\n")
                sentence = []
