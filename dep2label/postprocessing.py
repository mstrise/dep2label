import codecs
import dep2label.eval_dep.conll18_ud_eval as conll18
from argparse import ArgumentParser
class LabelPostProcessor():

    def tag_BOS(self,task, words_with_labels):
        bos = "-BOS-"
        if task == "1-task":
            label = str(bos)
        elif task == "2-task-combined":
            label = str(
                bos + "@" + bos + "{}" + bos)
        elif task == "2-task":
            label = str(
                bos + "{}" + bos)
        elif task =="3-task":
            label = str(
                bos + "{}" + bos + "{}" + bos)
        full_label = str(bos + "\t" + bos + "\t" + label)
        words_with_labels.update({0: full_label})
        return words_with_labels

    def tag_EOS(self,task, words_with_labels):
        eos = "-EOS-"
        if task == "1-task":
            label = str(eos)
        elif task == "2-task-combined":
            label = str(
                eos + "@" + eos + "{}" + eos)
        elif task == "2-task":
            label = str(
                eos + "{}" + eos)
        elif task == "3-task":
            label = str(
                eos + "{}" + eos + "{}" + eos)

        full_label = str(eos + "\t" + eos + "\t" + label)
        words_with_labels.update({len(words_with_labels) : full_label})
        return words_with_labels

    def separate_labels(self,file2decode):
        words_sep_labels = []
        sent_sep_labels = {}
        nb_sent = 1
        nb_words = 1

        def split_label(multitag, multitask_char):
            #if label separated by either @ or {}
            multitag_dep = multitag.split(multitask_char)
            labels = multitag_dep[0:5]
            return labels

        def split_label_combined(multitag, multitask_char):
            #if label separated by both @ and {}
            labels = []
            multitag = multitag.split(multitask_char)
            m = multitag[0].split("@")
            labels.append(m[0])
            labels.append(multitag[1])
            labels.append(m[1])
            return labels

        with codecs.open(file2decode) as label2f:
            lines = label2f.readlines()

        for line in lines:
            if line != "\n":
                word, postag, label_raw = line.strip().split("\t")[0], line.strip().split("\t")[
                    1], line.strip().split("\t")[-1]               
                if not word == "-BOS-" and not word == "-EOS-":
                    if "@" in label_raw and "{}" in label_raw:
                        label_sep = split_label_combined(label_raw, "{}")
                    elif "@" in label_raw:
                        label_sep = split_label(label_raw, "@")
                    else:
                        label_sep = split_label(label_raw, "{}")              
                    label_sep.insert(0, postag)
                    label_sep.insert(0, word)
                    words_sep_labels.append(label_sep)
                    nb_words += 1
            else:
                sent_sep_labels.update({nb_sent: words_sep_labels})
                nb_words = 1
                nb_sent += 1
                words_sep_labels = []
        
        return sent_sep_labels
    
    def has_root(self,decoded_words):
        root = False
        for word in decoded_words:
            node = decoded_words[word]
            if node[5] == 0:
                root = True
        return root

    def find_candidates_for_root(self,decoded_words, headless_words):
        candidates_for_root = {}
        root="root"
        for candidate in decoded_words:
            candidate_for_root = decoded_words[candidate]
            if candidate_for_root[6] == root:
                candidate_for_root[5] = 0
                candidates_for_root.update({candidate: candidate_for_root})

        for candidate in headless_words:
            candidate_for_root = headless_words[candidate]
            if candidate_for_root[6] == root:
                candidate_for_root[5] = 0
                candidates_for_root.update({candidate: candidate_for_root})

        if candidates_for_root:
            for candidate_index in candidates_for_root:
                word = decoded_words[candidate_index]
                word[5] = 0
                if headless_words.__contains__(candidate_index):
                 headless_words.pop(candidate_index)
    
    def assign_root_index_one(self,decoded_words, headless_nodes):
        update_node = decoded_words[1]
        update_node[5] = 0
        update_node[6] = "root"
        if headless_nodes.__contains__(1):
            headless_nodes.pop(1)
    
    def choose_root_from_multiple_candidates(self,decoded_words):
        multiple_roots = {}
        ind = 0
        nb_multiple_roots = 1
        for w in decoded_words:
            if not w == 0:
                ind += 1
                node = decoded_words[w]
                if node[5] == 0:
                    multiple_roots.update({nb_multiple_roots: node})
                    nb_multiple_roots += 1
        chosen_root = multiple_roots[1]
        posit_of_chosen_root = chosen_root[1]
        multiple_roots.pop(1)

        for m in multiple_roots:
            root_dependent = multiple_roots[m]
            word_updated_info = {1: root_dependent[1], 2: root_dependent[2],
                                3: root_dependent[3], 4: root_dependent[4],
                                5: posit_of_chosen_root, 6: root_dependent[6]}
            decoded_words.update({root_dependent[1]: word_updated_info})

    def has_multiple_roots(self, decoded_words):
        count = 0
        multiple_roots = False
        for w in decoded_words:
            node = decoded_words[w]
            if node[5] == 0:
                count += 1
        if count > 1:
            multiple_roots = True
        return multiple_roots

    def assign_headless_words_to_root(self,decoded_words, headless_words):
        root_ind = 0
        for w in decoded_words:
            node = decoded_words[w]
            if node[5] == 0:
                root_ind = node[1]

        for w2 in headless_words:
            node = decoded_words[w2]
            node[5] = root_ind

    def find_cycles(self,decoded_words):
        still_cycle = True
        while still_cycle:
            still_cycle = False
            for ind_word in decoded_words:
                info_about_word = decoded_words[ind_word]
                focus_node_index = int(info_about_word[5])
                seen = [ind_word]
                while not focus_node_index == 0:
                    if not focus_node_index in seen:
                        info_about_word = decoded_words[focus_node_index]
                        seen.append(focus_node_index)
                        focus_node_index = int(info_about_word[5])
                    else:
                        still_cycle = True
                        po = repr(focus_node_index) + " cycle"
                        seen.append(po)
                        second_node_index = seen[-2]
                        ind_root = 0
                        for n in decoded_words:
                            pot_root = decoded_words[n]
                            if pot_root[5] == 0:
                                ind_root = int(pot_root[1])
                        info_about_word = decoded_words[focus_node_index]
                        words_full_info = {1: focus_node_index, 2: info_about_word[2],
                                        3: info_about_word[3], 4: info_about_word[4],
                                        5: ind_root, 6: info_about_word[6]}
                        decoded_words.update({focus_node_index: words_full_info})
                        break
                seen.append(focus_node_index)
    
class CoNLLPostProcessor():

    def write_to_file(self, dict_with_sent, f):
        output_f = open(f, "w")
        for index_sentence in dict_with_sent:
            s = dict_with_sent[index_sentence]
            for index_word in s:
                one_line_word = s[index_word]
                output_f.write(one_line_word)
                output_f.write("\n")
            output_f.write("\n")
        output_f.close()
    
    def convert_labels2conllu(self,decoded_words):
        for w in decoded_words:
            info_word = decoded_words[w]
            line_conllu = str(
                repr(info_word[1]) + "\t" + info_word[2] + "\t" + "_" + "\t"
                + info_word[4] + "\t" + "_" + "\t" + "_" + "\t" + repr(info_word[5])
                + "\t" + info_word[6] + "\t" + "_" + "\t" + "_")
            decoded_words.update({w: line_conllu})

    def dump_into_lookup(self,raw_file_path):
        # only need for UD treebanks that contains special indexes like 1-2 and 1.1
        lookup = {}
        sentence_id = 0
        lookup[sentence_id] = {}
        id_insert_before = 1

        with codecs.open(raw_file_path) as raw_file:
            for line in raw_file:
                if line.startswith('#'):
                    continue
                tok = line.strip().split('\t')
                if not tok or tok == ['']:  # If it is empty line
                    sentence_id += 1
                    id_insert_before = 1
                    lookup[sentence_id] = {}
                else:
                    if "." in tok[0] or "-" in tok[0]:
                        lookup[sentence_id][id_insert_before] = line
                    else:
                        id_insert_before += 1
        return lookup


    def merge_lookup(self,conll_path, lookup):
        # only need for UD treebanks that contains special indexes like 1-2
        sentence_id = 0
        word_id = 1
        with codecs.open(conll_path, "r") as f_conll:
            lines = f_conll.readlines()
        f_conll = codecs.open(conll_path, "w")
        for line in lines:
            tok = line.strip().split('\t')
            if tok == ['']:  # If it is empty line
                sentence_id += 1
                word_id = 1
            else:
                if sentence_id in lookup:
                    if word_id in lookup[sentence_id]:
                        f_conll.write(lookup[sentence_id][word_id])
                word_id += 1
            f_conll.write(line)
        f_conll.close()
    
    def evaluate_dependencies(self, gold, path_output):
        # EVALUATE on conllu
        
        subparser = ArgumentParser()
        subparser.add_argument("gold_file", type=str,
                            help="Name of the CoNLL-U file with the gold data.")
        subparser.add_argument("system_file", type=str,
                            help="Name of the CoNLL-U file with the predicted data.")
        subparser.add_argument("--verbose", "-v", default=False, action="store_true",
                            help="Print all metrics.")
        subparser.add_argument("--counts", "-c", default=False, action="store_true",
                            help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.")

        subargs = subparser.parse_args([gold, path_output])

        # Evaluate
        evaluation = conll18.evaluate_wrapper(subargs)
        uas = 100 * evaluation["UAS"].f1
        las = 100 * evaluation["LAS"].f1

        print(repr(round(uas, 2)) + "\t" + repr(round(las, 2)))
        return las
        """
        #Evaluation for PTB
        output = sp.check_output(
            "perl dep2label/eval_dep/eval-spmrl.pl -p -q -g " + gold + " -s " + path_output, shell=True)

        lines = output.decode().split('\n')

        las = float(lines[0].split()[9])
        uas = float(lines[1].split()[9])
        print("UAS: " + repr(uas) + " LAS: " + repr(las))

        return las
        """

    

    
    
    