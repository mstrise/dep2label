import codecs
import operator
import decoding


def dump_into_lookup(raw_file_path):
    lookup = {}
    sentence_id = 0
    lookup[sentence_id] = {}
    id_insert_before = 1

    with codecs.open(raw_file_path) as raw_file:
        for line in raw_file:

            if line.startswith('#'):
                id_insert_before += 1
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


def merge_lookup(conll_path, lookup):
    sentence_id = 0
    word_id = 1

    with codecs.open(conll_path) as f_conll:
        lines = f_conll.readlines()

    # DUMPING the content of the file
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


def write_to_conllu(dict_with_sent, final_file, all_text):
    final_conllu = codecs.open(final_file, "w")
    for index_sentence in dict_with_sent:
        if not all_text == 0:
            line_text = all_text[index_sentence]
            if line_text:
                for n in line_text:
                    final_conllu.write(n)

        s = dict_with_sent[index_sentence]
        for index_word in s:
            one_line_word = s[index_word]
            final_conllu.write(one_line_word)
            final_conllu.write("\n")

        final_conllu.write("\n")
    final_conllu.close()


def check_if_any_root(decoded_words):
    has_sentence_any_root = False
    for d in decoded_words:
        node = decoded_words[d]
        if node[5] == 0:
            has_sentence_any_root = True
    return has_sentence_any_root


def check_if_multiple_roots(decoded_words):
    count = 0
    has_multiple_roots = False
    for d in decoded_words:
        node = decoded_words[d]
        if node[5] == 0:
            count += 1
    if count > 1:
        has_multiple_roots = True
    return has_multiple_roots


def find_candidates_for_root(all_probs, decoded_words, homeless_nodes):
    candidates_for_root = {}

    for d in all_probs[1:]:
        ind = 0
        if not candidates_for_root:
            for w in d:
                if not w == 0:
                    ind += 1
                    candidate = d[w]
                    if candidate[6] == "root":
                        candidates_for_root.update({float(candidate[7]): ind})

    # choose the root bsed in their probabilities from the second label
    if candidates_for_root:
        candidate_that_won = max(candidates_for_root, key=float)
        candidate_index = candidates_for_root[candidate_that_won]
        change_root = decoded_words[candidate_index]
        change_root[5] = 0
        change_root[6] = "root"
        if homeless_nodes.__contains__(candidate_index):
            homeless_nodes.pop(candidate_index)


def assign_root_to_index_one(decoded_words, homeless_nodes):
    update_node = decoded_words[1]
    update_node[5] = 0
    update_node[6] = "root"
    if homeless_nodes.__contains__(1):
        homeless_nodes.pop(1)


def choose_root_from_multiple_enc1_2(
        decoded_sentence,
        decoded_sentence2,
        decoded_words):
    count = 0

    for d in decoded_words:
        node = decoded_words[d]
        if int(node[5]) == 0:
            count += 1
    most_probably = {}
    ind = 0
    if count > 1:
        for w in decoded_sentence:
            if not w == 0:
                ind += 1
                node = decoded_sentence[w]
                if node[6] == "root":
                    proc = float(node[7])
                    most_probably.update({ind: proc})

    if most_probably:
        highest = max(most_probably.iteritems(), key=operator.itemgetter(1))[0]
        most_probably.pop(highest)
        # assign other to their label from prob 2
        for nodes in most_probably:
            if not nodes == highest or len(most_probably) == 1:
                info_about_word = decoded_sentence2[nodes]
                position_head = int(info_about_word[5])
                if len(decoded_sentence2) > position_head > 0:
                    words_full_info = {
                        1: nodes,
                        2: info_about_word[2],
                        3: info_about_word[3],
                        4: info_about_word[4],
                        5: position_head,
                        6: info_about_word[6]}
                    decoded_words.update({nodes: words_full_info})
                else:
                    words_full_info = {
                        1: nodes,
                        2: info_about_word[2],
                        3: info_about_word[3],
                        4: info_about_word[4],
                        5: highest,
                        6: info_about_word[6]}
                    decoded_words.update({nodes: words_full_info})
            else:
                words_full_info = {
                    1: nodes,
                    2: info_about_word[2],
                    3: info_about_word[3],
                    4: info_about_word[4],
                    5: highest,
                    6: info_about_word[6]}
                decoded_words.update({nodes: words_full_info})


def choose_root_from_multiple_enc3(
        decoded_sentence,
        decoded_sentence2,
        decoded_words):
    most_probably = {}
    ind = 0
    for w in decoded_sentence:
        if not w == 0:
            ind += 1
            node = decoded_sentence[w]
            if node[6] == "root":
                proc = float(node[7])
                most_probably.update({ind: proc})

    if most_probably:
        highest = max(most_probably.iteritems(), key=operator.itemgetter(1))[0]
        most_probably.pop(highest)

        # assign other to their label from prob 2

        for nodes in most_probably:
            if not nodes == highest or len(most_probably) == 1:
                info_about_word = decoded_sentence2[nodes]
                position_head = int(info_about_word[5])
                post_of_head = info_about_word[8]
                abs_posit = abs(int(info_about_word[5]))

                # assign new head to the wrong roots
                if position_head < 0:
                    found_head = False
                    words_full_info = decoding.assignHeadL(
                        nodes, info_about_word, post_of_head, decoded_sentence2, abs_posit)
                    # find head with the relative position -1,-2....
                    if words_full_info:
                        decoded_words.update({nodes: words_full_info})
                        found_head = True
                elif position_head > 0:
                    found_head = False
                    words_full_info = decoding.assignHeadR(
                        nodes, info_about_word, post_of_head, decoded_sentence2, abs_posit)

                    # find head with the relative position +1,+2....
                    if words_full_info:
                        decoded_words.update({nodes: words_full_info})
                        found_head = True

                # attach to the ROOT
                if not found_head:
                    words_full_info = {
                        1: nodes,
                        2: info_about_word[2],
                        3: info_about_word[3],
                        4: info_about_word[4],
                        5: highest,
                        6: info_about_word[6]}
                    decoded_words.update({nodes: words_full_info})


def choose_root_from_multiple_enc4(
        decoded_sentence,
        decoded_sentence2,
        decoded_words):
    count = 0
    for d in decoded_words:
        node = decoded_words[d]
        if int(node[5]) == 0:
            count += 1
    most_probably = {}
    ind = 0
    if count > 1:
        for w in decoded_sentence:
            if not w == 0:
                ind += 1
                node = decoded_sentence[w]
                if node[6] == "root":
                    proc = float(node[7])
                    most_probably.update({ind: proc})

    if most_probably:
        highest = max(most_probably.iteritems(), key=operator.itemgetter(1))[0]
        most_probably.pop(highest)
        info_about_word = decoded_sentence[highest]
        words_full_info = {1: highest, 2: info_about_word[2],
                           3: info_about_word[3], 4: info_about_word[4],
                           5: 0, 6: info_about_word[6]}
        decoded_words.update({highest: words_full_info})
        # assign other to their label from prob 2
        for nodes in most_probably:
            info_about_word2 = decoded_sentence2[nodes]
            words_full_info = {1: nodes, 2: info_about_word2[2],
                               3: info_about_word2[3], 4: info_about_word2[4],
                               5: highest, 6: info_about_word2[6]}
            decoded_words.update({nodes: words_full_info})


def get_rid_of_multiple_roots(decoded_words):
    count = 0
    while count > 1:
        count = 0
        for d in decoded_words:
            node = decoded_words[d]
            if node[5] == 0:
                count += 1
        posit_root = 0
        while count > 1:
            count = 0
            for d in decoded_words:
                node = decoded_words[d]
                if node[5] == 0 and count == 0:
                    count += 1
                    posit_root = node[1]
                elif node[5] == 0:
                    node[5] = posit_root
                    count += 1


def assign_homeless_nodes_to_root(decoded_words, homeless_nodes):
    root_ind = 0
    for d in decoded_words:
        node = decoded_words[d]
        if node[5] == 0:
            root_ind = node[1]

    for t in homeless_nodes:
        n = decoded_words[t]
        n[5] = root_ind


def find_cycles(decoded_words, decoded_sentence):
    more_cycles = True

    while more_cycles:
        more_cycles = False
        for ind_word in decoded_words:
            info_about_word = decoded_words[ind_word]
            focus_node_index = int(info_about_word[5])
            seen = [ind_word]
            while not focus_node_index == 0:
                if focus_node_index not in seen:
                    info_about_word = decoded_words[focus_node_index]
                    seen.append(focus_node_index)
                    focus_node_index = int(info_about_word[5])
                else:
                    more_cycles = True
                    po = repr(focus_node_index) + " cycle"
                    seen.append(po)
                    second_node_index = seen[-2]
                    ind_root = 0
                    for n in decoded_words:
                        pot_root = decoded_words[n]
                        if pot_root[5] == 0:
                            ind_root = int(pot_root[1])

                    info_about_word = decoded_sentence[focus_node_index]
                    words_full_info = {
                        1: focus_node_index,
                        2: info_about_word[2],
                        3: info_about_word[3],
                        4: info_about_word[4],
                        5: ind_root,
                        6: info_about_word[6]}
                    decoded_words.update({focus_node_index: words_full_info})
                    break
            seen.append(focus_node_index)


def prepare_conll_format(decoded_words, uxpostag, all_columns):
    for w in decoded_words:
        info_about_word = decoded_words[w]
        dict_columns = all_columns[w]
        # 1: lemma 2: upos 3: xpos 4: feats 5: deps 6: misc
        # print(dict_columns)

        if uxpostag == "UPOS":
            line_conllu = str(
                repr(
                    info_about_word[1]) +
                "\t" +
                info_about_word[2] +
                "\t" +
                dict_columns[1] +
                "\t" +
                info_about_word[4] +
                "\t" +
                dict_columns[3] +
                "\t" +
                dict_columns[4] +
                "\t" +
                repr(
                    info_about_word[5]) +
                "\t" +
                info_about_word[6] +
                "\t" +
                dict_columns[5] +
                "\t" +
                dict_columns[6].rstrip())
        else:

            line_conllu = str(
                repr(
                    info_about_word[1]) +
                "\t" +
                info_about_word[2] +
                "\t" +
                dict_columns[1] +
                "\t" +
                dict_columns[2] +
                "\t" +
                info_about_word[4] +
                "\t" +
                dict_columns[4] +
                "\t" +
                repr(
                    info_about_word[5]) +
                "\t" +
                info_about_word[6] +
                "\t" +
                dict_columns[5] +
                "\t" +
                dict_columns[6].rstrip())

        decoded_words.update({w: line_conllu})


def count_nb_words(sent_to_write_to_file):
    nb_words = 0
    for index in sent_to_write_to_file:
        w = len(sent_to_write_to_file[index])
        nb_words += w
    return nb_words
