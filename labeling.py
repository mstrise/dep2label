import encoding
import decoding
import pre_post_processing as processing
import codecs


class Encoding:
    """
    Example sentence: Alice ate an apple

    Encoding 1: Naive positional encoding
    label: (2,subj) (0,root) (4,det) (2,obj)

    Encoding 2: Relative positional encoding
    label: (-1,subj)(-2,root) (+1,det) (-2,obj)

    Encoding 3: Relative PoS-based encoding
    label: (V,+1,nsubj) (ROOT,-1,root) (N,+1,det) (V, -1, dobj)

    Encoding 4: Bracketing-based encoding
    label: (_,nsubj) (<\,root) (/,det) (<\>,dobj)

    """

    def __init__(self, nb_enc, uxpostag):

        self.nb_encoding = nb_enc
        self.uxpostag = uxpostag
        self.index_of_sentence = 1
        self.index_to_decode = 1
        self.index_of_word = 1

    def encode(self, gold_file_to_encode_labels):
        """
        :param gold_file_to_encode_labels: gold CONLL-X file to encode with labels:
        :return: encoded_sentences: dictionary with words and their labels,
                all_info_sentences: dictionary with all column information for each word (excluding word, head,
                relation),
                all_text: dictionary with special #text preceding each sentence in CONLLU file
        """
        gold_sentence = {}
        all_info_words = {}
        all_info_sentences = {}
        all_text = {}
        encoded_sentences = {}
        index_of_sentence = 1
        text = []

        with codecs.open(gold_file_to_encode_labels) as to_convert:
            lines = to_convert.readlines()

        for line in lines:
            if line.startswith('#'):
                text.append(line)
            if not line == '\n':
                details_about_a_word_from_gold = line.split("\t")
                if "." in details_about_a_word_from_gold[0] or "-" in details_about_a_word_from_gold[0]:
                    continue
                elif details_about_a_word_from_gold[0].isdigit():
                    index_of_a_word_in_sentence = int(
                        details_about_a_word_from_gold[0])

                    # 1: word, 2: lemma, 3: postag, 4: head, 5: relation
                    if self.uxpostag == "XPOS":
                        words = {
                            1: details_about_a_word_from_gold[1],
                            2: details_about_a_word_from_gold[2],
                            3: details_about_a_word_from_gold[4],
                            4: details_about_a_word_from_gold[6],
                            5: details_about_a_word_from_gold[7]}
                    else:
                        words = {
                            1: details_about_a_word_from_gold[1],
                            2: details_about_a_word_from_gold[2],
                            3: details_about_a_word_from_gold[3],
                            4: details_about_a_word_from_gold[6],
                            5: details_about_a_word_from_gold[7]}
                    gold_sentence.update({index_of_a_word_in_sentence: words})

                    # 1: lemma 2: upos 3: xpos 4: feats 5: deps 6: misc
                    all_columns = {
                        1: details_about_a_word_from_gold[2],
                        2: details_about_a_word_from_gold[3],
                        3: details_about_a_word_from_gold[4],
                        4: details_about_a_word_from_gold[5],
                        5: details_about_a_word_from_gold[8],
                        6: details_about_a_word_from_gold[9]}
                    all_info_words.update(
                        {index_of_a_word_in_sentence: all_columns})
                    all_info_sentences.update(
                        {self.index_of_sentence: all_info_words})

            else:
                # include a dummy root in the dic
                words = {1: "ROOT", 2: "ROOT",
                         3: "ROOT", 4: 0, 5: "root"}
                gold_sentence.update({0: words})
                words_with_labels = {}

                if self.nb_encoding == 0:
                    words_with_labels = encoding.encode_0(gold_sentence)
                elif self.nb_encoding == 1:
                    words_with_labels = encoding.encode_1(gold_sentence)
                elif self.nb_encoding == 2:
                    words_with_labels = encoding.encode_2(gold_sentence)
                elif self.nb_encoding == 3:
                    words_with_labels = encoding.encode_3(gold_sentence)
                elif self.nb_encoding == 4:
                    words_with_labels = encoding.encode_4(gold_sentence)
                else:
                    print("Invalid encoding number", self.nb_encoding)

                all_text.update({self.index_of_sentence: text})
                encoded_sentences.update(
                    {index_of_sentence: words_with_labels})
                index_of_sentence += 1
                gold_sentence = {}
                all_info_words = {}
                text = []
                self.index_of_sentence += 1
        to_convert.close()
        self.index_of_sentence = 1

        return encoded_sentences, all_info_sentences, all_text

    def decode(self, file_with_predicted_labels, encoding_type, all_sent):
        """
        :param file_with_predicted_labels: output file of NCRFpp with the top 3 labels for each word
                with its probabilities
        :param encoding_type: [1,2,3,4]
        :param all_sent: all_info_sentences: dictionary with all column information for each word (excluding word, head,
                relation)
        :return: sent_to_write_to_file: dictionary with lines in CONLL-X format to be written to a file
                nb_words:
        """
        self.nb_encoding = encoding_type
        decoded_sentence = {}
        decoded_sentence2 = {}
        decoded_sentence3 = {}
        all_decoded = {}
        nb_of_sentence = 1
        sent_to_write_to_file = {}
        decoded_words = {}
        homeless_nodes = {}

        with codecs.open(file_with_predicted_labels) as to_decode:
            lines = to_decode.readlines()

        for label_for_each_word in lines:
            if not label_for_each_word == '\n':
                for nb_label in range(1, 4):
                    decoded = {}
                    word_details = {}
                    column = all_sent[self.index_to_decode][self.index_of_word]
                    if self.uxpostag == "XPOS":
                        lemma = column[1]
                        postag = column[3]
                    else:
                        lemma = column[1]
                        postag = column[2]

                    if self.nb_encoding == 3:
                        split_decoder = label_for_each_word.split("\t")
                        token = split_decoder[0]
                        label_proc = split_decoder[nb_label].rsplit("_", 1)
                        proc = label_proc[1]
                        head_label_postag = label_proc[0].rsplit("_", 1)
                        postag_of_head = head_label_postag[1]
                        split_parts = head_label_postag[0].rsplit("_", 1)
                        position_head = split_parts[0]
                        label_of_word = split_parts[1]
                        word_details = {
                            1: self.index_of_word,
                            2: token,
                            3: lemma,
                            4: postag,
                            5: position_head,
                            6: label_of_word,
                            7: proc,
                            8: postag_of_head}
                    else:
                        split_decoder = label_for_each_word.split("\t")
                        token = split_decoder[0]
                        label_proc = split_decoder[nb_label].rsplit("_", 1)
                        proc = label_proc[1]
                        head_rel = label_proc[0].rsplit("_", 1)
                        position_head = head_rel[0]
                        relation = head_rel[1]

                        if self.nb_encoding == 1 or self.nb_encoding == 2:
                            word_details = {
                                1: self.index_of_word,
                                2: token,
                                3: lemma,
                                4: postag,
                                5: int(position_head),
                                6: relation,
                                7: proc}
                        elif self.nb_encoding == 4:
                            word_details = {
                                1: self.index_of_word,
                                2: token,
                                3: lemma,
                                4: postag,
                                5: position_head,
                                6: relation,
                                7: proc}
                    decoded.update({self.index_of_word: word_details})
                    all_decoded.update({nb_label: decoded})

                decoded_sentence.update(all_decoded[1])
                decoded_sentence2.update(all_decoded[2])
                decoded_sentence3.update(all_decoded[3])
                all_decoded = {}
                self.index_of_word += 1

            else:
                word_details = {1: 0, 2: "ROOT", 3: "ROOT",
                                4: "ROOT", 5: 0, 6: "root", 7: "root"}
                decoded_sentence.update({0: word_details})
                decoded_sentence2.update({0: word_details})
                decoded_sentence3.update({0: word_details})
                all_probs = [
                    decoded_sentence,
                    decoded_sentence2,
                    decoded_sentence3]

                # assign a head according to the most probable label
                if self.nb_encoding == 1:
                    decoding.decode_1(
                        decoded_sentence, decoded_words, homeless_nodes)
                elif self.nb_encoding == 2:
                    decoding.decode_2(
                        decoded_sentence, decoded_words, homeless_nodes)
                elif self.nb_encoding == 3:
                    decoding.decode_3(
                        decoded_sentence, decoded_words, homeless_nodes)
                elif self.nb_encoding == 4:
                    decoding.decode_4(
                        decoded_sentence, decoded_words, homeless_nodes)

                # check if a sentences has a root
                # if not, find candidates for root in all probab. Choose the
                # most probable one
                if not processing.check_if_any_root(decoded_words):
                    processing.find_candidates_for_root(
                        all_probs, decoded_words, homeless_nodes)

                # check if a sentence has a root
                # if still not, assign the root to be the first word
                if not processing.check_if_any_root(decoded_words):
                    processing.assign_root_to_index_one(
                        decoded_words, homeless_nodes)

                # check if a sentence has multiple roots
                # choose a root with the highest probability from label 1
                # assign the other roots to the label from probs 2
                # else assign the other candidates to the chosen single root
                if processing.check_if_multiple_roots(decoded_words):
                    if self.nb_encoding == 1:
                        processing.choose_root_from_multiple_enc1_2(
                            decoded_sentence, decoded_sentence2, decoded_words)
                    elif self.nb_encoding == 2:
                        processing.choose_root_from_multiple_enc1_2(
                            decoded_sentence, decoded_sentence2, decoded_words)
                    elif self.nb_encoding == 3:
                        processing.choose_root_from_multiple_enc3(
                            decoded_sentence, decoded_sentence2, decoded_words)
                    elif self.nb_encoding == 4:
                        processing.choose_root_from_multiple_enc4(
                            decoded_sentence, decoded_sentence2, decoded_words)

                # double check if there are no multiple roots. If there are still any,
                # attach them to the single root
                processing.get_rid_of_multiple_roots(decoded_words)

                # attach nodes without heads to the single root
                processing.assign_homeless_nodes_to_root(
                    decoded_words, homeless_nodes)

                # attach first node involved in a cycle to the single root
                processing.find_cycles(decoded_words, decoded_sentence)
                all_columns = all_sent[nb_of_sentence]

                processing.prepare_conll_format(
                    decoded_words, self.uxpostag, all_columns)

                sent_to_write_to_file.update({nb_of_sentence: decoded_words})
                nb_of_sentence += 1
                self.index_to_decode += 1
                self.index_of_word = 1
                decoded_sentence = {}
                decoded_sentence2 = {}
                decoded_sentence3 = {}
                decoded_words = {}
                homeless_nodes = {}

        nb_words = processing.count_nb_words(sent_to_write_to_file)

        self.index_to_decode -= 1
        to_decode.close()
        return sent_to_write_to_file, nb_words
