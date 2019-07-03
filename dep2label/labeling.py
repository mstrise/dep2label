import encoding
import pre_post_processing as processing
import decoding
import codecs


class Encoding:

    def __init__(self):

        self.uxpostag = "UPOS"
        self.index_of_sentence = 1
        self.index_to_decode = 1
        self.index_of_word = 1
        self.dict_roots = {"English":"root", "Basque":"ROOT", "French":"root", "German":"--","Hebrew":"prd", "Hungarian":"ROOT",
                            "Korean":"root", "Polish":"pred", "Swedish":"ROOT"}

    def encode(self, gold_file_to_encode_labels, task):

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
                    index_of_a_word_in_sentence = int(details_about_a_word_from_gold[0])

                    # 1: word, 2: lemma, 3: postag, 4: head, 5: relation
                    if self.uxpostag == "XPOS":
                        words = {1: details_about_a_word_from_gold[1], 2: details_about_a_word_from_gold[2],
                                 3: details_about_a_word_from_gold[4], 4: details_about_a_word_from_gold[6],
                                 5: details_about_a_word_from_gold[7]}
                    else:
                        words = {1: details_about_a_word_from_gold[1], 2: details_about_a_word_from_gold[2],
                                 3: details_about_a_word_from_gold[3], 4: details_about_a_word_from_gold[6],
                                 5: details_about_a_word_from_gold[7]}
                    gold_sentence.update({index_of_a_word_in_sentence: words})

                    # 1: lemma 2: upos 3: xpos 4: feats 5: deps 6: misc
                    all_columns = {1: details_about_a_word_from_gold[2],
                                   2: details_about_a_word_from_gold[3], 3: details_about_a_word_from_gold[4],
                                   4: details_about_a_word_from_gold[5],
                                   5: details_about_a_word_from_gold[8],
                                   6: details_about_a_word_from_gold[9]}
                    all_info_words.update({index_of_a_word_in_sentence: all_columns})
                    all_info_sentences.update({self.index_of_sentence: all_info_words})

            else:
                # include a dummy root in the dic
                words = {1: "ROOT", 2: "ROOT",
                         3: "ROOT", 4: 0, 5: "root"}
                gold_sentence.update({0: words})

                words_with_labels = encoding.encode_3(gold_sentence, task)
                all_text.update({self.index_of_sentence: text})
                encoded_sentences.update({index_of_sentence: words_with_labels})
                index_of_sentence += 1
                gold_sentence = {}
                all_info_words = {}
                text = []
                self.index_of_sentence += 1
        to_convert.close()
        self.index_of_sentence = 1

        return encoded_sentences, all_info_sentences, all_text

    def decode(self, sentences_with_depend, output, language):

        sent_to_write_to_file = {}
        nb_of_sentence = 1
        countNoRoot = 0

        root = self.dict_roots[language]
        for sent_id in sentences_with_depend:
            count = 1;
            sentence_to_decode = sentences_with_depend.get(sent_id)
            sent_with_index = {}
            sent_with_index[0] = ["ROOT", "ROOT", 0, root, "ROOT"]
            for word in sentence_to_decode:
                sent_with_index.update({count:word})
                count+=1

            decoded_words, homeless_nodes = decoding.decode_3(sent_with_index, root)

            #FIND SINGLE ROOT

            if not processing.check_if_any_root(decoded_words):
                processing.find_candidates_for_root(decoded_words,homeless_nodes, root)

            if not processing.check_if_any_root(decoded_words):
               processing.assign_root_to_index_one(decoded_words, homeless_nodes,root)

            if not processing.check_if_any_root(decoded_words):
                countNoRoot += 1

            #CHECK MULTIPLE ROOTS and choose the first one from the candidates

            if processing.check_if_multiple_roots(decoded_words):
                processing.choose_root_from_multiple_enc3(decoded_words)

            processing.assign_homeless_nodes_to_root(decoded_words, homeless_nodes)
            processing.find_cycles(decoded_words)
            processing.prepare_conll_format(decoded_words)

            sent_to_write_to_file.update({nb_of_sentence: decoded_words})
            nb_of_sentence += 1

        processing.write_to_conllu(sent_to_write_to_file, output, 0)


