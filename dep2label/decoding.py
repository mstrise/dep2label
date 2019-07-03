def decode_3(decoded_sentence, root):
    decoded_words = {}
    homeless_nodes = {}
    # 1 : ['The', 'DT', '+1', 'det', 'NN']
    for index_of_word in decoded_sentence:
        if not index_of_word == 0:

            word_line = decoded_sentence.get(index_of_word)
            info_about_word = word_line
            found_head = False
            if not word_line[2]=="-EOS-" and not word_line[2]=="-BOS-":

                position_head = int(word_line[2])
                post_of_head = word_line[4]
                abs_posit = abs(position_head)
                abs_posit_minus = abs_posit - 1
                abs_posit_plus = abs_posit + 1

                # find head with the relative position -1,-2....
                if position_head < 0:
                    words_full_info = assignHeadL(index_of_word, info_about_word, post_of_head,
                                                  decoded_sentence, abs_posit)
                    if words_full_info:
                        decoded_words.update({index_of_word: words_full_info})
                        found_head = True

                    elif not abs_posit_minus == 0:
                        words_full_info = assignHeadL(index_of_word, info_about_word, post_of_head,
                                                      decoded_sentence, abs_posit_minus)
                        if words_full_info:
                            decoded_words.update({index_of_word: words_full_info})
                            found_head = True

                    else:
                        words_full_info = assignHeadL(index_of_word, info_about_word, post_of_head,
                                                      decoded_sentence, abs_posit_plus)
                        if words_full_info:
                            found_head = True
                            decoded_words.update({index_of_word: words_full_info})

                # find head with the relative position +1,+2....
                elif position_head > 0:
                    found_head = False
                    words_full_info = assignHeadR(index_of_word, info_about_word, post_of_head,
                                                  decoded_sentence, abs_posit)
                    if words_full_info:
                        decoded_words.update({index_of_word: words_full_info})
                        found_head = True

                    elif not abs_posit_minus == 0:
                        words_full_info = assignHeadR(index_of_word, info_about_word, post_of_head,
                                                      decoded_sentence, abs_posit_minus)
                        if words_full_info:
                            decoded_words.update({index_of_word: words_full_info})
                            found_head = True

                    else:
                        words_full_info = assignHeadR(index_of_word, info_about_word, post_of_head,
                                                      decoded_sentence, abs_posit_plus)
                        if words_full_info:
                            decoded_words.update({index_of_word: words_full_info})
                            found_head = True

                if not found_head:
                    words_full_info = {1: index_of_word, 2: info_about_word[0],
                                       3: "_", 4: info_about_word[1],
                                       5: -1, 6: info_about_word[3]}
                    homeless_nodes.update({index_of_word: words_full_info})
                    decoded_words.update({index_of_word: words_full_info})

            else:
                words_full_info = {1: index_of_word, 2: info_about_word[0],
                                       3: "_", 4: info_about_word[1],
                                       5: -1, 6: root}
                homeless_nodes.update({index_of_word: words_full_info})
                decoded_words.update({index_of_word: words_full_info})

    return decoded_words, homeless_nodes

def assignHeadL(node_index, info_about_word, head, decoded, abs_posit):
    # assign new head to the wrong roots

    count_posit = 0

    # find head with the relative position -1,-2....
    for index in range(node_index - 1, -1, -1):
            info_candidate_word = decoded[index]
            postag_candidate = info_candidate_word[1]
            if postag_candidate == head:
                count_posit += 1
                if abs_posit == count_posit:
                    words_full_info = {1: node_index, 2: info_about_word[0],
                                       3: "_", 4: info_about_word[1],
                                       5: index, 6: info_about_word[3]}

                    return words_full_info


def assignHeadR(node_index, info_about_word, head, decoded, abs_posit):
    count_posit = 0

    # find head with the relative position +1,+2....
    for index in range(node_index + 1, len(decoded)):
        info_candidate_word = decoded[index]

        postag_candidate = info_candidate_word[1]
        if postag_candidate == head:
            count_posit += 1
            if abs_posit == count_posit:
                words_full_info = {1: node_index, 2: info_about_word[0],
                                   3: "_", 4: info_about_word[1],
                                   5: index, 6: info_about_word[3]}
                return words_full_info
