def decode_3(decoded_sentence, root):
    decoded_words = {}
    homeless_nodes = {}
    # 1 : ['The', 'DT', '+1', 'det', 'NN']
    for index_of_word in decoded_sentence:
        #print(index_of_word)
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

        """
        
        if not index_of_word == 0:
            found_head = False
            info_about_word = decoded_sentence[index_of_word]
            position_head = int(info_about_word[5])
            post_of_head = info_about_word[8]
            abs_posit = abs(int(info_about_word[5]))
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
                words_full_info = {1: index_of_word, 2: info_about_word[2], 3: info_about_word[3],
                                   4: info_about_word[4],
                                   5: -1, 6: info_about_word[6]}
                homeless_nodes.update({index_of_word: words_full_info})
                decoded_words.update({index_of_word: words_full_info})
        """
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

def decode_4(decoded_sentence, decoded_words, homeless_nodes):
    stack1 = []
    stack2 = []
    for each_word in range(1, len(decoded_sentence), 1):
        info_about_word = decoded_sentence[each_word]
        words_full_info = {1: each_word, 2: info_about_word[2],
                           3: info_about_word[3],
                           4: info_about_word[4],
                           5: -1,
                           6: info_about_word[6]}
        decoded_words.update({each_word: words_full_info})
        homeless_nodes.update({each_word: words_full_info})

    for index_of_word in range(1, len(decoded_sentence), 1):
        # exclude the dummy root
        if not len(decoded_sentence) - 1 == 1:
            if index_of_word + 1 < len(decoded_sentence):
                proceeding_word = decoded_sentence[index_of_word + 1]
                focus_word = decoded_sentence[index_of_word]
                # get the label from the proceeding_word w(i+1)
                char = list(proceeding_word[5])
                # add which word owns the label based on the index of the word f.ex. "<_2
                # first search for entire labels like "<L" and "R>"
                if focus_word[6] == "root":
                    info_about_word = decoded_sentence[index_of_word]
                    words_full_info = {1: index_of_word, 2: info_about_word[2],
                                       3: info_about_word[3],
                                       4: info_about_word[4],
                                       5: 0,
                                       6: info_about_word[6]}
                    decoded_words.update({index_of_word: words_full_info})
                    if homeless_nodes.__contains__(index_of_word):
                        homeless_nodes.pop(index_of_word)

                if proceeding_word[5] == "<L":
                    # w(i) pointed by w(i+1). Remove w(i)
                    info_about_word = decoded_sentence[index_of_word]
                    words_full_info = {1: index_of_word, 2: info_about_word[2],
                                       3: info_about_word[3],
                                       4: info_about_word[4],
                                       5: int(index_of_word) + 1,
                                       6: info_about_word[6]}
                    decoded_words.update({index_of_word: words_full_info})
                    if homeless_nodes.__contains__(index_of_word):
                        homeless_nodes.pop(index_of_word)

                elif proceeding_word[5] == "R>":
                    # w(i+1) pointed by w(i). Remove w(i+1)
                    proceeding_word[5] = index_of_word
                    info_about_word = decoded_sentence[index_of_word + 1]
                    words_full_info = {1: int(index_of_word) + 1, 2: info_about_word[2],
                                       3: info_about_word[3],
                                       4: info_about_word[4],
                                       5: int(index_of_word),
                                       6: info_about_word[6]}
                    decoded_words.update({int(index_of_word) + 1: words_full_info})
                    if homeless_nodes.__contains__(index_of_word + 1):
                        homeless_nodes.pop(index_of_word + 1)

                else:
                    # every time when you append an item check from the beggining of the list if something matches
                    # the pattern <\ or /> and remove it from the stack
                    for x in char:

                        if x == "<" or x == "L":
                            x += "_" + repr(index_of_word + 1)
                            # elements in the stack1 are concatenated symbols f.ex "<_2"
                            stack1.append(x)
                            if len(stack1) > 1:
                                for w in range(0, len(stack1), 1):
                                    if w + 1 < len(stack1):
                                        s1 = stack1[w].rsplit("_", 1)
                                        s2 = stack1[w + 1].rsplit("_", 1)
                                        symbol = s1[0] + s2[0]
                                        if symbol == "<L":
                                            pos = int(s1[1]) - 1
                                            head = int(s2[1])
                                            info_about_word = decoded_sentence[pos]
                                            words_full_info = {1: pos, 2: info_about_word[2],
                                                               3: info_about_word[3],
                                                               4: info_about_word[4],
                                                               5: head,
                                                               6: info_about_word[6]}
                                            decoded_words.update({pos: words_full_info})
                                            if homeless_nodes.__contains__(pos):
                                                homeless_nodes.pop(pos)

                                            # remove the w and w+1
                                            w1 = stack1[w]
                                            w2 = stack1[w + 1]
                                            stack1.remove(w1)
                                            stack1.remove(w2)
                        if x == ">" or x == "R":
                            x += "_" + repr(index_of_word + 1)
                            stack2.append(x)
                            if len(stack2) > 1:
                                for w in range(0, len(stack2), 1):
                                    if w + 1 < len(stack2):
                                        s1 = stack2[w].rsplit("_", 1)
                                        s2 = stack2[w + 1].rsplit("_", 1)
                                        symbol = s1[0] + s2[0]
                                        if symbol == "R>":
                                            pos = int(s2[1])
                                            head = int(s1[1]) - 1
                                            info_about_word = decoded_sentence[pos]
                                            words_full_info = {1: pos,
                                                               2: info_about_word[2],
                                                               3: info_about_word[3],
                                                               4: info_about_word[4],
                                                               5: head,
                                                               6: info_about_word[6]}
                                            decoded_words.update({pos: words_full_info})
                                            if homeless_nodes.__contains__(pos):
                                                homeless_nodes.pop(pos)
                                            # remove the w and w+1
                                            w1 = stack2[w]
                                            w2 = stack2[w + 1]
                                            stack2.remove(w1)
                                            stack2.remove(w2)