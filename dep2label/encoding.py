def encode_3(gold_sentence, task):
    words_with_labels = {}

    bos ="-BOS-"
    if task == "single":
        label = str(bos)
    elif task == "combined":
        label = str(
            bos + "@" + bos + "{}" + bos)
    else:
        label = str(
            bos + "{}" + bos + "{}" + bos)
    full_label = str(bos + "\t" + bos + "\t" + label)
    words_with_labels.update({0: full_label})

    for index_of_word in gold_sentence:
        # exclude the dummy root
        if not index_of_word == 0:
            info_of_a_word = gold_sentence[index_of_word]
            # head is on the right side from the word
            head = int(info_of_a_word[4])
            if index_of_word < head:
                relative_position_head = 1
                info_about_head = gold_sentence[head]
                postag_head = info_about_head[3]

                for x in range(index_of_word + 1, head):
                    another_word = gold_sentence[x]
                    postag_word_before_head = another_word[3]
                    if postag_word_before_head == postag_head:
                        relative_position_head += 1
                if task == "single":
                    label = str(
                        "+" + repr(relative_position_head) + "@" + info_of_a_word[5] + "@" + postag_head)
                elif task == "combined":
                    label = str(
                        "+" + repr(relative_position_head) + "@" + postag_head + "{}" + info_of_a_word[5])
                else:
                    label = str(
                        "+" + repr(relative_position_head) + "{}" + info_of_a_word[5] + "{}" + postag_head)
                full_label = str(
                    info_of_a_word[1] + "\t" + info_of_a_word[3] + "\t" + label)

                words_with_labels.update({index_of_word: full_label})

            # head is on the left side from the word
            elif index_of_word > head:
                relative_position_head = 1
                info_about_head = gold_sentence[head]
                postag_head = info_about_head[3]
                for x in range(head + 1, index_of_word):
                    another_word = gold_sentence[x]
                    postag_word_before_head = another_word[3]
                    if postag_word_before_head == postag_head:
                        relative_position_head += 1
                if task == "single":
                    label = str(
                        "-" + repr(relative_position_head) + "@" + info_of_a_word[5] + "@" + postag_head)
                elif task == "combined":
                    label = str(
                        "-" + repr(relative_position_head) + "@" + postag_head + "{}" + info_of_a_word[5])
                else:
                    label = str(
                        "-" + repr(relative_position_head) + "{}" + info_of_a_word[5] + "{}" + postag_head)
                full_label = str(
                    info_of_a_word[1] + "\t" + info_of_a_word[3] + "\t" + label)

                words_with_labels.update({index_of_word: full_label})

    eos = "-EOS-"
    if task == "single":
        label = str(eos)
    elif task == "combined":
        label = str(
            eos + "@" + eos + "{}" + eos)
    else:
        label = str(
            eos + "{}" + eos + "{}" + eos)

    full_label = str(eos + "\t" + eos + "\t" + label)
    words_with_labels.update({len(words_with_labels) + 1: full_label})

    return words_with_labels


