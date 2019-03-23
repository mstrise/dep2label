
def encode_0(gold_sentence):
    words_with_labels = {}
    for index_of_word in gold_sentence:
        # exclude the dummy root
        if not index_of_word == 0:
            info_of_a_word = gold_sentence[index_of_word]

            full_label = str(
                info_of_a_word[1] +
                "\t" +
                "[POStag]" +
                info_of_a_word[3] +
                "\t" +
                "0")
            words_with_labels.update({index_of_word: full_label})
    return words_with_labels


def encode_1(gold_sentence):
    words_with_labels = {}
    for index_of_word in gold_sentence:
        # exclude the dummy root
        if not index_of_word == 0:
            info_of_a_word = gold_sentence[index_of_word]
            head = int(info_of_a_word[4])
            label = str(repr(head) + "_" + info_of_a_word[5])
            full_label = str(
                info_of_a_word[1] +"\t" +
                "[POStag]" +
                info_of_a_word[3] +
                "\t" +
                label)
            words_with_labels.update({index_of_word: full_label})

    return words_with_labels


def encode_2(gold_sentence):
    words_with_labels = {}
    for index_of_word in gold_sentence:
        # exclude the dummy root
        if not index_of_word == 0:
            info_of_a_word = gold_sentence[index_of_word]
            head = int(info_of_a_word[4])

            if index_of_word < head:
                add = head - index_of_word
                label = str("+" + repr(add) + "_" + info_of_a_word[5])
                full_label = str(
                    info_of_a_word[1] + "\t" +
                    "[POStag]" +
                    info_of_a_word[3] +
                    "\t"  +
                    label)
                words_with_labels.update({index_of_word: full_label})
            else:
                minus = head - index_of_word
                label = str(repr(minus) + "_" + info_of_a_word[5])
                full_label = str(
                    info_of_a_word[1] +"\t"
                    "[POStag]" +
                    info_of_a_word[3] +
                    "\t" +
                    label)
                words_with_labels.update({index_of_word: full_label})
    return words_with_labels


def encode_3(gold_sentence):
    words_with_labels = {}
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
                label = str(
                    "+" +
                    repr(relative_position_head) +
                    "_" +
                    info_of_a_word[5] +
                    "_" +
                    postag_head)
                full_label = str(
                    info_of_a_word[1] +
                    "\t" +
                    "[POStag]" +
                    info_of_a_word[3] +
                    "\t" +
                    label)
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
                label = str(
                    "-" +
                    repr(relative_position_head) +
                    "_" +
                    info_of_a_word[5] +
                    "_" +
                    postag_head)
                full_label = str(
                    info_of_a_word[1] +
                    "\t" +
                    "[POStag]" +
                    info_of_a_word[3] +
                    "\t" +
                    label)
                words_with_labels.update({index_of_word: full_label})
    return words_with_labels


def encode_4(gold_sentence):

    words_with_labels = {}
    for index_of_word in range(1, len(gold_sentence), 1):
        # exclude the dummy root

        info_of_a_word = gold_sentence[index_of_word]
        if len(gold_sentence) - 1 == 1:
            label = str("._" + info_of_a_word[5])
            full_label = str(
                info_of_a_word[1] +
                "\t" +
                "[POStag]" +
                info_of_a_word[3] +
                "\t" +
                label)
            words_with_labels.update({index_of_word: full_label})
            break

        elif index_of_word + 1 < len(gold_sentence):
            proceeding_word = gold_sentence[index_of_word + 1]
            head = int(info_of_a_word[4])
            brackets = ""
            head_proc = int(proceeding_word[4])

            if index_of_word == 1 and info_of_a_word[5] == "root":
                label = str("._" + info_of_a_word[5])
                full_label = str(
                    info_of_a_word[1] +
                    "\t" +
                    "[POStag]" +
                    info_of_a_word[3] +
                    "\t" +
                    label)
                words_with_labels.update({index_of_word: full_label})

            # searching for symbols "<" and "/"
            # check if w(i) has an incoming arc coming from the right
            if head > index_of_word:
                if index_of_word == 1:
                    label = str("._" + info_of_a_word[5])
                    full_label = str(
                        info_of_a_word[1] +
                        "\t" +
                        "[POStag]" +
                        info_of_a_word[3] +
                        "\t" +
                        label)
                    words_with_labels.update({index_of_word: full_label})
                    brackets += "<"
                else:
                    brackets += "<"

            # check if w(i) has k outgoing arcs towards the right.

            for i in range(index_of_word + 1, len(gold_sentence)):

                check_words = gold_sentence[i]
                head_of_words = int(check_words[4])
                if head_of_words == index_of_word:
                    brackets += "R"

            # searching for symbols "\" and ">"

            for i in range(1, index_of_word + 1):
                check_words = gold_sentence[i]
                head_of_words = int(check_words[4])
                if head_of_words == index_of_word + 1:
                    brackets += 'L'

            # check if w(i+1) has an incoming arc coming from the left ">"
            if head_proc < index_of_word + 1 and not head_proc == 0:
                brackets += ">"

                # print(proceeding_word[1] + "> " + proceeding_word[5])
                # check if w(i+1) has k outgoing arcs towards the left "\"
            if not brackets == "":
                label = str(brackets + "_" + proceeding_word[5])
                full_label = str(
                    proceeding_word[1] +
                    "\t" +
                    "[POStag]" +
                    proceeding_word[3] +
                    "\t" +
                    label)
                words_with_labels.update({index_of_word + 1: full_label})
            else:
                label = str("._" + proceeding_word[5])
                full_label = str(
                    proceeding_word[1] +
                    "\t" +
                    "[POStag]" +
                    proceeding_word[3] +
                    "\t" +
                    label)
                words_with_labels.update({index_of_word + 1: full_label})
    return words_with_labels
