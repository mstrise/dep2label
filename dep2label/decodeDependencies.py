from labeling import Encoding
import subprocess as sp


def split_label(multitag, multitask_char):
    multitag_dep = multitag.split(multitask_char)

    if len(multitag_dep) > 3:
        return multitag_dep[3:]

    else:
        return multitag_dep[0:3]


def split_label_combined(multitag, multitask_char):
    labels = []

    multitag = multitag.split(multitask_char)

    if len(multitag) > 3:
        multitag_dep = multitag[3:]
        m = multitag_dep[0].split("@")
        labels.append(m[0])
        labels.append(multitag_dep[1])
        labels.append(m[1])
    else:
        m = multitag[0].split("@")
        labels.append(m[0])
        labels.append(multitag[1])
        labels.append(m[1])

    return labels


def decode(output_from_nn, path_output, language):
    labels = []
    words_with_dependencies = []
    sent_with_depend = {}
    sent_nb = 1
    count_words = 1
    enc = Encoding()
    for l in output_from_nn:

        if l != "\n":
            word, postag, label_raw = l.strip().split("\t")[0], l.strip().split("\t")[
                1], l.strip().split("\t")[-1]

            if not word == "-BOS-" and not word == "-EOS-":
                # information about rel.position and word's head (f.ex. +1@V)
                # is combined into one task and dependency relation as the
                # second task
                if "@" in label_raw and "{}" in label_raw:

                    label_dependency = split_label_combined(label_raw, "{}")

                # a label will be treated as single task where its components
                # are separated by a symbol "@"
                elif "@" in label_raw:

                    label_dependency = split_label(label_raw, "@")
                else:
                    # information in the label separated by "{}" will be
                    # treated as a separate task
                    label_dependency = split_label(label_raw, "{}")
                if label_dependency is not None:

                    label_dependency.insert(0, postag)
                    label_dependency.insert(0, word)
                    words_with_dependencies.append(label_dependency)
                    count_words += 1

        else:
            count_words = 1
            sent_with_depend.update({sent_nb: words_with_dependencies})
            sent_nb += 1
            words_with_dependencies = []
    enc.decode(sent_with_depend, path_output, language)


def evaluateDependencies(gold, path_output):

    # EVALUATE on UD
    """
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

    print("UAS: "+repr(uas)+" LAS: "+repr(las))
    return las
    """

    # EVALUATE on PTB or SPMRL
    output = sp.check_output(
        # excluding punctuation -p
        "perl dep2label/eval_dep/eval-spmrl.pl -p -q -g " + gold + " -s " + path_output, shell=True)

    lines = output.split('\n')

    las = float(lines[0].split()[9])
    uas = float(lines[1].split()[9])

    print("UAS: " + repr(uas) + " LAS: " + repr(las))

    return las
