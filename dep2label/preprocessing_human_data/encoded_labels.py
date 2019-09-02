import sys
sys.path.append("..")
import pre_post_processing as processing
import labeling as l


def get_encoded_labels(gold, file_with_encoded_labels, task, pos):
    """

    :type task: "single"- all tuples treated as single task +1@obj@N
                "combined"- rel.head and dep. relation treated as one task +1@obj{}N
                "multi"- all tuples as independent tasks +1{}obj{}N
    """
    label = l.Encoding()
    dict_encoded, all_sent, text = label.encode(gold, task, pos)
    processing.write_to_conllu(dict_encoded, file_with_encoded_labels, 0)


get_encoded_labels("PTB/dev.conll","dev-encoded.seq", "combined","UPOS")