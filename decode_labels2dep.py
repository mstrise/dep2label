import argparse
from dep2label.labeling import Labeler

def decode_labels(file_to_decode, output, enc, conllu=None):
    l = Labeler()
    l.decode(file_to_decode, output, enc, conllu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='File with labels to convert to CONNL-X format',
                        dest="file2decode",required=True)
    parser.add_argument('--conllu_f', help='Original conllu file', dest="conllu_f",required=False, default=None)
    parser.add_argument('--output', help='File with encoded trees', dest="output",required=True)
    parser.add_argument('--encoding', help='Type of encoding', dest="enc", choices=["rel-pos", "1-planar-brackets", "2-planar-brackets--greedy","2-planar-brackets-propagation",
    "arc-standard", "arc-eager","arc-hybrid", "covington"],required=True)
    args = parser.parse_args()
    decode_labels(args.file2decode, args.output, args.enc, args.conllu_f)
