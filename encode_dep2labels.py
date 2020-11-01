import argparse
from dep2label.labeling import Labeler

def encode_labels(file_to_encode, output, enc, mtl=None):

    l = Labeler()
    l.encode(file_to_encode, output, enc, mtl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='File in CONNL-X format with trees to convert into labels',
                        dest="file2encode")
    parser.add_argument('--output', help='File with encoded trees', dest="output")
    parser.add_argument('--encoding', help='Type of encoding', dest="enc", choices=["rel-pos", "1-planar-brackets", "2-planar-brackets--greedy","2-planar-brackets-propagation",
    "arc-standard", "arc-eager","arc-hybrid", "covington","zero"])
    parser.add_argument('--mtl', help='Type of Multi-task', dest="mtl",choices=["1-task","2-task","2-task-combined","3-task"],required=False, default=None)
    args = parser.parse_args()
    encode_labels(args.file2encode, args.output, args.enc, args.mtl)
