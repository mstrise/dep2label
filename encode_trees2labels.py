import argparse
from dep2label import labeling
from dep2label import pre_post_processing as processing

def encode(file_to_encode,output,task):
    print("encoding ...")
    train_enc = labeling.Encoding()
    dict_encoded, all_sent, _ = train_enc.encode(file_to_encode, task)
    processing.write_to_conllu(dict_encoded, output, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_to_encode', help='File in CONNL-X format with trees to convert into labels', dest="file_to_encode")
    parser.add_argument('--output', help='File with encoded trees', dest="output")
    parser.add_argument('--task', help='Type of MTL task', dest="task", choices=['single', 'combined', 'multi'])


    args = parser.parse_args()

    encode(args.file_to_encode, args.output, args.task)

#python encode_trees2labels.py --file_to_ecode treebanks/PTB/dev.conll --output myModel/encodedOutput.seq --task single