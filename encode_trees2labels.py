import argparse
import labeling as labeling
import pre_post_processing as processing

def encode(file_to_encode,output,encoding,pos):
    print("encoding ...")
    train_enc = labeling.Encoding(int(encoding), pos)
    dict_encoded, all_sent, _ = train_enc.encode(file_to_encode)
    processing.write_to_conllu(dict_encoded, output, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileToEncode', help='File in CONNL-X format with trees to convert into labels', dest="file_to_encode")
    parser.add_argument('--output', help='File with encoded trees', dest="output")
    parser.add_argument('--encoding', help='Type of encoding', dest="encoding")
    parser.add_argument('--pos', help='Type of Pos', dest="pos")

    args = parser.parse_args()

    encode(args.file_to_encode, args.output, args.encoding, args.pos)

#python encode_trees2labels.py --fileToEncode PTB/dev.conll --output testModel/encodedOutput.seq --encoding 3 --pos UPOS
