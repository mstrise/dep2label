import codecs
import argparse


def merge_treebank(enc_cons, enc_dep, mergedFile):
    file_merged = codecs.open(mergedFile, "w")
    with codecs.open(enc_cons, "r") as file_constituency, codecs.open(enc_dep, "r") as file_dependency:
        for line_con, line_dep in zip(file_constituency, file_dependency):
            if not line_con == "\n":
                con = line_con.strip("\n")
                depen = line_dep.strip('\n').split("\t")
                label_depend = depen[2]
                #in case the POS tags differ in cons and depend file, take only the dependency label and merge it with the rest of constituency file
                merged_cons_dep = "{}".join((con, label_depend))
                file_merged.write(merged_cons_dep)
                file_merged.write("\n")
            else:
                file_merged.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoded_file_constituency', dest="enc_constituency")
    parser.add_argument('--encoded_file_dependency',  dest="enc_dependency")
    parser.add_argument('--merged_output', dest="merged_output")

    args = parser.parse_args()
    merge_treebank(args.enc_constituency, args.enc_dependency, args.merged_output)

