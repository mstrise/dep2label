from argparse import ArgumentParser
import codecs
import os
import time
import sys
import uuid
STATUS_TEST = "test"
STATUS_TRAIN = "train"
from dep2label import decode_dependencies


if __name__ == '__main__':
     
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test",help="Path to the input test file as sequences", default=None)
    arg_parser.add_argument("--goldDependency", dest="goldDependency",
                            help="Path to the original linearized trees, without preprocessing")
    arg_parser.add_argument("--model", dest="model", help="Path to the model")
    arg_parser.add_argument("--status", dest="status", help="[train|test]")
    arg_parser.add_argument("--gpu",dest="gpu",default="False")
    arg_parser.add_argument("--multitask", dest="multitask", default=False, action="store_true")
    arg_parser.add_argument("--outputDependency", dest="outputDependency", default="/tmp/trees.txt", required=False)
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository")
    arg_parser.add_argument("--offset", dest="offset", default="False")
    args = arg_parser.parse_args()

    #If not, it gives problem with Chinese chracters
    reload(sys)
    sys.setdefaultencoding('UTF8')
    path_raw_dir = args.test
    path_name = args.model
    path_output = "/tmp/" + path_name.split("/")[-1] + ".output"
    path_tagger_log = "/tmp/" + path_name.split("/")[-1] + ".tagger.log"
    path_dset = path_name + ".dset"
    path_model = path_name + ".model"

    conf_str = """
        ### Decode ###
        status=decode
        """
    conf_str += "raw_dir=" + path_raw_dir + "\n"
    conf_str += "decode_dir=" + path_output + "\n"
    conf_str += "dset_dir=" + path_dset + "\n"
    conf_str += "load_model_dir=" + path_model + "\n"
    conf_str += "gpu=" + args.gpu + "\n"

    decode_fid = str(uuid.uuid4())
    decode_conf_file = codecs.open("/tmp/" + decode_fid, "w")
    decode_conf_file.write(conf_str)
    os.system("python " + args.ncrfpp + "/main.py --config " + decode_conf_file.name + " > " + path_tagger_log)

    log_lines = codecs.open(path_tagger_log).readlines()
    time_prediction = float([l for l in log_lines
                             if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:", "").replace("s", ""))
    output_content = codecs.open(path_output)

    if args.multitask:
        split_char = "{}"
        if args.offset == "True":
            start = time.time()
            decode_dependencies.decode_combined_tasks(output_content, args.outputDependency, split_char)
            time_dependency = time.time() - start
        else:
            start = time.time()
            decode_dependencies.decode(output_content, args.outputDependency, split_char)
            time_dependency = time.time() - start
        decode_dependencies.evaluate_dependencies(args.goldDependency, args.outputDependency)
    else:
        split_char = "@"
        start = time.time()
        decode_dependencies.decode(output_content, args.outputDependency, split_char)
        time_dependency = time.time() - start
        f = open(args.outputDependency, "r")
        decode_dependencies.evaluate_dependencies(args.goldDependency, args.outputDependency)
