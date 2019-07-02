HOME=/home/david.vilares/
#Update this path to your virtual environment
source $HOME/env/cons2labels/bin/activate

HOME_NCRFpp=$HOME/Escritorio/CPH/NCRFpp_multitask1.5/
TEST_NAME="test"
INPUT=$HOME_NCRFpp/sample_data/cp_datasets/ctb/ctb-$TEST_NAME.seq_lu
#INPUT=$HOME/Escritorio/dataset/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=$HOME_NCRFpp/sample_data/cp_datasets/CTB_pred_tags/$TEST_NAME"_ch.trees"
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=$HOME_NCRFpp/outputs/ctb/
MODELS=$HOME_NCRFpp/parsing_models/ctb/
NCRFPP=$HOME_NCRFpp/
LOGS=$HOME_NCRFpp/logs/ctb/
#MULTITASK=True

: '


################################################
#								BASIC MODELS
################################################


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.emnlp2018.f.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


###################################################
#							 MULTI-TASK MODELS
###################################################


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1





################################################
#   				+ AUXILIARY TASKS
################################################

#NEXT X

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.nex_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.nex_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.nex_lab.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 



#PREV X

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.pre_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.pre_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.pre_lab.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1 


#COMBINATIONS PREV/NEXT x


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.pre_lev.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.pre_lev.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.pre_lev.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.pre_lab.nex_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.pre_lab.nex_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP  > $LOGS/ctb.multitask.3R.-2.pre_lab.nex_lab.$TEST_NAME.$USE_GPU.log 2>&1 


#DISTANCES

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.dis.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.3R.-2.dis.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.multitask.3R.-2.dis.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.noch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.noch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP  > $LOGS/ctb.noch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $HOME_NCRFpp/parsing_models/ctb-RL/ctb.ch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP > $LOGS/ctb.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



'



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $HOME_NCRFpp/parsing_models/ctb-RL/ctb.noch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP > $LOGS/ctb.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1

