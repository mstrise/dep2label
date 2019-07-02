#Update this path to your virtual environment
HOME=/home/david.vilares/
source $HOME/env/proof/bin/activate

HOME_NCRFpp=$HOME/Escritorio/CPH/NCRFpp_multitask1.5

TEST_NAME="test"
INPUT=$HOME_NCRFpp/sample_data/cp_datasets/ptb/ptb-$TEST_NAME.seq_lu
#INPUT=$HOME/Escritorio/dataset/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=$HOME_NCRFpp/sample_data/cp_datasets/PTB_pred_tags/$TEST_NAME.trees
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=$HOME_NCRFpp/outputs/ptb/
MODELS=$HOME_NCRFpp/parsing_models/ptb/
NCRFPP=$HOME_NCRFpp/
LOGS=$HOME_NCRFpp/logs/ptb/
#MULTITASK=True




################################################
#								BASIC MODELS
################################################


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output /tmp/proof.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP # > $LOGS/ptb.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1

: '
taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.emnlp2018.f.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


###################################################
#							 MULTI-TASK MODELS
###################################################


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


################################################
#   				+ AUXILIARY TASKS
################################################

#NEXT X

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.nex_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.nex_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.nex_lab.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 


#PREV X

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.pre_lab.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1 


#COMBINATIONS PREV/NEXT x


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lev.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lev.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.pre_lev.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lab.nex_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lab.nex_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP  > $LOGS/ptb.multitask.3R.-2.pre_lab.nex_lab.$TEST_NAME.$USE_GPU.log 2>&1 


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lev.pre_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lev.pre_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.pre_lev.pre_lab.$TEST_NAME.$USE_GPU.log 2>&1 


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.nex_lev.nex_lab \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.nex_lev.nex_lab.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.nex_lev.nex_lab.$TEST_NAME.$USE_GPU.log 2>&1 

#DISTANCES

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.dis.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.dis.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.multitask.3R.-2.dis.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



#BEST WITHOUT CHAR EMBEDDINGS


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.noch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.noch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP > $LOGS/ptb.noch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


#######################################
# REINFORCEMENT LEARNING
#######################################

taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $HOME_NCRFpp/parsing_models/ptb-RL/ptb.RL.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP  > $LOGS/ptb.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $HOME_NCRFpp/parsing_models/ptb-RL/ptb.RL.noise.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.RL.noise.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP > $LOGS/ptb.RL.noise.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../decode.py \
--test $INPUT \
--gold $TEST_PATH \
--model $HOME_NCRFpp/parsing_models/ptb-RL/ptb.noch.RL.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.noch.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP  > $LOGS/ptb.noch.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


'
