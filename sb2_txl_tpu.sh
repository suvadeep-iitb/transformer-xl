#!/bin/bash

# Data
DATA_ROOT=~/Workspace/data/simplebooks-2
WARM_START=False

# Model
DIV_VAL=1
D_MODEL=256
D_EMBED=256
N_HEAD=4
D_HEAD=64
D_INNER=1024
DROPOUT=0.1
DROPATT=0.1
DROPPATH=0.0
N_LAYER=8
PRE_LNORM=True

# Training
TGT_LEN=4
MEM_LEN=4
BSZ=8
EVAL_BSZ=8
NUM_CORE=1

# Testing
TEST_TGT_LEN=4
TEST_MEM_LEN=4
TEST_CLAMP_LEN=-1
TEST_BSZ=8
TEST_NUM_CORE=1

# Optimization
LR_RATE=0.00025
LR_MIN=0.004
WU_STEPS=0
TRAIN_STEPS=20
SAVE_STEPS=10
ITERS=10
MAX_EVAL_BCH=10
CLIP=0.25
UNTIE_R=False

#Param initialization
INIT_RANGE=0.1
INIT_STD=0.02
PROJ_INIT_STD=0.01




if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=sb2 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${EVAL_BSZ} \
        --num_passes=1 \
        --use_tpu=True \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=sb2 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_passes=1 \
        --use_tpu=True \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_xl.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --checkpoint_dir=${DATA_ROOT}/sb2 \
	--warm_start=${WARM_START} \
        --div_val=${DIV_VAL} \
        --untie_r=${UNTIE_R} \
	--n_layer=${N_LAYER} \
	--d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
	--pre_lnorm=${PRE_LNORM} \
	--init_range=${INIT_RANGE} \
        --init_std=${INIT_STD} \
	--proj_init_std=${PROJ_INIT_STD} \
        --learning_rate=${LR_RATE} \
	--min_lr_ratio=${LR_MIN} \
	--clip=${CLIP} \
        --warmup_steps=${WU_STEPS} \
        --train_steps=${TRAIN_STEPS} \
        --tgt_len=${TGT_LEN} \
	--mem_len=${MEM_LEN} \
	--test_tgt_len=${TEST_TGT_LEN} \
	--test_mem_len=${TEST_MEM_LEN} \
        --train_batch_size=${BSZ} \
	--eval_batch_size=${EVAL_BSZ} \
	--test_batch_size=${TEST_BSZ} \
	--max_eval_batch=${MAX_EVAL_BCH} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=${ITERS} \
        --save_steps=${SAVE_STEPS} \
	--do_eval=True \
	--do_test=True \
	--use_tpu=True \
        ${@:2}
elif [[ $1 == 'test' ]]; then
    echo 'Run evaluation...'
    python train_xl.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --checkpoint_dir=${DATA_ROOT}/sb2 \
	--warm_start=${WARM_START} \
        --div_val=${DIV_VAL} \
        --untie_r=${UNTIE_R} \
	--n_layer=${N_LAYER} \
	--d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
	--pre_lnorm=${PRE_LNORM} \
	--init_range=${INIT_RANGE} \
        --init_std=${INIT_STD} \
	--proj_init_std=${PROJ_INIT_STD} \
        --learning_rate=${LR_RATE} \
	--min_lr_ratio=${LR_MIN} \
	--clip=${CLIP} \
        --warmup_steps=${WU_STEPS} \
        --train_steps=${TRAIN_STEPS} \
        --tgt_len=${TGT_LEN} \
	--mem_len=${MEM_LEN} \
	--test_tgt_len=${TEST_TGT_LEN} \
	--test_mem_len=${TEST_MEM_LEN} \
        --train_batch_size=${BSZ} \
	--eval_batch_size=${EVAL_BSZ} \
	--test_batch_size=${TEST_BSZ} \
	--max_eval_batch=${MAX_EVAL_BCH} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=${ITERS} \
        --save_steps=${SAVE_STEPS} \
	--do_train=False \
	--do_test=True \
	--use_tpu=True
        ${@:2}
else
    echo 'unknown argment 1'
fi
