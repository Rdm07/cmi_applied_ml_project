#!/usr/bin/env bash

if [ ${PWD##*/} == "scripts" ]; then
    cd ..
else
    :
fi

BASE_DIR="."
MODEL_NAME="model1"
LOG_FOLDER=${BASE_DIR}/models/training_logs
LOG_FILE_NAME=${LOG_FOLDER}/log_training_${MODEL_NAME}.txt

python -u ${BASE_DIR}/src/train_new_cuda.py \
        --lr 1e-2 \
        --weight_decay 1e-4 \
		--trainer "sgd" \
		--batch_size 64 \
        --num_workers 8 \
        --num_epochs 25 \
        --check_after 1 \
		--checkpoint ${BASE_DIR}/models/model_checkpoint \
		--data_folder ${BASE_DIR}/data \
		--model_folder ${BASE_DIR}/models/saved_models \
		--model_name ${MODEL_NAME}.pt | tee ${LOG_FILE_NAME}

exit 0