#!/usr/bin/env bash

if [ ${PWD##*/} == "scripts" ]; then
    cd ..
else
    :
fi

BASE_DIR="."
MODEL_NAME="model_efficient_net.pt_0.992_9"
MODEL_NAME_PRETTY="EfficientNet"

python -u ${BASE_DIR}/src/predict_log_mlflow.py \
		--batch_size 64 \
        --num_workers 8 \
		--data_folder ${BASE_DIR}/data \
		--checkpoint_folder ${BASE_DIR}/models/model_checkpoint \
		--log_folder ${BASE_DIR}/log \
		--model_name ${MODEL_NAME}.t7 \
		--model_name_pretty ${MODEL_NAME_PRETTY} \
        --mean 0.485 0.456 0.406 \
        --std 0.229 0.224 0.225

exit 0