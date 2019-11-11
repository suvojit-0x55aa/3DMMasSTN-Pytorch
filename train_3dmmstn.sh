#!/bin/bash
set -e

WRK_DIR=/home/ubuntu/3DMMasSTN-Pytorch

# Where the training (fine-tuned) checkpoint and logs will be saved to.
CKPT_DIR=$WRK_DIR/weights

# Where the dataset is saved to.
DATA_DIR=$WRK_DIR/data/aflw_processed_data
DATA_CSV=$WRK_DIR/data/aflw_cropped_label.csv

# Pretrained Model Paths
TUTTE_EMB_PATH=$WRK_DIR/models/model.mat
VGG_FACES_WEIGHT=$WRK_DIR/models/vgg_face_dag.pth

python train_3dmmstn.py \
    --learning_rate=1e-10 \
    --batch_size=32 \
    --max_nb_epochs=1000 \
    --gpus=1 \
    --checkpoint_path=$CKPT_DIR \
    --tutte_emb_path=$TUTTE_EMB_PATH \
    --vgg_faces_path=$VGG_FACES_WEIGHT \
    --dataset_root=$DATA_DIR \
    --dataset_csv=$DATA_CSV \
    --worker=8
    # --dev_run