#!/bin/bash

# ATTENTION! This automatisation script uses tool telert for automatic notification.
# Read more about this tool you can here: https://github.com/navig-me/telert?tab=readme-ov-file
# If you don't want to use this tool, just rewrite all of the commands without telert using
# For example:
# python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
# python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
# python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn


telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.batch_size=256 model.student=true training.max_epochs=10 onnx.checkpoint_path='./logs/keyword_spotter/version_0/checkpoints/best_epoch\=11_val_acc\=0.91.ckpt'
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.batch_size=256 model.student=true training.max_epochs=15 onnx.checkpoint_path='./logs/keyword_spotter/version_0/checkpoints/best_epoch\=11_val_acc\=0.91.ckpt'
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.batch_size=256 model.student=true training.max_epochs=20 onnx.checkpoint_path='./logs/keyword_spotter/version_0/checkpoints/best_epoch\=11_val_acc\=0.91.ckpt'
sleep 5
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN data.batch_size=256 model.student=true training.max_epochs=25 onnx.checkpoint_path='./logs/keyword_spotter/version_0/checkpoints/best_epoch\=11_val_acc\=0.91.ckpt'