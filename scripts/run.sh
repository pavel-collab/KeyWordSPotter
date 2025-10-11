#!/bin/bash

# ATTENTION! This automatisation script uses tool telert for automatic notification.
# Read more about this tool you can here: https://github.com/navig-me/telert?tab=readme-ov-file
# If you don't want to use this tool, just rewrite all of the commands without telert using
# For example:
# python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
# python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
# python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn


telert run --label "train model with backbone MicroCNN $(date)" python3 train.py data.batch_size=128 training.max_epochs=15 model.backbone=MicroCNN
sleep 5
telert run --label "train model with backbone MicroCNN with augmented data $(date)" python3 train.py data.batch_size=128 data.train_manifest_name=manifest_train_augmented.csv training.max_epochs=15 model.backbone=MicroCNN