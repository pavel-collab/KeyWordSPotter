#!/bin/bash

# ATTENTION! This automatisation script uses tool telert for automatic notification.
# Read more about this tool you can here: https://github.com/navig-me/telert?tab=readme-ov-file
# If you don't want to use this tool, just rewrite all of the commands without telert using
# For example:
# python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
# python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
# python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn

# telert run --label "train model with backbone micro_cnn $(date)" python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
sleep 5
telert run --label "train model with backbone nano_cnn $(date)" python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
sleep 5
telert run --label "train model with backbone tiny_cnn $(date)" python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn
sleep 5
telert run --label "train model with backbone tiny_cnn $(date)" python3 ./train.py data.num_workers=4 model.backbone=baseline
