#!/bin/bash

# ATTENTION! This automatisation script uses tool telert for automatic notification.
# Read more about this tool you can here: https://github.com/navig-me/telert?tab=readme-ov-file
# If you don't want to use this tool, just rewrite all of the commands without telert using
# For example:
# python3 ./train.py data.num_workers=4 model.backbone=micro_cnn
# python3 ./train.py data.num_workers=4 model.backbone=nano_cnn
# python3 ./train.py data.num_workers=4 model.backbone=tiny_cnn

#==============================================MicroCNN==============================================
telert run --label "train model with backbone MicroCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=MicroCNN
sleep 5
#====================================================================================================

#==============================================NanoCNN==============================================
telert run --label "train model with backbone NanoCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=NanoCNN
sleep 5
#====================================================================================================

#==============================================TinyCNN==============================================
telert run --label "train model with backbone TinyCNN $(date)" python3 ./train.py data.num_workers=4 model.backbone=TinyCNN
sleep 5
#====================================================================================================

#==============================================Conv1dNet==============================================
telert run --label "train model with backbone Conv1dNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=Conv1dNet
sleep 5
#====================================================================================================

#==============================================CustomNet==============================================
telert run --label "train model with backbone CustomNet $(date)" python3 ./train.py data.num_workers=4 model.backbone=CustomNet
sleep 5
#====================================================================================================

#==============================================MobileNetV1_1D==============================================
telert run --label "train model with backbone MobileNetV1_1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=MobileNetV1_1D
sleep 5
#====================================================================================================

#==============================================SimpleMobileNet1D==============================================
telert run --label "train model with backbone SimpleMobileNet1D $(date)" python3 ./train.py data.num_workers=4 model.backbone=SimpleMobileNet1D
#====================================================================================================