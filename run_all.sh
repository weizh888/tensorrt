#!/bin/bash

# image_size should use the same value as the one during the model training.

python3 main.py --native --FP32 --FP16 --INT8 \
                --image_file 01.jpg \
                --image_size 224 \
                --network resnet_v1_50_frozen.pb \
                --input_node input \
                --output_node resnet_v1_50/predictions/Reshape_1 \
                --num_loops 1 \
                --topN 5 \
                --cuda_device 1 \
                --batch_size 1 \
                --workspace_size 2048
