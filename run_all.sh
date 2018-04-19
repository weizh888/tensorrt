#!/bin/bash

python3 main.py --native --FP32 --FP16 --INT8 \
                        --num_loops 20 \
                        --topN 5 \
                        --batch_size 4 \
                        --workspace_size 2048 \
                        --log_file log.txt \
                        --network resnet_v1_50_frozen.pb \
                        --input_node input \
                        --output_nodes resnet_v1_50/predictions/Reshape_1 \
                        --image_size 300 \
                        --image_file  grace_hopper.jpg