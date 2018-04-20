from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils import *

import numpy as np
import time
import argparse, sys, itertools, datetime
import json
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # selects a specific device

if "__main__" in __name__:
    parser = argparse.ArgumentParser(prog="test")
    parser.add_argument('--FP32', action='store_true')
    parser.add_argument('--FP16', action='store_true')
    parser.add_argument('--INT8', action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--dump_diff', action='store_true')
    parser.add_argument('--with_timeline', action='store_true')
    parser.add_argument('--update_graphdef', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--input_node', type=str, default='input')
    parser.add_argument('--output_node', type=str, default='resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workspace_size', type=int, default=1 << 10, help="workspace size in MB")
    parser.add_argument('--num_loops', type=int, default=20)
    parser.add_argument('--topN', type=int, default=10)

    args, unparsed = parser.parse_known_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)  # selects a specific device
    
    frozen_graph_file = 'resnet_v1_50_frozen.pb'
    # Although networks can use NHWC and NCHW, TensorFlow users are encouraged to
    # convert their networks to use NCHW data ordering explicitly
    # in order to achieve the best possible performance.
    # TensorRT's C++ API input and output tensors are in NCHW format.
    batch_size = args.batch_size
    input_height = args.image_size
    input_width = args.image_size
    input_channel = 3
    input_node = args.input_node
    output_node = args.output_node
    num_loops = args.num_loops

    valnative = None
    valfp32 = None
    valfp16 = None
    valint8 = None
    res = [None, None, None, None]
    print("Starting at", datetime.datetime.now())
    if args.update_graphdef:
        updateGraphDef(frozen_graph_file)
    dummy_input = np.random.random_sample((batch_size, 224, 224, 3))

    with open("labellist.json", "r") as lf:
        labels = json.load(lf)

    image_file = "grace_hopper.jpg"
    t = read_tensor_from_image_file(image_file,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=0,
                                    input_std=1.0)
    tshape = list(t[0].shape)
    tshape[0] = batch_size
    tnhwcbatch = np.tile(t[0], (batch_size, 1, 1, 1))
    dummy_input = tnhwcbatch
    workspace_size = args.workspace_size << 20
    timeline_file = None
    if args.native:
        if args.with_timeline: timeline_file = "timeline/NativeTimeline.json"
        timings, comp, valnative, mdstats = timeGraph(
            getResnet50(frozen_graph_file),
            batch_size,
            num_loops,
            dummy_input,
            timeline_file)
        print('=' * 40)
        print('mdstats: ' + str(mdstats))
        printStats("Native", timings, batch_size)
        printStats("NativeRS", mdstats, batch_size)
        print('=' * 40)

    if args.FP32:
        if args.with_timeline: timeline_file = "timeline/FP32Timeline.json"
        timings, comp, valfp32, mdstats = timeGraph(
            getFP32(frozen_graph_file, batch_size, workspace_size),
            batch_size,
            num_loops,
            dummy_input,
            timeline_file)
        print('=' * 40)
        print('mdstats: ' + str(mdstats))
        printStats("TRT-FP32", timings, batch_size)
        printStats("TRT-FP32RS", mdstats, batch_size)
        print('=' * 40)

    if args.FP16:
        if args.with_timeline: timeline_file = "timeline/FP16Timeline.json"
        timings, comp, valfp16, mdstats = timeGraph(
            getFP16(frozen_graph_file, batch_size, workspace_size),
            batch_size,
            num_loops,
            dummy_input,
            timeline_file)
        print('=' * 40)
        print('mdstats: ' + str(mdstats))
        printStats("TRT-FP16", timings, batch_size)
        printStats("TRT-FP16RS", mdstats, batch_size)
        print('=' * 40)

    if args.INT8:
        calibGraph = getINT8CalibGraph(frozen_graph_file, batch_size, workspace_size)
        print("Running Calibration")
        timings, comp, _, mdstats = timeGraph(
            calibGraph,
            batch_size,
            1,
            dummy_input)
        print('=' * 40)
        print("Creating inference graph")
        int8Graph = getINT8InferenceGraph(calibGraph)
        del calibGraph
        if args.with_timeline: timeline_file = "timeline/INT8Timeline.json"
        timings, comp, valint8, mdstats = timeGraph(
            int8Graph,
            batch_size,
            num_loops,
            dummy_input,
            timeline_file)
        print('=' * 40)
        print('mdstats: ' + str(mdstats))
        printStats("TRT-INT8", timings, batch_size)
        printStats("TRT-INT8RS", mdstats, batch_size)
        print('=' * 40)
    vals = [valnative, valfp32, valfp16, valint8]
    enabled = [(args.native, "native", valnative),
               (args.FP32, "FP32", valfp32),
               (args.FP16, "FP16", valfp16),
               (args.INT8, "INT8", valint8)]
    print("Done timing", datetime.datetime.now())
    for i in enabled:
        if i[0]:
            print(i[1], getLabels(labels, topX(i[2], args.topN)[1][0]))

    sys.exit(0)
