# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# !/bin/env python -tt
r""" TF-TensorRT integration sample script """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

import numpy as np
import time
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
import argparse, sys, itertools, datetime
import json
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # selects a specific device
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_tensor_from_image_file(filename, input_height, input_width,
                                input_mean, input_std):
    """ Read a jpg image file and return a tensor """
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(filename, input_name)
    image_reader = tf.image.decode_png(file_reader, channels=3,
                                       name='jpg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))
    sess = tf.Session(config=config)

    # Convert NHWC to NCHW
    result = sess.run([normalized, tf.transpose(normalized, perm=(0, 3, 1, 2))])
    del sess

    return result


def getSimpleGraphDef():
    """Create a simple graph and return its graph_def"""
    if gfile.Exists("origgraph"):
        gfile.DeleteRecursively("origgraph")
    g = tf.Graph()
    with g.as_default():
        A = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name="input")
        e = tf.constant(
            [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.], [1., 1., 1., 1., 1., 1.]]]],
            name="weights",
            dtype=tf.float32)
        conv = tf.nn.conv2d(
            input=A, filter=e, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], padding="SAME", name="conv")
        b = tf.constant([4., 1.5, 2., 3., 5., 7.], name="bias", dtype=tf.float32)
        t = tf.nn.bias_add(conv, b, name="biasAdd")
        relu = tf.nn.relu(t, "relu")
        idty = tf.identity(relu, "ID")
        v = tf.nn.max_pool(
            idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
        out = tf.squeeze(v, name="resnet_v1_50/predictions/Reshape_1")
        writer = tf.summary.FileWriter("origgraph", g)
        writer.close()

    return g.as_graph_def()


def updateGraphDef(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(graph_def, name="")
        with gfile.FastGFile(filename, 'wb') as f:
            f.write(g.as_graph_def().SerializeToString())


def get_GraphDef(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def get_trt_graph(filename, batch_size, workspace_size, precision, output_pb):
    print('Start to optimize graph')
    trt_graph = trt.create_inference_graph(
        get_GraphDef(filename),
        ["resnet_v1_50/predictions/Reshape_1"],
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=precision)  # Get optimized graph

    with gfile.FastGFile(output_pb, 'wb') as f:
        f.write(trt_graph.SerializeToString())
    return trt_graph


def get_int8_infer_graph(calib_graph, output_pb):
    trt_graph = trt.calib_graph_to_infer_graph(calib_graph)
    with gfile.FastGFile(output_pb, 'wb') as f:
        f.write(trt_graph.SerializeToString())
    return trt_graph


def printStats(graph_name, timings, batch_size):
    if timings is None:
        return
    times = np.array(timings)
    speeds = batch_size / times
    avg_time = np.mean(timings)
    avg_speed = batch_size / avg_time
    std_time = np.std(timings)
    std_speed = np.std(speeds)

    print('Results:')
    print('Graph Name: %s' % graph_name)
    print('Batch Size: %d' % batch_size)
    print('Avg Speed: %.2f +/- %.2f (images/sec)' % (avg_speed, std_speed))
    print('Avg Time: %.5f +/- %.5f (sec/batch)' % (avg_time, std_time))


def timeGraph(gdef, batch_size, num_loops, dummy_input=None, timeline_file=None):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    tf.logging.info("Starting execution")
    tf.reset_default_graph()
    g = tf.Graph()
    if dummy_input is None:
        dummy_input = np.random.random_sample((batch_size, 224, 224, 3))
    outlist = []
    with g.as_default():
        inc = tf.constant(dummy_input, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensors(inc)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        out = tf.import_graph_def(
            graph_def=gdef,
            input_map={"input": next_element},
            return_elements=["resnet_v1_50/predictions/Reshape_1"]
        )
        out = out[0].outputs[0]
        outlist.append(out)

    timings = []

    with tf.Session(graph=g, config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        tf.logging.info("Starting Warmup cycle")

        def mergeTraceStr(mdarr):
            tl = timeline.Timeline(mdarr[0][0].step_stats)
            ctf = tl.generate_chrome_trace_format()
            Gtf = json.loads(ctf)
            deltat = mdarr[0][1][1]
            for md in mdarr[1:]:
                tl = timeline.Timeline(md[0].step_stats)
                ctf = tl.generate_chrome_trace_format()
                tmp = json.loads(ctf)
                deltat = 0
                Gtf["traceEvents"].extend(tmp["traceEvents"])
                deltat = md[1][1]

            return json.dumps(Gtf, indent=2)

        rmArr = [[tf.RunMetadata(), 0] for x in range(20)]
        if timeline_file:
            if gfile.Exists(timeline_file):
                gfile.Remove(timeline_file)
            ttot = int(0)
            tend = time.time()
            for i in range(20):
                tstart = time.time()
                valt = sess.run(outlist, options=run_options, run_metadata=rmArr[i][0])
                tend = time.time()
                rmArr[i][1] = (int(tstart * 1.e6), int(tend * 1.e6))
            with gfile.FastGFile(timeline_file, "a") as tlf:
                tlf.write(mergeTraceStr(rmArr))
        else:
            for i in range(20):
                valt = sess.run(outlist)
        tf.logging.info("Warmup done. Starting real timing")

        num_iters = 50
        for i in range(num_loops):
            tstart = time.time()
            for k in range(num_iters):
                val = sess.run(outlist)
            # print([max(val[0][i]) for i in range(4)])
            timings.append((time.time() - tstart) / float(num_iters))
            print('iter %2d: %.6f s' % (i + 1, timings[-1]))
        comp = sess.run(tf.reduce_all(tf.equal(val[0], valt[0])))
        print("Comparison = %s" % comp)
        sess.close()
        tf.logging.info("Timing loop done!")
        return timings, comp, val[0], None


def score(nat, trt, topN=5):
    ind = np.argsort(nat)[:, -topN:]
    tind = np.argsort(trt)[:, -topN:]
    return np.array_equal(ind, tind), howClose(nat, trt, topN)


def topX(arr, X):
    ind = np.argsort(arr)[:, -X:][:, ::-1]
    return arr[np.arange(np.shape(arr)[0])[:, np.newaxis], ind], ind


def howClose(arr1, arr2, X):
    val1, ind1 = topX(arr1, X)
    val2, ind2 = topX(arr2, X)
    ssum = 0.
    for i in range(X):
        in1 = ind1[0]
        in2 = ind2[0]
        if (in1[i] == in2[i]):
            ssum += 1
        else:
            pos = np.where(in2 == in1[i])
            pos = pos[0]
            if pos.shape[0]:
                if np.abs(pos[0] - i) < 2:
                    ssum += 0.5
    return ssum / X


def getLabels(labels, ids):
    return [labels[str(x + 1)] for x in ids]


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("Created folder: %s" % directory)
        