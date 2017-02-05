# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts KITTI data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'image_2'. Similarly, bounding box annotations are supposed to be
stored in the 'label_2'

This TensorFlow script converts the training and validation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing PNG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'PNG'

    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import os.path
import sys
import random

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.kitti_common import KITTI_LABELS

DEFAULT_IMAGE_DIR = 'image_2/'
DEFAULT_LABEL_DIR = 'label_2/'


def _png_image_shape(image_data, sess, decoded_png, inputs):
    rimg = sess.run(decoded_png, feed_dict={inputs: image_data})
    return rimg.shape


def _process_image(directory, name, f_png_image_shape,
                   image_dir=DEFAULT_IMAGE_DIR, label_dir=DEFAULT_LABEL_DIR):
    """Process a image and annotation file.

    Args:
      directory: KITTI dataset directory;
      name: file name.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the PNG image file.
    filename = os.path.join(directory, image_dir, name + '.png')
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    shape = list(f_png_image_shape(image_data))

    # Get object annotations.
    labels = []
    labels_text = []
    truncated = []
    occluded = []
    alpha = []
    bboxes = []
    dimensions = []
    locations = []
    rotation_y = []

    # Read the txt label file, if it exists.
    filename = os.path.join(directory, label_dir, name + '.txt')
    if os.path.exists(filename):
        with open(filename) as f:
            label_data = f.readlines()
        for l in label_data:
            data = l.split()
            if len(data) > 0:
                # Label.
                labels.append(int(KITTI_LABELS[data[0]][0]))
                labels_text.append(data[0].encode('ascii'))
                # truncated, occluded and alpha.
                truncated.append(float(data[1]))
                occluded.append(int(data[2]))
                alpha.append(float(data[3]))
                # bbox.
                bboxes.append((float(data[4]) / shape[1],
                               float(data[5]) / shape[0],
                               float(data[6]) / shape[1],
                               float(data[7]) / shape[0]
                               ))
                # 3D dimensions.
                dimensions.append((float(data[8]),
                                   float(data[9]),
                                   float(data[10])
                                   ))
                # 3D location and rotation_y.
                locations.append((float(data[11]),
                                  float(data[12]),
                                  float(data[13])
                                  ))
                rotation_y.append(float(data[14]))

    return (image_data, shape, labels, labels_text, truncated, occluded,
            alpha, bboxes, dimensions, locations, rotation_y)


def _convert_to_example(image_data, shape, labels, labels_text,
                        truncated, occluded, alpha, bboxes,
                        dimensions, locations, rotation_y):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    # Transpose bboxes, dimensions and locations.
    bboxes = list(map(list, zip(*bboxes)))
    dimensions = list(map(list, zip(*dimensions)))
    locations = list(map(list, zip(*locations)))
    # Iterators.
    it_bboxes = iter(bboxes)
    it_dims = iter(dimensions)
    its_locs = iter(locations)

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'object/label': int64_feature(labels),
            'object/label_text': bytes_feature(labels_text),
            'object/truncated': float_feature(truncated),
            'object/occluded': int64_feature(occluded),
            'object/alpha': float_feature(alpha),
            'object/bbox/xmin': float_feature(next(it_bboxes, [])),
            'object/bbox/ymin': float_feature(next(it_bboxes, [])),
            'object/bbox/xmax': float_feature(next(it_bboxes, [])),
            'object/bbox/ymax': float_feature(next(it_bboxes, [])),
            'object/dimensions/height': float_feature(next(it_dims, [])),
            'object/dimensions/width': float_feature(next(it_dims, [])),
            'object/dimensions/length': float_feature(next(it_dims, [])),
            'object/location/x': float_feature(next(its_locs, [])),
            'object/location/y': float_feature(next(its_locs, [])),
            'object/location/z': float_feature(next(its_locs, [])),
            'object/rotation_y': float_feature(rotation_y),
            }))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer, f_png_image_shape,
                     image_dir=DEFAULT_IMAGE_DIR, label_dir=DEFAULT_LABEL_DIR):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    l_data = _process_image(dataset_dir, name, f_png_image_shape,
                            image_dir, label_dir)
    example = _convert_to_example(*l_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)


def run(dataset_dir, output_dir, name='kitti_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    tf_filename = _get_output_filename(output_dir, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        # return
    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DEFAULT_IMAGE_DIR)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(12345)
        random.shuffle(filenames)

    # PNG decoding.
    inputs = tf.placeholder(dtype=tf.string)
    decoded_png = tf.image.decode_png(inputs)
    with tf.Session() as sess:

        # Process dataset files.
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                name = filename[:-4]
                _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                 lambda x: _png_image_shape(x, sess, decoded_png, inputs))
        print('\nFinished converting the KITTI dataset!')
