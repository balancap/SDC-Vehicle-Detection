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
"""KITTI dataset.
"""
import os

import tensorflow as tf

from datasets import dataset_utils
from datasets.kitti_common import KITTI_LABELS, NUM_CLASSES, KITTI_DONTCARE

slim = tf.contrib.slim

FILE_PATTERN = 'kitti_%s.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
SPLITS_TO_SIZES = {
    'train': 7481,
    'test': 7518,
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'object/label': tf.VarLenFeature(dtype=tf.int64),
        'object/truncated': tf.VarLenFeature(dtype=tf.float32),
        'object/occluded': tf.VarLenFeature(dtype=tf.int64),
        'object/alpha': tf.VarLenFeature(dtype=tf.float32),
        'object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'object/dimensions/height': tf.VarLenFeature(dtype=tf.float32),
        'object/dimensions/width': tf.VarLenFeature(dtype=tf.float32),
        'object/dimensions/length': tf.VarLenFeature(dtype=tf.float32),
        'object/location/x': tf.VarLenFeature(dtype=tf.float32),
        'object/location/y': tf.VarLenFeature(dtype=tf.float32),
        'object/location/z': tf.VarLenFeature(dtype=tf.float32),
        'object/rotation_y': tf.VarLenFeature(dtype=tf.float32),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('object/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=SPLITS_TO_SIZES[split_name],
            items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
            num_classes=NUM_CLASSES,
            labels_to_names=labels_to_names)
