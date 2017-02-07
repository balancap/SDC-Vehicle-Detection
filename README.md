# Udacity SDC: Vehicle Detection

The goad of this project is to implement a robust pipeline capable of detecting moving vehicles in real-time. Even though the project was designed for using classic Computer Vision techniques, namely HOG features and SVM classifier, in agreement the course organizers, I decided like a few other students to go for a deep learning approach.

Several important papers on object detection using deep convolutional networks have been published the last few years. More specifically, [Faster R-CNN](https://arxiv.org/abs/1506.01497), [YOLO](https://arxiv.org/abs/1612.08242) and [Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) are the present state-of-the-art in using CNN for real-time object detection.

Even though there are a few differences between the three previous approaches, they share the same general pipeline. Namely, the detection network is designed based on the following rules:
* Use a deep convolutional network trained on ImageNet as a multi-scale source of features. Typically, VGG, ResNet or Inception;
* Provide a collection of pre-defined *anchors boxes* tiling the image at different positions and scales. They serve the same purpose as the sliding window approach in classic CV detection algorithms;
* For every anchor box, the modified CNN provides a probability for every class of object (and a no detection probability), and offsets (x, y, width and height) between the detected box and the associated anchor box.
* The detection output of the network is post-processed using a Non-Maximum Selection algorithm, in order to remove overlapping boxes.

For this project, I decided to implement the SSD detector, as the later provides a good compromise between accuracy and speed (note that the last YOLOv2 article describes in fact a SSD-like network).

# SSD: Single Shot MultiBox Detector for vehicle detection

The author of the original SSD research paper had implemented [SSD using the framework Caffe](https://github.com/weiliu89/caffe/tree/ssd). As I could not find any satisfying TensorFlow implementation of the former, I decided to write my own from scratch. This task was more time-consuming than I had originally thought, but also allowed me to learn how to properly write a large TensorFlow pipeline, from TFRecords to TensorBoard! I left my pure SSD port in a different [GitHub repository](https://github.com/balancap/SSD-Tensorflow), and modified it for this vehicle detection project.

## SSD architecture

As previously outlined, the SSD network used the concept of anchor boxes for object detection. The image below illustrates the concept: at several scales are pre-defined boxes with different sizes and ratios. The goal of SSD convolutional network is, for each of these anchor boxes, to detect if there is an object inside this box (or closely), and compute the offset between the object bounding box and the fixed anchor box.

![](pictures/ssd_anchors.png "SSD anchors")

In the case of SSD network, we use VGG as a based architecture: it provides high quality features at different scales, the former being then used as inputs for *multibox* modules in charge of computing the object type and coordinates for each anchor boxes. The architecture of the network we use is illustrated in the following TensorBoard graph. It follows the original SSD paper:
* Convolutional Blocks 1 to 7 are exactly VGG modules. Hence, these weights can be imported from VGG weights, speeding massively training time;
* Blocks 8 to 11 are additional feature blocks. They consist of two convolutional layers each: a 3x3 convolution followed by a 1x1 convolution;
* Yellow blocks are *multibox* modules: they take VGG-type features as inputs, and outputs two components: a softmax Tensor which gives for every anchor box the probability of an object being detected, and an offset Tensor which describes the offset between the object bounding box and the anchor box.

![](pictures/ssd_network.png "SSD Network")


# SSD: Single Shot MultiBox Detector in TensorFlow

SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). TF checkpoints are directly converted from SSD Caffe models.

The organisation is inspired by the TF-Slim models repository which contains the implementation of popular architectures (ResNet, Inception and VGG). Hence, it is separated in three main parts:
* datasets: interface to popular datasets (Pascal VOC, COCO, ...) and scripts to convert the former to TF-Records;
* networks: definition of SSD networks, and common encoding and decoding methods (we refer to the paper on this precise topic);
* pre-processing: pre-processing routines, inspired by original VGG and Inception implementation.

## Minimal example

Here is a very simple and minimal example showing how to use the SSD network on some demo images.
```python
import os
import matplotlib.image as mpimg
import tensorflow as tf
slim = tf.contrib.slim

from nets import ssd_vgg_300, ssd_common
from preprocessing import ssd_preprocessing

isess = tf.InteractiveSession()
# Input placeholder.
net_shape = (300, 300)
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, resize=ssd_preprocessing.Resize.PAD_AND_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the model
ssd = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False)

# SSD default anchor boxes.
layers_anchors = ssd.anchors(net_shape)

# Restore SSD model.
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, './checkpoints/ssd_300_vgg.ckpt')

# Main processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=0.35, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes, rlayers, ridxes = ssd_common.ssd_bboxes_select(
            rpredictions, rlocalisations, layers_anchors,
            threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = ssd_common.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = ssd_common.bboxes_sort(rclasses, rscores, rbboxes)
    rclasses, rscores, rbboxes = ssd_common.bboxes_nms(rclasses, rscores, rbboxes, threshold=nms_threshold)
    # Resize bboxes to original image shape.
    rbboxes = ssd_common.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# Test on some demo image.
path = './demo/'
image_names = sorted(os.listdir(path))
img = mpimg.imread(path + image_names[0])
rclasses, rscores, rbboxes =  process_image(img)
```

## Training

In order to train a new architecture, or fine tune and existing on, we first need to convert a dataset into TFRecords. For instance, in the case of Pascal VOC dataset:
```bash
DATASET_DIR=./VOC2007/test/
OUTPUT_DIR=./tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
```

The script `train_ssd_network.py` is then in charged of training the network. Similarly to TF-Slim models, one can pass numerous options to the training (dataset, optimiser, hyper-parameters, model, ...). In particular, it is possible to provide a checkpoint file which can be use as starting point for fine tuning a network.
```bash
DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/ssd_300_vgg
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=32
```

## ToDo
Several important parts have not been implemented yet:
* proper pre-processing for training;
* recall / precision / mAP computation on a test dataset.
