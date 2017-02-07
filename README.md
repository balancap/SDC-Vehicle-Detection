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
* Yellow blocks are *multibox* modules: they take VGG-type features as inputs, and outputs two components: a softmax Tensor which gives for every anchor box the probability of an object being detected, and an offset Tensor which describes the offset between the object bounding box and the anchor box. These two Tensors are the results of two different 3x3 convolutions of the input Tensor.

For instance, consider the 8x8 feature block described in the image above. At every coordinate in the grid, it defines 4 anchor boxes of different dimensions. The *multibox* module taking this feature Tensor as input will thus provide two output Tensors: a classification Tensor of shape 8x8x4xNClasses and an offset Tensor of shape 8x8x4x4, where in the latter, the last dimension stands for the 4 coordinates of every bounding box.

As a result, the global SSD network will provide a classification score and an offset for a total of 8732 anchor boxes. During training, we therefore try to minimize both errors: the classification error on every anchor box and the localization error when there is a positive match with a grountruth bounding box. We refer to the original SSD paper for the precise equations defining the loss function.

![](pictures/ssd_network.png "SSD Network")

## SSD in TensorFlow

Porting the SSD network to TensorFlow has been a worthy but ambitious project on its own! Designing a robust pipeline in TensorFlow requires quite a bit of engineering, and debugging, especially in the case of object detection networks.
For this SSD pipeline, I took inspiration from the implementation of common deep CNNs in TF-Slim (https://github.com/tensorflow/models/tree/master/slim). Basically, the pipeline is divided into three main components (and directories):
* ```datasets```: the Python source files implement the interface for different dataset, and describe how to convert the original raw data into *TFRecords* files. In our case, as we use the KITTI dataset, the file ```kitti_to_tfrecords.py``` performs this convertion, and the files ```kitti.py``` and ```kitti_common.py``` implements the interface, in the form of TF-Slim dataset object. Note that we also left the source files describing the Pascal VOC dataset, in case we would like to combine the latter with the KITTI dataset.
* ```preprocessing```: this directory contains the implementation of the pre-processing before training (or evaluation). More specifically, we described our pipeline in the file ```ssd_vgg_preprocessing.py```. During training, our pre-processing pipeline performs three different important random transformations: 
    * random cropping: a zone containing objects is randomly generated and used for cropping the input image;
    * random flipping: the image is randomly left-right flipped;
    * random color distortions: we randomly distort the saturation, hue, contrast and brightness of the images.
Note that the implementation of these random pre-processing steps are inspired by the TensorFlow pre-processing pipeline of the Inception network. In short, the previous steps correspond to the following code:
```python
# Crop and distort image and bounding boxes.
dst_image, labels, bboxes, distort_bbox = \
    distorted_bounding_box_crop(image, labels, bboxes,
                                aspect_ratio_range=CROP_RATIO_RANGE)
# Resize image to output size.
dst_image = tf_image.resize_image(dst_image, out_shape,
                                  method=tf.image.ResizeMethod.BILINEAR)
# Randomly flip the image horizontally.
dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)
# Randomly distort the colors. There are 4 ways to do it.
dst_image = apply_with_random_selector(
        dst_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)
# Rescale to VGG input scale.
image = dst_image * 255.
image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
return image, labels, bboxes
```
The following image presents the result the pre-processing on an image.
![](pictures/ssd_preprocessing.png "SSD Pre-processing")

* ```nets```: the last important piece in this puzzle gathers the definition of the SSD network. For that purpose, we used the TF-slim library, which is a simpler interface provided in TensorFlow. It allows to define very simply and in a very lines a deep network. In our case, the SSD network described above is implemented the source file ```ssd_vgg_300.py``` and only consists of the following few lines:
```python
feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],

def ssd_multibox_layer(inputs, num_classes,
                       sizes, ratios=[1],
                       normalization=-1):
    """Construct a multibox layer, return class and localization predictions Tensors.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)
    # Localization.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], scope='conv_loc')
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], scope='conv_cls')
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred

def ssd_net(inputs, num_classes,
            feat_layers, anchor_sizes, anchor_ratios,
            normalizations,
            is_training=True,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
```
The ```nets``` directory contains a few more important methods necessary to the SSD network. The complex loss function, combining classification and localization losses is implemented in the file ```ssd_vgg_300.py``` as the method ```ssd_losses```. The source file ```ssd_common.py``` contains multiple functions related to bounding boxes computations (jaccard score, intersection, resizing, filtering, ...). More specific to the SSD network, it also contains the functions ``tf_ssd_bboxes_encode```` and ```tf_ssd_bboxes_decode``` responsible of encoding (and decoding) labels and bounding boxes into the output format of the SSD network, i.e. for each feature layer, two Tensors corresponding to classification and localisation. 

## SSD Training




Files in this directory describe the structure of the dataset, how to convert to TFRecords, and some data generation if necessary.
* preprocessing: Define the preprocessing of train and test datasets (color variations and so on)
* nets: Definition of the CNNs architecture, using TF-Slim easy interface.









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
