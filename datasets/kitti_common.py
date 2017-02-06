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
import numpy as np
import tensorflow as tf

KITTI_LABELS = {
    'none': (0, 'Background'),
    'Car': (1, 'Vehicle'),
    'Van': (2, 'Vehicle'),
    'Truck': (3, 'Vehicle'),
    'Cyclist': (4, 'Vehicle'),
    'Pedestrian': (5, 'Person'),
    'Person_sitting': (6, 'Person'),
    'Tram': (7, 'Vehicle'),
    'Misc': (8, 'Misc'),
    'DontCare': (9, 'DontCare'),
}
KITTI_DONTCARE = 9
NUM_CLASSES = 8

