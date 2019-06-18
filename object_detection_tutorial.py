#####################################################################################################
# This is adapted from the object detection ipython notebook
#  that comes in tensorflow/models/research/object_detection
# To use.
# 1 clone tensorflow/models into your home dir
# 2 install jupyter notebook and load the object detection tut in models/research/object_detection
# 3 follow the install instructions in the tut
# 3a check that the tutorial runs
# 4 pull down the ssd_mobilenet_v1_coco model from the model zoo and extract into your home dir
# 5 use the model exporter to re-export the model for tf v1.13.1 appending _tf1.13.1 to the output dir name
# 6 run from models/research/object_detection/
#####################################################################################################

import numpy as np
import os
import sys
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.contrib.tensorrt as trt

from distutils.version import StrictVersion
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/nigel/models/research")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

sys.path.append("/home/nigel/models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = '/home/nigel/ssd_mobilenet_v1_coco_2018_01_28_tf1.13.1'
PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,
                                  'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = 0.5))
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_graph_outputs(graph):
  tensor_dict = {}
  with graph.as_default():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    for key in ['raw_detection_boxes', 'raw_detection_scores']: #, 'num_detections', 'detection_classes', 'detection_masks']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
  return tensor_dict

def load_detection_graph(path):
  graph     = tf.Graph()
  graph_def = tf.GraphDef()

  with graph.as_default():
    with tf.gfile.GFile(path, 'rb') as fid:
      serialized_graph = fid.read()
      graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(graph_def, name='')

  for op in graph.get_operations():
    if op.name.find('raw_') > -1:
      print('xprev:{}'.format(op.name))

  return graph

def trt_optimize_graph(graph, max_batch = 1):
  graph_def = graph.as_graph_def()

  tensor_dict  = get_graph_outputs(graph)
  output_nodes = tensor_dict.keys()

  new_graph = tf.Graph()

  with new_graph.as_default():
    print('pre:outputs:{}. nodes:{}'.format(output_nodes, len(graph_def.node)))
    graph_def = trt.create_inference_graph(graph_def,
                                           output_nodes,
                                           max_workspace_size_bytes = 2*1024*1024,
                                           max_batch_size = max_batch,
                                           precision_mode = 'FP16',
                                           is_dynamic_op = True,
                                           minimum_segment_size = 15,
                                           maximum_cached_engines = 2,
                                           session_config = config)

    print('post:outputs:{}. nodes:{}'.format(output_nodes, len(graph_def.node)))
    tf.import_graph_def(graph_def, name='')

  for op in new_graph.get_operations():
    if op.name.find('Conv2d_0') > 0:
      print('trt:{}'.format(op.name))

  return new_graph

# # List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


def get_image_batches(tile=2, batch=1, size=None):
  images = []
  for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image = image.resize(size) if size is not None else image
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    print('test image:{}, shape:{}'.format(image_path, image_np_expanded.shape))
    images.append(np.tile(image_np_expanded, (batch, 1, 1, 1)))

  images = images * tile

  print('len(images):{}'.format(len(images)))

  return images

def main(n_iters=1000, batch=8, tile=2, trt=True):
  graph = load_detection_graph(PATH_TO_FROZEN_GRAPH)

  if trt: graph = trt_optimize_graph(graph)

  tensor_dict = get_graph_outputs(graph)

  images = get_image_batches(size=(300,300), batch=batch, tile=tile)

  with graph.as_default():
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    #try to bypass image pre-processing
    #image_tensor = tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D:0')
    with tf.Session(config=config) as sess:
      for image in images:
        start_time = time.time()
        for i in range(n_iters):
          output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
        elapsed = time.time()-start_time
        print('batch[{}], [{:.1f}s] i.e. {:.3f} images per sec:got output_dict:{}, person/dog/kite > 0.5 {}/{}/{}, shapes b/s {}/{}'.format(image.shape[0],
                                                                                          elapsed,
                                                                                          n_iters*image.shape[0]/elapsed,
                                                                                          output_dict.keys(),
                                                                                          np.sum(output_dict['raw_detection_scores'][:,:, 1] > 0.5, axis=1),
                                                                                          np.sum(output_dict['raw_detection_scores'][:,:,18] > 0.5, axis=1),
                                                                                          np.sum(output_dict['raw_detection_scores'][:,:,38] > 0.5, axis=1),
                                                                                          output_dict['raw_detection_boxes' ].shape,
                                                                                          output_dict['raw_detection_scores'].shape,
        ))

        print(output_dict['raw_detection_boxes'][0])
        print(output_dict['raw_detection_scores'][0,:,(1,18,38)])

        ## all outputs are float32 numpy arrays, so convert types as appropriate
        #output_dict[   'num_detections'] = int(output_dict['num_detections'][0])
        #output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
        #output_dict[  'detection_boxes'] = output_dict['detection_boxes'][0]
        #output_dict[ 'detection_scores'] = output_dict['detection_scores'][0]
        #if 'detection_masks' in output_dict:
        #  output_dict['detection_masks'] = output_dict['detection_masks'][0]

        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   image_np,
        #   output_dict['detection_boxes'],
        #   output_dict['detection_classes'],
        #   output_dict['detection_scores'],
        #   category_index,
        #   instance_masks=output_dict.get('detection_masks'),
        #   use_normalized_coordinates=True,
        #   line_thickness=8)
        # matplotlib.use('GTK3Agg')
        # fig = plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        # plt.show(fig)

    print(output_dict['raw_detection_boxes'][0])
    print(output_dict['raw_detection_scores'][0])

main(n_iters=1000, batch=16, tile=2, trt=True)
