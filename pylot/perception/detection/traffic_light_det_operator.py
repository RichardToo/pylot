from absl import flags
import erdos
import logging
import numpy as np
import tensorflow as tf
import time

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle
from pylot.perception.detection.traffic_light import TrafficLightColor
from pylot.perception.messages import ObstaclesMessage
from pylot.utils import set_tf_loglevel, time_epoch_ms

flags.DEFINE_string(
    'traffic_light_det_model_path',
    'dependencies/models/traffic_light_detection/faster-rcnn/frozen_inference_graph.pb',
    'Path to the traffic light model protobuf')
flags.DEFINE_float('traffic_light_det_min_score_threshold', 0.3,
                   'Min score threshold for bounding box')
flags.DEFINE_float(
    'traffic_light_det_gpu_memory_fraction', 0.3,
    'GPU memory fraction allocated to each traffic light detector')


class TrafficLightDetOperator(erdos.Operator):
    """ Subscribes to a camera stream, and runs a model for each frame."""
    def __init__(self,
                 camera_stream,
                 traffic_lights_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        # Register a callback on the camera input stream.
        camera_stream.add_callback(self.on_frame, [traffic_lights_stream])
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        set_tf_loglevel(logging.ERROR)
        with self._detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self._flags.traffic_light_det_model_path,
                                   'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=flags.
            traffic_light_det_gpu_memory_fraction)
        # Create a TensorFlow session.
        self._tf_session = tf.compat.v1.Session(
            graph=self._detection_graph,
            config=tf.compat.v1.ConfigProto(gpu_options=self._gpu_options))
        # Get the tensors we're interested in.
        self._image_tensor = self._detection_graph.get_tensor_by_name(
            'image_tensor:0')
        self._detection_boxes = self._detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        self._detection_scores = self._detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self._detection_classes = self._detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name(
            'num_detections:0')
        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF
        }
        # The bounding box colors to use in the visualizer.
        self._bbox_colors = {
            TrafficLightColor.GREEN: [0, 128, 0],
            TrafficLightColor.RED: [255, 0, 0],
            TrafficLightColor.YELLOW: [255, 255, 0],
            TrafficLightColor.OFF: [0, 0, 0]
        }

    @staticmethod
    def connect(camera_stream):
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]

    def on_frame(self, msg, traffic_lights_stream):
        """ Invoked when the operator receives a message on the data stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        start_time = time.time()
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(msg.frame.as_rgb_numpy_array(),
                                           axis=0)
        (boxes, scores, classes, num) = self._tf_session.run(
            [
                self._detection_boxes, self._detection_scores,
                self._detection_classes, self._num_detections
            ],
            feed_dict={self._image_tensor: image_np_expanded})

        num_detections = int(num[0])
        labels = [self._labels[label] for label in classes[0][:num_detections]]
        boxes = boxes[0][:num_detections]
        scores = scores[0][:num_detections]

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.frame.camera_setup.height,
            msg.frame.camera_setup.width)

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            msg.timestamp, self._name, traffic_lights))

        if (self._flags.visualize_detected_traffic_lights
                or self._flags.log_traffic_light_detector_output):
            msg.frame.annotate_with_bounding_boxes(msg.timestamp,
                                                   traffic_lights,
                                                   self._bbox_colors)
            if self._flags.visualize_detected_traffic_lights:
                msg.frame.visualize(self._name)
            if self._flags.log_traffic_light_detector_output:
                msg.frame.save(msg.timestamp.coordinates[0],
                               self._flags.data_path,
                               'tl-detector-{}'.format(self._name))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, msg.timestamp,
                                                     runtime))

        traffic_lights_stream.send(
            ObstaclesMessage(traffic_lights, msg.timestamp, runtime))

    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        for index in range(len(scores)):
            if scores[
                    index] > self._flags.traffic_light_det_min_score_threshold:
                bbox = BoundingBox2D(
                    int(boxes[index][1] * width),  # x_min
                    int(boxes[index][3] * width),  # x_max
                    int(boxes[index][0] * height),  # y_min
                    int(boxes[index][2] * height)  # y_max
                )
                traffic_lights.append(
                    DetectedObstacle(bbox, scores[index], labels[index]))
        return traffic_lights
