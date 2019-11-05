import threading
import collections

import carla
import numpy as np
from pid_controller.pid import PID

from erdos.op import Op
from erdos.utils import setup_logging, setup_csv_logging

import pylot.utils
from pylot.control.messages import ControlMessage
from pylot.simulation.utils import to_pylot_transform
from pylot.simulation.carla_utils import get_world, to_carla_location,\
        to_carla_transform
from pylot.planning.utils import get_waypoint_vector_and_angle


class PerfectPlanningOperator(Op):
    def __init__(self,
                 name,
                 goal,
                 behavior,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """ Initializes the operator with the given information.

        Args:
            name: The name to be used for the operator in the dataflow graph.
            goal: The final location used to plan until.
            behavior: The behavior to show in case of emergencies.
            flags: The command line flags passed to the driver.
            log_file_name: The file name to log the intermediate messages to.
            csv_file_name: The file name to log the experimental results to.
        """
        super(PerfectPlanningOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._goal = goal

        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._world = world
        self._map = world.get_map()

        # Input retrieved from the various input streams.
        self._lock = threading.Lock()
        self._can_bus_msgs = collections.deque()
        self._pedestrians = collections.deque()
        self._obstacle_msgs = collections.deque()

        # PID Controller
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)

        # Planning constants.
        self.PLANNING_BEHAVIOR = behavior
        self.SPEED = self._flags.target_speed
        self.DETECTION_DISTANCE = 12
        self.GOAL_DISTANCE = self.SPEED
        self.SAMPLING_DISTANCE = self.SPEED / 3
        self._goal_reached = False

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PerfectPlanningOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                PerfectPlanningOperator.on_pedestrians_update)
        input_streams.filter(pylot.utils.is_obstacles_stream).add_callback(
            PerfectPlanningOperator.on_obstacle_update)
        input_streams.add_completion_callback(
            PerfectPlanningOperator.on_notification)
        return [pylot.utils.create_control_stream()]

    def on_obstacle_update(self, msg):
        """ Receives the message from the detector and adds it to the queue.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a detector update for the timestamp: {}".format(
                msg.timestamp))
        with self._lock:
            self._obstacle_msgs.append(msg)

    def on_can_bus_update(self, msg):
        """ Receives the CAN Bus update and adds it to the queue of messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a CAN Bus update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        """ Receives the pedestrian update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a pedestrian update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._pedestrians.append(msg)

    def synchronize_msg_buffers(self, timestamp, buffers):
        """ Synchronizes the given buffers for the given timestamp.

       Args:
           timestamp: The timestamp to push all the top of the buffers to.
           buffers: The buffers to synchronize.

       Returns:
           True, if the buffers were successfully synchronized. False,
           otherwise.
       """
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def __get_steer(self, wp_angle):
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)
        return steer

    def __get_throttle_brake_without_factor(self, current_speed, target_speed):
        self._pid.target = target_speed
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(max(self._flags.default_throttle - 1.3 * pid_gain, 0),
                       self._flags.throttle_max)
        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0
        return throttle, brake

    def __get_forward_vector(self, waypoint):
        fwd_vector = waypoint.get_forward_vector()
        return [fwd_vector.x, fwd_vector.y, fwd_vector.z]

    def on_notification(self, msg):
        """ The callback function invoked upon receipt of a WatermarkMessage.

        The function uses the latest location of the vehicle and drives to the
        next waypoint, while doing either a stop or a swerve upon the
        detection of a pedestrian.

        Args:
            msg: The timestamp for which the WatermarkMessage is retrieved.
        """
        self._logger.info("Received a watermark for the timestamp {}".format(
            msg.timestamp))
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                [self._can_bus_msgs, self._pedestrians, self._obstacle_msgs]):
                self._logger.info("Could not synchronize the message buffers "
                                  "for the timestamp {}".format(msg.timestamp))

            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrian_msg = self._pedestrians.popleft()
            obstacle_msg = self._obstacle_msgs.popleft()

        # Assert that the timestamp of all the messages are the same.
        assert (can_bus_msg.timestamp == pedestrian_msg.timestamp ==
                obstacle_msg.timestamp)

        self._logger.info(
            "Can Bus Message: {}, Pedestrian Message: {}, Obstacle Message: {}"
            .format(can_bus_msg.timestamp, pedestrian_msg.timestamp,
                    obstacle_msg.timestamp))
        self._logger.info(
            "The vehicle is travelling at a speed of {} m/s.".format(
                can_bus_msg.data.forward_speed))

        # Heuristic to tell us how far away do we detect the pedestrian.
        ego_transform = to_carla_transform(can_bus_msg.data.transform)
        ego_location = ego_transform.location
        ego_wp = self._map.get_waypoint(ego_location)
        for pedestrian in pedestrian_msg.pedestrians:
            pedestrian_loc = to_carla_location(pedestrian.transform.location)
            pedestrian_wp = self._map.get_waypoint(pedestrian_loc,
                                                   project_to_road=False)
            if pedestrian_wp and pedestrian_wp.road_id == ego_wp.road_id:
                for obj in obstacle_msg.detected_objects:
                    if obj.label == 'person':
                        self._csv_logger.info(
                            "Detected a person {}m away".format(
                                pedestrian_loc.distance(ego_location)))
                        self._csv_logger.info(
                            "The vehicle is travelling at a speed of {} m/s.".
                            format(can_bus_msg.data.forward_speed))

        # Figure out the location of the ego vehicle and compute the next
        # waypoint.
        if self._goal_reached or ego_location.distance(
                self._goal) <= self.GOAL_DISTANCE:
            print("The distance was {} and we reached the goal.".format(ego_location.distance(self._goal)))
            self.get_output_stream('control_stream').send(
                ControlMessage(0.0, 0.0, 1.0, True, False, msg.timestamp))
            self._goal_reached = True
        else:
            pedestrian_detected = False
            for pedestrian in pedestrian_msg.pedestrians:
                pedestrian_loc = to_carla_location(
                    pedestrian.transform.location)
                pedestrian_wp = self._map.get_waypoint(pedestrian_loc,
                                                       project_to_road=False)
                if pedestrian_wp and ego_location.distance(
                        pedestrian_loc) <= self.DETECTION_DISTANCE:
                    pedestrian_detected = True
                    break

            if pedestrian_detected and self.PLANNING_BEHAVIOR == 'stop':
                self.get_output_stream('control_stream').send(
                    ControlMessage(0.0, 0.0, 1.0, True, False, msg.timestamp))
                return

            # Get the waypoint that is SAMPLING_DISTANCE away.
            sample_distance = self.SAMPLING_DISTANCE if\
                    ego_transform.get_forward_vector().x > 0 else \
                    -1 * self.SAMPLING_DISTANCE
            wp_steer = self._map.get_waypoint(ego_location + carla.Location(
                x=sample_distance))

            in_swerve = False
            if pedestrian_detected:
                # If a pedestrian was detected, make sure we're driving on the
                # wrong direction.
                ego_vehicle_fwd = self.__get_forward_vector(
                    to_carla_transform(can_bus_msg.data.transform))
                waypoint_fwd = self.__get_forward_vector(wp_steer.transform)

                if np.dot(ego_vehicle_fwd, waypoint_fwd) > 0:
                    # We're not driving in the wrong direction, get left
                    # lane waypoint.
                    if wp_steer.get_left_lane():
                        wp_steer = wp_steer.get_left_lane()
                        in_swerve = True
                else:
                    # We're driving in the right direction, continue driving.
                    pass
            else:
                # The pedestrian was not detected, come back from the swerve.
                ego_vehicle_fwd = self.__get_forward_vector(
                    to_carla_transform(can_bus_msg.data.transform))
                waypoint_fwd = self.__get_forward_vector(wp_steer.transform)

                if np.dot(ego_vehicle_fwd, waypoint_fwd) < 0:
                    # We're driving in the wrong direction, get the left lane
                    # waypoint.
                    if wp_steer.get_left_lane():
                        wp_steer = wp_steer.get_left_lane()
                        in_swerve = True
                else:
                    # We're driving in the right direction, continue driving.
                    pass

            self._world.debug.draw_point(wp_steer.transform.location,
                                         size=0.2,
                                         life_time=30000.0)
            wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
                to_pylot_transform(wp_steer.transform),
                can_bus_msg.data.transform)

            current_speed = max(0, can_bus_msg.data.forward_speed)
            steer = self.__get_steer(wp_steer_angle)
            # target_speed = self.SPEED if not in_swerve else self.SPEED / 5.0
            target_speed = self.SPEED
            throttle, brake = self.__get_throttle_brake_without_factor(
                current_speed, target_speed)

            self.get_output_stream('control_stream').send(
                ControlMessage(steer, throttle, brake, False, False,
                               msg.timestamp))
