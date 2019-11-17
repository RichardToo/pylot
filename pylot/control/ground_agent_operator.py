from collections import deque
import math
import threading
from pid_controller.pid import PID

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.map.hd_map import HDMap
from pylot.simulation.carla_utils import get_map
import pylot.utils


class GroundAgentOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(GroundAgentOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._can_bus_msgs = deque()
        self._pedestrian_msgs = deque()
        self._vehicle_msgs = deque()
        self._traffic_light_msgs = deque()
        self._speed_limit_sign_msgs = deque()
        self._waypoint_msgs = deque()
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            GroundAgentOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                GroundAgentOperator.on_pedestrians_update)
        input_streams.filter(
            pylot.utils.is_ground_vehicles_stream).add_callback(
                GroundAgentOperator.on_vehicles_update)
        input_streams.filter(
            pylot.utils.is_ground_traffic_lights_stream).add_callback(
                GroundAgentOperator.on_traffic_lights_update)
        input_streams.filter(
            pylot.utils.is_ground_speed_limit_signs_stream).add_callback(
                GroundAgentOperator.on_speed_limit_signs_update)
        input_streams.filter(pylot.utils.is_waypoints_stream).add_callback(
            GroundAgentOperator.on_waypoints_update)
        input_streams.add_completion_callback(
            GroundAgentOperator.on_notification)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_notification(self, msg):
        # Get hero vehicle info.
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        # Get waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        # Get ground pedestrian info.
        pedestrians_msg = self._pedestrian_msgs.popleft()
        pedestrians = pedestrians_msg.pedestrians
        # Get ground vehicle info.
        vehicles_msg = self._vehicle_msgs.popleft()
        vehicles = vehicles_msg.vehicles
        # Get ground traffic lights info.
        traffic_lights_msg = self._traffic_light_msgs.popleft()
        traffic_lights = traffic_lights_msg.traffic_lights
        # Get ground traffic signs info.
        speed_limit_signs_msg = self._speed_limit_sign_msgs.popleft()
        speed_limit_signs = speed_limit_signs_msg.speed_signs
        # TODO(ionel): Use traffic signs info as well.

        speed_factor, state = self.stop_for_agents(vehicle_transform.location,
                                                   wp_angle,
                                                   wp_vector,
                                                   vehicles,
                                                   pedestrians,
                                                   traffic_lights)
        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)

    def on_waypoints_update(self, msg):
        with self._lock:
            self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        with self._lock:
            self._pedestrian_msgs.append(msg)

    def on_vehicles_update(self, msg):
        with self._lock:
            self._vehicle_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        with self._lock:
            self._traffic_light_msgs.append(msg)

    def on_speed_limit_signs_update(self, msg):
        with self._lock:
            self._speed_limit_sign_msgs.append(msg)

    def stop_for_agents(self,
                        ego_vehicle_location,
                        wp_angle,
                        wp_vector,
                        vehicles,
                        pedestrians,
                        traffic_lights):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        if self._flags.stop_for_vehicles:
            for obs_vehicle in vehicles:
                # Only brake for vehicles that are in ego vehicle's lane.
                if self._map.are_on_same_lane(
                        ego_vehicle_location,
                        obs_vehicle.transform.location):
                    new_speed_factor_v = pylot.control.utils.stop_vehicle(
                        ego_vehicle_location,
                        obs_vehicle.transform.location,
                        wp_vector,
                        speed_factor_v,
                        self._flags)
                    speed_factor_v = min(speed_factor_v, new_speed_factor_v)

        if self._flags.stop_for_pedestrians:
            for pedestrian in pedestrians:
                # Only brake for pedestrians that are on the road.
                if self._map.is_on_lane(pedestrian.transform.location):
                    new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                        ego_vehicle_location,
                        pedestrian.transform.location,
                        wp_vector,
                        speed_factor_p,
                        self._flags)
                    speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        if self._flags.stop_for_traffic_lights:
            for tl in traffic_lights:
                if (self._map.must_obbey_traffic_light(
                        ego_vehicle_location, tl.transform.location) and
                    self._is_traffic_light_visible(
                        ego_vehicle_location, tl.transform.location)):
                    new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                        ego_vehicle_location,
                        tl.transform.location,
                        tl.state,
                        wp_vector,
                        wp_angle,
                        speed_factor_tl,
                        self._flags)
                    speed_factor_tl = min(speed_factor_tl, new_speed_factor_tl)

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }

        return speed_factor, state

    def execute(self):
        self.spin()

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, timestamp):
        current_speed = max(current_speed, 0)
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = self._flags.target_speed * speed_factor / 2
        else:
            target_speed_adjusted = self._flags.target_speed * speed_factor

        self._pid.target = target_speed_adjusted
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(
            max(self._flags.default_throttle - 1.3 * pid_gain, 0),
            self._flags.throttle_max)

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0

        return ControlMessage(steer, throttle, brake, False, False, timestamp)

    def _is_traffic_light_visible(self, ego_vehicle_location, tl_location):
        _, tl_dist = pylot.control.utils.get_world_vec_dist(
            ego_vehicle_location.x,
            ego_vehicle_location.y,
            tl_location.x,
            tl_location.y)
        return tl_dist > self._flags.traffic_light_min_dist_thres
