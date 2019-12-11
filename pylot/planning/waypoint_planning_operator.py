import carla
import collections
import itertools
import numpy as np

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.map.hd_map import HDMap
from pylot.planning.messages import WaypointsMessage
from pylot.simulation.carla_utils import get_map
from pylot.simulation.utils import Transform, Rotation, Location
from pylot.planning.utils import get_waypoint_vector_and_angle

DEFAULT_NUM_WAYPOINTS = 5


class WaypointPlanningOperator(Op):
    """ Waypoint Planning operator for Carla 0.9.x.

    IMPORTANT: Do not use with older Carla versions.
    The operator either receives all the waypoints from the scenario runner
    agent (on the global trajectory stream), or computes waypoints using the
    HD Map.
    """
    def __init__(self,
                 name,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        super(WaypointPlanningOperator, self).__init__(name)
        self._log_file_name = log_file_name
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags

        self._wp_index = 4  # use 5th waypoint for steering and speed
        self._vehicle_transform = None
        self._waypoints = None
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._goal_location = carla.Location(*goal_location)

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            WaypointPlanningOperator.on_can_bus_update)
        return [pylot.utils.create_waypoints_stream()]

    def on_can_bus_update(self, msg):
        """
        Recompute path to goal location and send WaypointsMessage.

        :param msg: CanBus message
        :return: None
        """
        self._vehicle_transform = msg.data.transform
        self.__update_waypoints()
        self.__remove_completed_waypoints(self._vehicle_transform)
        next_waypoint = self._waypoints[self._wp_index]
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)

        waypoints = collections.deque(
            itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS)
        )  # only take 50 meters

        hack = collections.defaultdict(float)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for i, x in enumerate(range(-5, 6)):
            hack[i + 280] = sigmoid(-x)
        for i, x in enumerate(range(-5, 5)):
            hack[i + 270] += sigmoid(x)

        new_wps = []
        for wp in waypoints:
            x = wp.location.x
            y = wp.location.y
            new_wp = Transform(
                location=Location(
                    x=x,
                    y=y + 4 * hack[int(x)],
                    z=0
                ),
                rotation=Rotation()
            )
            new_wps.append(new_wp)
        waypoints = new_wps

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=waypoints,
            wp_angle=wp_steer_angle,
            wp_vector=wp_steer_vector,
            wp_angle_speed=wp_speed_angle
        )
        self.get_output_stream('waypoints').send(output_msg)

    def __update_waypoints(self):
        ego_location = self._vehicle_transform.location.as_carla_location()
        if not self._waypoints:
            self._waypoints = self._map.compute_waypoints(ego_location, self._goal_location)

    def __remove_completed_waypoints(self, vehicle_transform):
        """ Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint, removes all waypoints
        that are before the closest waypoint, and finally removes the
        closest waypoint if the ego vehicle is very close to it
        (i.e., close to completion)."""
        min_dist = 10000000
        min_index = 0
        index = 0
        for waypoint in self._waypoints:
            # XXX(ionel): We only check the first 10 waypoints.
            if index > 10:
                break
            dist = pylot.planning.utils.get_distance(waypoint.location,
                                                     vehicle_transform.location)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self._waypoints.popleft()
            min_index -= 1

        # The closest waypoint is almost complete, remove it.
        if min_dist < 5:
            self._waypoints.popleft()


