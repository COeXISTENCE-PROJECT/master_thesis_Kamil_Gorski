from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box
from gym import spaces
import numpy as np

ADDITIONAL_ENV_PARAMS = {
    'max_accel': 3,
    'max_decel': 3,
    'target_velocity': 10,
    'sort_vehicles': True
}


class RectangleEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        num_rl_vehicles = self.initial_vehicles.num_rl_vehicles

        accel= spaces.Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(num_rl_vehicles, ),
            dtype=np.float32)

        route = [spaces.Discrete(2) for _ in range(num_rl_vehicles)]

        return spaces.Tuple((accel, *route))

    @property
    def observation_space(self):
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=-200,
            high=10,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        # print(sorted_rl_ids)
        # print(rl_actions)
        accel, *rl_routes = rl_actions
        available_routes = {
            0 : ["upper-long", "cross"],
            1 : ["upper-long", "upper-short"],
        }
        choosed_routes = [available_routes[route] for route in rl_routes]
        for rl_id, route in zip(sorted_rl_ids, choosed_routes):
            if self.k.vehicle.get_edge(rl_id) == "upper-long":
                self.k.vehicle.choose_routes(rl_id, route)
        self.k.vehicle.apply_acceleration(sorted_rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]
        # stopped_vehicles = list(filter(lambda x: x < 0.01, speed))
        # print(stopped_vehicles)

        return np.array(speed + pos)

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
