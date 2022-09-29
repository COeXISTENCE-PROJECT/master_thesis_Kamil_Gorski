"""Contains a list of custom routing controllers."""
from __future__ import print_function
import random
import numpy as np

from flow.controllers.base_routing_controller import BaseRouter






class BraessRouter(BaseRouter):
    max_se = 0
    max_ne = 0
    arrived_vehicles = 0
    last_tc = 0

    def choose_route(self, env):
        ne_size = len(env.k.vehicle.get_ids_by_edge("ne"))
        se_size = len(env.k.vehicle.get_ids_by_edge("se"))
        # print(env.k.v)
        if ne_size > BraessRouter.max_ne: BraessRouter.max_ne = ne_size 
        if se_size > BraessRouter.max_se: BraessRouter.max_se = se_size 
        # print(BraessRouter.max_se)
        if BraessRouter.last_tc != env.k.vehicle.time_counter:
            BraessRouter.last_tc = env.k.vehicle.time_counter
            num_arrived = env.k.vehicle.get_num_arrived()
            BraessRouter.arrived_vehicles += num_arrived
            # print(BraessRouter.arrived_vehicles)

        edge = env.k.vehicle.get_edge(self.veh_id)
        # if edge == 'sw':
            # return env.available_routes[edge][0][0]
        # elif edge == "inflow_highway":
            # return env.available_routes[edge][1][0]
        lane = np.random.randint(0, 2)
        return env.available_routes[edge][lane][0] if edge == "inflow_highway" else None
        # return None



class CircleRLRouter(BaseRouter):
    def choose_route(self, env):
        edge = env.k.vehicle.get_edge(self.veh_id) 
        if edge in env.available_routes:
            if edge == "first":
                return env.available_routes[edge][1][0]
            else:
                return env.available_routes[edge][0][0]
        else:
            return None

class CircleHumanRouter(BaseRouter):
    def choose_route(self, env):
        edge = env.k.vehicle.get_edge(self.veh_id) 
        current_route = env.k.vehicle.get_route(self.veh_id)
        if "human" in self.veh_id and edge in env.available_routes:
            route = env.available_routes["human"][0][0]
            return env.available_routes[edge][0][0]
        elif "robot" in self.veh_id and edge in env.available_routes:
            if edge == "first":
                return env.available_routes[edge][1][0]
            else:
                return env.available_routes[edge][0][0]
        return None 
        # if len(current_route) == 0:
        #     # this occurs to inflowing vehicles, whose information is not added
        #     # to the subscriptions in the first step that they departed
        #     return None
        # elif edge == current_route[-1]:
        #     # choose one of the available routes based on the fraction of times
        #     # the given route can be chosen
        #     num_routes = len(env.available_routes[edge])
        #     frac = [val[1] for val in env.available_routes[edge]]
        #     route_id = np.random.choice(
        #         [i for i in range(num_routes)], size=1, p=frac)[0]

        #     # pass the chosen route
        #     return env.available_routes[edge][route_id][0]
        # else:
        #     return None


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)
        
        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_routes[edge])
            frac = [val[1] for val in env.available_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return env.available_routes[edge][route_id][0]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity network.

    This class allows the vehicle to pick a random route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.k.network.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.k.vehicle.get_edge(self.veh_id) == \
                env.k.vehicle.get_route(self.veh_id)[-1]:
            return [env.k.vehicle.get_edge(self.veh_id)]
        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class I210Router(ContinuousRouter):
    """Assists in choosing routes in select cases for the I-210 sub-network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        # vehicles on these edges in lanes 4 and 5 are not going to be able to
        # make it out in time
        if edge == "119257908#1-AddedOffRampEdge" and lane in [5, 4, 3]:
            new_route = env.available_routes[
                "119257908#1-AddedOffRampEdge"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class RectangleRLRouter(ContinuousRouter):
    def choose_route(self, env):
        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "upper-long":
            return None
        else:
            return super().choose_route(env)

class CandyRLRouter(ContinuousRouter):
    def choose_route(self, env):
        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge in ["square_left", "square_right"]:
            return None
        else:
            return super().choose_route(env)