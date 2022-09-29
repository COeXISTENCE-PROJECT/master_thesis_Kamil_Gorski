import numpy as np
from numpy import pi, sin, cos, linspace

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40
}


class CandyNetwork(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a figure 8 network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        ring_radius = net_params.additional_params["radius_ring"]
        self.ring_edgelen = ring_radius * np.pi / 2.
        self.intersection_len = 2 * ring_radius
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # # instantiate "length" in net params
        # net_params.additional_params["length"] = \
        #     6 * self.ring_edgelen + 2 * self.intersection_len + \
        #     2 * self.junction_len + 10 * self.inner_space_len

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]

        nodes = [{
            "id": "center_top",
            "x": 0,
            "y": 0,
            # "radius": (2.9 + 3.3 * net_params.additional_params["lanes"])/2,
            "type": "priority"
        }, {
            "id": "right",
            "x": r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "top",
            "x": 0,
            "y": r,
            "type": "priority"
        }, {
            "id": "square_left",
            "x": -2*r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "square_right",
            "x": 0,
            "y": -2*r,
            "type": "priority"
        }, {
            "id": "left",
            "x": -3*r,
            "y": -2*r,
            "type": "priority"
        }, {
            "id": "bottom",
            "x": -2*r,
            "y": -3*r,
            "type": "priority"
        }, {
            "id": "center_bottom",
            "x": -2*r,
            "y": -2*r,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_edgelen = 3 * r * pi / 2.
        intersection_edgelen = 2 * r

        # intersection edges
        edges = [{
            "id": "bottom",
            "type": "edgeType",
            "priority": "78",
            "from": "center_bottom",
            "to": "bottom",
            "length": intersection_edgelen / 2
        }, {
            "id": "top",
            "type": "edgeType",
            "priority": 78,
            "from": "center_top",
            "to": "top",
            "length": intersection_edgelen / 2
        }, {
            "id": "right",
            "type": "edgeType",
            "priority": 46,
            "from": "right",
            "to": "center_top",
            "length": intersection_edgelen / 2
        }, {
            "id": "left",
            "type": "edgeType",
            "priority": 46,
            "from": "left",
            "to": "center_bottom",
            "length": intersection_edgelen / 2
        }]

        # ring edges
        edges += [{
            "id": "upper_ring",
            "type": "edgeType",
            "from": "top",
            "to": "right",
            "length": ring_edgelen,
            "shape": [(r * (1 - cos(t)), r * (1 + sin(t)))
                      for t in linspace(0, 3 * pi / 2, resolution)]
        }, {
            "id": "lower_ring",
            "type": "edgeType",
            "from": "bottom",
            "to": "left",
            "length": ring_edgelen,
            "shape": [(-3*r + r * cos(t), -3*r + r * sin(t))
                      for t in linspace(2*pi, pi/2, resolution)]
        }]

        edges += [{
            "id": "square_left",
            "type": "edgeType",
            "priority": 46,
            "from": "square_left",
            "to": "center_bottom",
            "length": intersection_edgelen
        }, {
            "id": "square_top",
            "type": "edgeType",
            "priority": 46,
            "from": "center_top",
            "to": "square_left",
            "length": intersection_edgelen
        } , {
            "id": "square_right",
            "type": "edgeType",
            "priority": 46,
            "from": "square_right",
            "to": "center_top",
            "length": intersection_edgelen
        } , {
            "id": "square_bottom",
            "type": "edgeType",
            "priority": 46,
            "from": "center_bottom",
            "to": "square_right",
            "length": intersection_edgelen
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "square_right": (
                (["square_right", "top"], 0.5),
                (["square_right", "square_top"], 0.5)
            ),
            "square_left": (
                (["square_left", "bottom"], 0.5),
                (["square_left", "square_bottom"], 0.5)
            ),
            "top": ["top", "upper_ring"],
            "upper_ring": ["upper_ring", "right"],
            "right": ["right", "square_top"],
            "square_top": ["square_top", "square_left"],
            "bottom": ["bottom", "lower_ring"],
            "lower_ring": ["lower_ring", "left"],
            "left": ["left", "square_bottom"],
            "square_bottom": ["square_bottom", "square_right"],
        }

        return rts

    # def specify_connections(self, net_params):
    #     """See parent class."""
    #     lanes = net_params.additional_params["lanes"]
    #     conn_dict = {}
    #     conn = []
    #     for i in range(lanes):
    #         conn += [{"from": "bottom",
    #                   "to": "top",
    #                   "fromLane": str(i),
    #                   "toLane": str(i)}]
    #         conn += [{"from": "right",
    #                   "to": "left",
    #                   "fromLane": str(i),
    #                   "toLane": str(i)}]
    #     conn_dict["center"] = conn
    #     return conn_dict

    # def specify_edge_starts(self):
    #     edgestarts = [
    #         # ("bottom", 0),
    #         # ("lower_ring", 1),
    #         # ("left", 2),
    #         # ("square_bottom", 3),
    #         # ("square_right", 4),
    #         # ("top", 5),
    #         # ("upper_ring", 6),
    #         # ("right", 7),
    #         # ("square_top", 8),
    #         # ("square_left", 9),

    #         ("square_right", self.inner_space_len),
    #         ("top", self.intersection_len / 2 + self.junction_len +
    #          self.inner_space_len),
    #         ("upper_ring", self.intersection_len + self.junction_len +
    #          2 * self.inner_space_len),
    #         ("right", self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 3 * self.inner_space_len),

    #         ("square_top", 2*self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 3 * self.inner_space_len),
    #         ("square_left", 3*self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 3 * self.inner_space_len),
    #         ("bottom", 3*self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 3 * self.inner_space_len + self.junction_len),

    #         ("lower_ring", 4 * self.intersection_len + 3 * self.ring_edgelen
    #          + 3 * self.junction_len + 4 * self.inner_space_len),
    #         ("left", 4 * self.intersection_len + 3 * self.ring_edgelen
    #          + 4 * self.junction_len + 3 * self.inner_space_len),

    #         ("square_bottom", 5 * self.intersection_len + 3 * self.ring_edgelen
    #          + 4 * self.junction_len + 3 * self.inner_space_len),
            
    #     ]
        # return None
        # return edgestarts

    # def specify_internal_edge_starts(self):
    #     """See base class."""
    #     internal_edgestarts = [
    #         (":bottom", 0),
    #         (":center_{}".format(self.net_params.additional_params['lanes']),
    #          self.intersection_len / 2 + self.inner_space_len),
    #         (":top", self.intersection_len + self.junction_len +
    #          self.inner_space_len),
    #         (":right", self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 2 * self.inner_space_len),
    #         (":center_0", 3 / 2 * self.intersection_len + 3 * self.ring_edgelen
    #          + self.junction_len + 3 * self.inner_space_len),
    #         (":left", 2 * self.intersection_len + 3 * self.ring_edgelen
    #          + 2 * self.junction_len + 3 * self.inner_space_len),
    #         # for aimsun
    #         ('bottom_to_top',
    #          self.intersection_len / 2 + self.inner_space_len),
    #         ('right_to_left',
    #          + self.junction_len + 3 * self.inner_space_len),
    #     ]

    #     return internal_edgestarts
