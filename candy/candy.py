from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter

from flow.envs import CandyEnv
from flow.networks import CandyNetwork
from flow.core.experiment import Experiment
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS

HORIZON = 1500


sim_params = SumoParams(sim_step=0.1, render=True, emission_path=None)

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=18,
)

env_params = EnvParams(
    horizon=HORIZON,
    additional_params={
        "target_velocity": 20,
        "max_accel": 3,
        "max_decel": 3,
        "sort_vehicles": False,
    },
)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig()

flow_params = dict(
    exp_tag="candy",
    env_name=CandyEnv,
    network=CandyNetwork,
    simulator="traci",
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params["env"].horizon = HORIZON
exp = Experiment(flow_params)

_ = exp.run(1)
