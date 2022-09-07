from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.networks import FigureEightNetwork
from flow.core.experiment import Experiment

HORIZON = 3000

sim_params = SumoParams(sim_step=0.1, render=True)

vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=13)

env_params = EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'sort_vehicles': False
        },
)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(bunching=20)

flow_params = dict(
    exp_tag='ring',
    env_name=AccelEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params['env'].horizon = 1500
exp = Experiment(flow_params)

_ = exp.run(1)