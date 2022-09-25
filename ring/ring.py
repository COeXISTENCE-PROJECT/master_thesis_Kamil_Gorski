from flow.core.experiment import Experiment
from flow.core.params import (
    NetParams,
    EnvParams,
    InitialConfig,
    VehicleParams,
    SumoParams,
    SumoCarFollowingParams,
)
from flow.controllers import IDMController
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.routing_controllers import ContinuousRouter

vehicles = VehicleParams()
vehicles.add(
    "human",
    acceleration_controller=(IDMController, {"noise": 0.2}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.0,
    ),
    num_vehicles=22,
)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()

net_params = NetParams(additional_params=additional_net_params)

sim_params = SumoParams(render=True, sim_step=0.1, emission_path=None)
initial_config = InitialConfig(spacing="uniform", bunching=5)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params = dict(
    exp_tag="ring_perfect",
    env_name=AccelEnv,
    network=RingNetwork,
    simulator="traci",
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params["env"].horizon = 3000
exp = Experiment(flow_params)

_ = exp.run(1)
