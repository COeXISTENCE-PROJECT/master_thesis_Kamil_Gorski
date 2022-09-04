from flow.core.experiment import Experiment
from flow.core.params import NetParams, EnvParams, \
                             VehicleParams, SumoParams, SumoCarFollowingParams
from flow.controllers import IDMController
from flow.networks.rectangle_network import RectangleNetwork, ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.routing_controllers import ContinuousRouter

vehicles = VehicleParams()
vehicles.add("human",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed"),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=16)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()


net_params = NetParams(additional_params=additional_net_params)

sim_params = SumoParams(render=True, sim_step=0.2, emission_path="data")

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params = dict(
    exp_tag='rectangle',
    env_name=AccelEnv,
    network=RectangleNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
)

flow_params['env'].horizon = 2000
exp = Experiment(flow_params)

_ = exp.run(1)
