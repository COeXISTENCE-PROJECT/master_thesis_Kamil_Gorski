from flow.core.experiment import Experiment
from flow.core.params import NetParams, EnvParams, InitialConfig, \
                             VehicleParams, SumoParams
from flow.controllers import IDMController, RLController
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.routing_controllers import ContinuousRouter

HORIZON = 3000
N_ROLLOUTS = 20
N_CPUS = 2
N_HUMANS = 21
N_AVS = 1

vehicles = VehicleParams()
vehicles.add("human",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()

net_params = NetParams(additional_params=additional_net_params)

sim_params = SumoParams(render=False, sim_step=0.1, restart_instance=True)
initial_config = InitialConfig(spacing="uniform", bunching=10)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params = dict(
    exp_tag='ring_perfect',
    env_name=AccelEnv,
    network=RingNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config
)

flow_params['env'].horizon = HORIZON

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
import json

ray.init(num_cpus=N_CPUS)
alg_run = "PPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [4, 4]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
# config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
config["sgd_minibatch_size"] = 256  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon


flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  
config['env_config']['flow_params'] = flow_json
config['env_config']['run'] = alg_run

create_env, gym_name = make_create_env(params=flow_params, version=0)

register_env(gym_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        # "restore": "/home/kamil/ray_results/eight_rl/PPO_AccelEnv-v0_13091478_2022-09-02_12-43-26nai86dj8/checkpoint_100/checkpoint-100",
        "checkpoint_freq": 10,
        "checkpoint_at_end": True,
        "max_failures": 0,
        "stop": {
            "training_iteration": 100,
        },
    },
})