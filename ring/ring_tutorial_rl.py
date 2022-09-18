from flow.networks import RingNetwork
from flow.core.params import NetParams, InitialConfig
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.controllers import RLController
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.envs import WaveAttenuationPOEnv


HORIZON=100
N_CPUS = 2
N_ROLLOUTS = 1

network_name = RingNetwork
env_name = WaveAttenuationPOEnv
name = "ring_tutorial"
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
initial_config = InitialConfig(spacing="uniform", perturbation=0)

vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=21)
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1)



sim_params = SumoParams(sim_step=0.1, render=False)

env_params = EnvParams(
    horizon=HORIZON,

    additional_params={
        "max_accel": 1,
        "max_decel": 1,
        "ring_length": [220, 270],
    },
)

flow_params = dict(
    exp_tag=name,
    env_name=env_name,
    network=network_name,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config
)


import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


ray.init(num_cpus=N_CPUS)
alg_run = "PPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1 
config["train_batch_size"] = HORIZON * N_ROLLOUTS
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]}) 
config["use_gae"] = True 
config["lambda"] = 0.97  
config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])
config["kl_target"] = 0.02 
config["num_sgd_iter"] = 10
config["horizon"] = HORIZON


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
        "checkpoint_freq": 1,
        "checkpoint_at_end": True,
        "max_failures": 0,
        "stop": {
            "training_iteration": 1000,
        },
    },
})