from tkinter import E
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import FigureEightNetwork
from eight_env import EightEnv

HORIZON = 1500
N_ROLLOUTS = 9
N_CPUS = 3
N_HUMANS = 0
N_AVS = 14

sim_params = SumoParams(sim_step=0.1, render=False, restart_instance=True)

vehicles = VehicleParams()

# vehicles.add(
#     veh_id='human',
#     acceleration_controller=(IDMController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode="obey_safe_speed",
#     ),
#     num_vehicles=13)

vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='obey_safe_speed',
    ),
    num_vehicles=14)

# vehicles.add(
#     veh_id='human2',
#     acceleration_controller=(IDMController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode="obey_safe_speed",
#     ),
#     num_vehicles=6)

# vehicles.add(
#     veh_id='rl2',
#     acceleration_controller=(RLController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_m+ode='obey_safe_speed',
#     ),
#     num_vehicles=1)

env_params = EnvParams(
        warmup_steps=5,
        horizon=HORIZON,
        additional_params={
            'target_velocity': 15,
            'max_accel': 1.5,
            'max_decel': 3.5,
            'sort_vehicles' : False,
        },
)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()

net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig()

flow_params = dict(
    exp_tag='eight_rl_OP',
    env_name=EightEnv,
    network=FigureEightNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    # initial=initial_config,
)



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
config["gamma"] = 0.99  # discount rate
config["model"].update({"fcnet_hiddens": [64, 64]})  # size of hidden layers in network
config["model"].update({"fcnet_activation": "tanh"})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97   
# config["sgd_minibatch_size"] = 128  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon
config['clip_actions'] = True
config["lr"] = 0.001
# config["clip_param"] = 0.2
# config["vf_clip_param"] = 100

# ES ----------------------------------------
# config["episodes_per_batch"] = N_ROLLOUTS
# config["eval_prob"] = 0.05
# # optimal parameters
# config["noise_stdev"] = 0.02
# config["stepsize"] = 0.02

# config["model"]["fcnet_hiddens"] = [100, 50, 25]
# config['clip_actions'] = False  
# config["observation_filter"] = "NoFilter"
# config["lr"] = 0.02
# ES ----------------------------------------


# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)


trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "restore": "/home/kamil/ray_results/eight_rl_OP/PPO_EightEnv-v0_cc78b818_2022-09-23_21-56-22wv41dvrr/checkpoint_390/checkpoint-390",
        "checkpoint_freq": 10,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 0,
        "stop": {  # stopping conditions
            "training_iteration": 1000,  # number of iterations to stop after
        },
    },
})