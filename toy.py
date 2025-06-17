import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
from agents import RCBF_SAC

seed = 7
set_random_seed(seed)

from scipy.io import loadmat
data = loadmat("grid_values.mat")
A = data.get('A').squeeze().astype(np.float32)
B = data.get('B').astype(np.float32)
P = data.get('P').astype(np.float32)

n_envs = 10
env = make_vec_env('PowerSystem-v0', env_kwargs=dict(A=A, B=B, P=P, dist_type='state', render_mode=None),
                   n_envs=n_envs, vec_env_cls=DummyVecEnv, seed=seed, monitor_dir='logs/')
_ = env.reset()

safe_env = make_vec_env('SafePowerSystem-v0', env_kwargs=dict(A=A, B=B, P=P, dist_type='state', render_mode=None),
                        n_envs=n_envs, vec_env_cls=DummyVecEnv, seed=seed, monitor_dir='safe_logs/')
_ = safe_env.reset()

cbf_env = make_vec_env('PowerSystem-v0', env_kwargs=dict(A=A, B=B, P=P, dist_type='state', render_mode=None),
                        n_envs=n_envs, vec_env_cls=DummyVecEnv, seed=seed, monitor_dir='cbf_logs/')
_ = cbf_env.reset()

policy_kwargs = dict(net_arch=dict(pi=[16], qf=[16]))

timesteps = 1e4
learn_start = 100
lr = 1e-2

cbf_agent = RCBF_SAC(policy="MlpPolicy", env=cbf_env, device='cpu', learning_rate=lr, learning_starts=learn_start, seed=seed,
                 policy_kwargs=policy_kwargs, kappa=10.0) # 2 maybe not the best, 3 similar, trying 10

sac_agent = SAC("MlpPolicy", env, device='cpu', learning_rate=lr, learning_starts=learn_start, seed=seed,
                policy_kwargs=policy_kwargs)

safe_agent = SAC("MlpPolicy", safe_env, device='cpu', learning_starts=learn_start, learning_rate=lr, seed=seed,
                 policy_kwargs=policy_kwargs)

sac_agent.learn(total_timesteps=timesteps, progress_bar=True)
sac_agent.save('saved_models/toy/sac_power_system')

safe_agent.save('saved_models/toy/safe_sac_power_system_0perc')
safe_agent.learn(total_timesteps=timesteps//10)
safe_agent.save('saved_models/toy/safe_sac_power_system_10perc')
safe_agent.learn(total_timesteps=(timesteps//4-timesteps//10))
safe_agent.save('saved_models/toy/safe_sac_power_system_25perc')
safe_agent.learn(total_timesteps=(timesteps-timesteps//4), progress_bar=True)
safe_agent.save('saved_models/toy/safe_sac_power_system')

cbf_agent.learn(total_timesteps=learn_start+1)
cbf_agent.save('saved_models/toy/rcbf_sac_power_system_0perc')
cbf_agent.learn(total_timesteps=(timesteps//10-learn_start-1))
cbf_agent.save('saved_models/toy/rcbf_sac_power_system_10perc')
cbf_agent.learn(total_timesteps=(timesteps//4-timesteps//10))
cbf_agent.save('saved_models/toy/rcbf_sac_power_system_25perc')
cbf_agent.learn(total_timesteps=(timesteps-timesteps//4), progress_bar=True)
cbf_agent.save('saved_models/toy/rcbf_sac_power_system')