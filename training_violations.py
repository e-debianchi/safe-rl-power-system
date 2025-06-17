import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from stable_baselines3 import SAC
import gymnasium as gym
from agents import RCBF_SAC

# from scipy.io import loadmat
# data = loadmat("C:/Users/edebi/Documents/MATLAB/progetti/TESI/grid_values.mat")
# A = data.get('A').squeeze().astype(np.float32)
# B = data.get('B').astype(np.float32)
# P = data.get('P').astype(np.float32)

from values import A, B, P

test_env = gym.make('PowerSystem-v0', A=A, B=B, P=P, dist_type='state')

# cbf_agent = RCBF_SAC.load('saved_models/toy/rcbf_sac_power_system_25perc', env=test_env)
# safe_agent = SAC.load('saved_models/toy/safe_sac_power_system_25perc')

cbf_agent = RCBF_SAC.load('saved_models/rcbf_sac_power_system_25perc', env=test_env)
safe_agent = SAC.load('saved_models/safe_sac_power_system_25perc')

# Define the control function for the new agent
def safe_u(x):
    action = safe_agent.predict(x, deterministic=True)[0]
    action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
    return action

def cbf_u(x):
    action = cbf_agent.predict(x, deterministic=True)[0].reshape(-1)
    action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
    return action

def model(t, y, A, B, P, control, d):
    u = control(y) if control is not None else np.zeros(B.shape[1])
    
    return  np.dot(A, y) + np.dot(B, u) + np.dot(P, d(y[n//2:].reshape(1, -1)).squeeze())

t_f = 5

n = A.shape[0]
from time import time
omega_cbf = []
omega_safe = []
t_cbf = []
t_safe = []
for _ in range(10):
    y0, _ = test_env.reset(seed=int(time()))

    sol_cbf = solve_ivp(model, (0, t_f), y0, args=(A, B, P, cbf_u, test_env.unwrapped.dist), max_step=0.01, rtol=1, atol=1)
    y_cbf = sol_cbf.y.T
    omega_cbf.append(y_cbf[:, n//2:])
    t_cbf.append(sol_cbf.t)

    sol_safe = solve_ivp(model, (0, t_f), y0, args=(A, B, P, safe_u, test_env.unwrapped.dist), max_step=0.01, rtol=1, atol=1)
    y_safe = sol_safe.y.T
    omega_safe.append(y_safe[:, n//2:])
    t_safe.append(sol_safe.t)

fig, axs = plt.subplots(1, n//2, sharex=True)
if n == 2:
    axs = [axs]

for i in range(n//2):
    axs[i].plot(t_safe[0], omega_safe[0][:, i], '#D95319')
    axs[i].plot(t_cbf[0], omega_cbf[0][:, i], '#0072BD')
    for j in range(1, 10):
        axs[i].plot(t_safe[j], omega_safe[j][:, i], '#D95319', alpha=0.5)
        axs[i].plot(t_cbf[j], omega_cbf[j][:, i], '#0072BD', alpha=0.5)
    axs[i].hlines([-0.5, 0.5], 0, t_f, linestyles='dashed', colors='r', linewidth=0.9)
    axs[i].grid()
    axs[i].set_xlabel('Time [s]')
    axs[i].set_ylabel(f'$\omega$ [rad/s]')
fig.legend(labels=['Safe SAC', 'RCBF-SAC'], loc='upper right', ncol=1, bbox_to_anchor=(0.98, 0.98))
fig.tight_layout()
plt.show()

fig.savefig('images/wscc_viol_25perc.pdf')