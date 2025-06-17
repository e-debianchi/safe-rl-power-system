import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class PowerSystemEnv(gym.Env):
    def __init__(self, A, B, P, act_limits=1., dist_type='state', render_mode=None, max_episode_steps=500):
        super(PowerSystemEnv, self).__init__()
        self.n_states = A.shape[0]
        self.n_actions = B.shape[1]
        self.n_dist = P.shape[1]
        self.dist_type = dist_type
        self.render_mode = render_mode
        self.dt = 0.01
        self.n_gen = self.n_states // 2
        self.max_episode_steps = max_episode_steps

        # Define action and observation space
        act_limits = np.full((self.n_actions), act_limits, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-act_limits, high=act_limits, shape=(self.n_actions,), dtype=np.float32)

        obs_limits = np.concatenate([np.full((self.n_gen,), 6*np.pi), np.full((self.n_gen,), 5.)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_limits, high=obs_limits, shape=(self.n_states,), dtype=np.float32)

        # System matrices A, B, P
        self.A = A
        self.B = B
        self.P = P

        # Initialize disturbance function
        if self.dist_type != 'const':
            self.dist = self.init_dist()

    def init_dist(self):
        if self.dist_type == 'none':
            return lambda _: np.zeros(self.n_dist)
        if self.dist_type == 'const':
            k = self.n_dist//2
            arr = np.zeros((self.n_dist))
            arr[np.random.choice(self.n_dist, k, replace=False)] = np.random.uniform(-0.3, 0.3, k)
            return lambda _: arr
        if self.dist_type == 'state':
            if self.n_dist == 6:
                return lambda omega: np.array([0.6*np.sum(np.sin(omega), axis=1) + 0.2*np.sum(np.abs(omega), axis=1), 
                                                0.3*np.sum(np.sin(omega), axis=1) + 0.3*np.sum(np.abs(omega), axis=1),
                                                -0.4*np.sum(np.sin(omega), axis=1) - 0.4*np.sum(np.abs(omega), axis=1),
                                                -0.2*np.sum(np.sin(omega), axis=1) - 0.3*np.sum(np.abs(omega), axis=1),
                                                -0.5*np.sum(np.sin(omega), axis=1) + 0.2*np.sum(np.abs(omega), axis=1),
                                                0.3*np.sum(np.sin(omega), axis=1) - 0.4*np.sum(np.abs(omega), axis=1)]) / 3.0
            elif self.n_dist == 5:
                return lambda omega: np.array([0.6*np.sum(np.sin(omega), axis=1) + 0.2*np.sum(np.abs(omega), axis=1), 
                                                0.3*np.sum(np.sin(omega), axis=1) + 0.3*np.sum(np.abs(omega), axis=1),
                                                -0.4*np.sum(np.sin(omega), axis=1) - 0.4*np.sum(np.abs(omega), axis=1),
                                                -0.2*np.sum(np.sin(omega), axis=1) - 0.3*np.sum(np.abs(omega), axis=1),
                                                0.3*np.sum(np.sin(omega), axis=1) - 0.4*np.sum(np.abs(omega), axis=1)]) / 3.0
        raise ValueError(f'Disturbance type "{self.dist_type}" not allowed. Choose from ["none", "const", "state"].')

    def step(self, act):
        act = np.clip(act, self.action_space.low, self.action_space.high)

        d = self.dist(self.state[self.n_gen:].reshape(1, -1)).squeeze()
        
        f_x = (np.matmul(self.A, self.state.T).T + np.matmul(self.B, act.T).T + np.matmul(self.P, d)).squeeze()

        self.state = self.state + self.dt*f_x

        delta, omega = self.state[:self.n_gen], self.state[self.n_gen:]
        reward = -np.square(np.linalg.norm(omega))
        reward -= 0.1 * np.square(np.linalg.norm(delta))

        if np.linalg.norm(omega) < 0.1:
            reward += 50
        if np.linalg.norm(omega) < 0.01:
            reward += 200

        assert np.all(np.isfinite(self.state)), "State contains NaNs or Infs"
        assert np.isfinite(reward), "Reward is NaN or Inf"

        done = False
        if np.linalg.norm(omega) < 1e-3:
            if self.patience < 5:
                self.patience += 1
                reward += 500
            else:
                done = True
                reward += 5000 - 10*self.time_steps
        else:
            self.patience = 0

        self.time_steps += 1

        truncated = (self.time_steps >= self.max_episode_steps) or (np.linalg.norm(self.state) > 1e3)
        if truncated:
            reward = -5000

        term_obs = self.state if done or truncated else None
        info = dict(is_success=done, terminal_observation=term_obs)

        return self.state.astype(np.float32), reward, done or truncated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            super().reset(seed=seed, options=options)

        delta = np.random.uniform(low=-1., high=1., size=(self.n_gen,))
        omega = np.random.uniform(low=-0.499, high=0.499, size=(self.n_gen,))
        self.state = np.concatenate((delta, omega))
        if self.dist_type == 'const':
            self.dist = self.init_dist()
        self.time_steps = 0
        self.patience = 0

        return self.state.astype(np.float32), {}

    def render(self, mode='human'):
        pass

class SafePowerSystemEnv(PowerSystemEnv):
    def __init__(self, A, B, P, act_limits=1., dist_type='none', render_mode=None):
        super(SafePowerSystemEnv, self).__init__(A, B, P, act_limits, dist_type, render_mode)

    def _h(self, x):
        omega = x[self.n_gen:]
        return np.min(0.5**2 - np.square(omega))

    def step(self, act):
        next_state, reward, done, truncated, info = super().step(act)
        h_val = self._h(next_state)
        if h_val < 0:
            reward -= 15
        return next_state, reward, done, truncated, info
    
# Example usage
if __name__ == "__main__":
    from scipy.io import loadmat
    data = loadmat("C:/Users/edebi/Documents/MATLAB/progetti/TESI/grid_values.mat")
    A = data.get('A').squeeze()
    B = data.get('B')
    P = data.get('P')

    # env = gym.make('PowerSystem-v0', A=A, B=B, P=P, dist_type='const', max_episode_steps=300, render_mode='human')
    env = PowerSystemEnv(A=A, B=B, P=P, dist_type='const', render_mode='human')
    state = env.reset()
    states = [state]
    for _ in range(100):
        action = env.unwrapped.action_space.sample()  # Example action
        state, reward, done, _, _ = env.step(action)
        states.append(state)

    plt.plot(states)
    env.close()