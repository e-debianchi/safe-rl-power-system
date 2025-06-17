import numpy as np
import torch
from torch.nn import functional as F
from typing import Union, Optional
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import  RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import VecEnv

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import Mean, MultitaskMean
from gpytorch.kernels import MultitaskKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from buffer import TransBuffer
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

class Scaler:
    def fit_transform(self, x):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        return (x - self.mean) / self.std
    def inverse_transform(self, x):
        return x * self.std + self.mean

class AbsoluteValueMean(Mean):
    def forward(self, x):
        return x.abs().sum(dim=-1)
    
class SinMean(Mean):
    def forward(self, x):
        return x.sin().sum(dim=-1)
    
class ComposedMean(Mean):
    def __init__(self):
        super().__init__()
        self.abs_mean = AbsoluteValueMean()
        self.sin_mean = SinMean()

        self.weight_abs = torch.nn.Parameter(torch.tensor(0.5))
        self.weight_sin = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.weight_abs * self.abs_mean(x) + self.weight_sin * self.sin_mean(x)
    
class MultiTaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultiTaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ComposedMean(), num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(MaternKernel(nu=2.5), num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

class RCBF_SAC(SAC):
    def __init__(self, l=10, limits=0.5, kappa=2., *args, **kwargs):
        super(RCBF_SAC, self).__init__(*args, **kwargs)
        self.l = l
        self.limits = limits
        self.kappa = kappa
        self.n_states = self.env.get_attr('n_states')[0]
        self.n_actions = self.env.get_attr('n_actions')[0]
        self.n_gen = self.n_states // 2
        self.dt = self.env.get_attr('dt')[0]

        # Buffer for GPR training
        self.noise_buf = TransBuffer(self.env.observation_space.shape[0]//2)
        self.gp_trained = False

        # Define the cvxpy layer
        u = cp.Variable(self.n_actions)
        A_param = cp.Parameter(self.n_actions+1)
        if self.n_gen == 1:
            A_param = cp.Parameter(self.n_actions)
            b_param = cp.Parameter()
            constraints = [A_param @ u >= b_param]
            objective = cp.Minimize(cp.square(cp.norm(u, 2)))
            prob = cp.Problem(objective, constraints)
            assert prob.is_dpp(), 'Problem must be DPP!'
            self.cvxpylayer = CvxpyLayer(prob, parameters=[A_param, b_param], variables=[u])
        elif self.n_gen == 3:
            A1 = cp.Parameter(self.n_actions)
            A2 = cp.Parameter(self.n_actions)
            A3 = cp.Parameter(self.n_actions)
            b1 = cp.Parameter()
            b2 = cp.Parameter()
            b3 = cp.Parameter()
            constraints = [A1 @ u >= b1, A2 @ u >= b2, A3 @ u >= b3]
            objective = cp.Minimize(cp.square(cp.norm(u, 2)))
            prob = cp.Problem(objective, constraints)
            assert prob.is_dpp(), 'Problem must be DPP!'
            self.cvxpylayer = CvxpyLayer(prob, parameters=[A1, b1, A2, b2, A3, b3], variables=[u])
        else:
            raise NotImplementedError('Only 1 or 3 generators are supported!')
        
    def train_gp(self, data):
        x_train, y_train = data['obs'], data['out']
        self.scaler = Scaler()
        y_train_scaled = self.scaler.fit_transform(y_train)
        
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.n_gen)
        self.model = MultiTaskGPModel(x_train, y_train_scaled, self.likelihood, self.n_gen)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(500):
            optimizer.zero_grad()
            output = self.model(x_train)
            loss = -mll(output, y_train_scaled)
            loss.backward()
            optimizer.step()

    def gp_predict(self, x_test):
        if not self.gp_trained:
            self.train_gp(self.noise_buf.sample_batch(300))
            self.gp_trained = True

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x_test))
        return self.scaler.inverse_transform(pred.mean)

    def h(self, state):
        return self.limits**2 - torch.square(state[:, self.n_gen:])

    def dh(self, state):
        H = self.limits**2 - torch.square(state[:, self.n_gen:])
        min_value, _ = torch.min(H, dim=1, keepdim=True)
        dh = torch.zeros_like(state)
        tolerance = 0.05
        mask = (torch.abs(H - min_value) < tolerance).float()                   # mask for the active constraints
        dh[:, self.n_gen:] = -2 * state[:, self.n_gen:] * mask.squeeze()
        return dh
    
    def _input_control(self, tens):
        if not torch.is_tensor(tens):
            tens = torch.tensor(tens, dtype=torch.float32).clone().detach().requires_grad_(True)
        if tens.dim() == 1:
            tens = tens.unsqueeze(0)
        return tens
    
    def _dynamics(self, state, action):
        A_dynamics = torch.as_tensor(self.env.get_attr('A')[0]).float()
        B_dynamics = torch.as_tensor(self.env.get_attr('B')[0]).float()

        f_func = torch.matmul(A_dynamics, torch.as_tensor(state).T).T
        g_func = torch.matmul(B_dynamics, torch.as_tensor(action).T).T
        return f_func + g_func
    
    def _get_dist(self, states):
        batch_size = states.shape[0]
        d_hat = self.gp_predict(states[:, self.n_gen:].detach()).reshape((batch_size, self.n_gen))
        return torch.hstack([torch.zeros((batch_size, self.n_gen)), d_hat]).float()

    def solve_qp(self, states, u_rl):
        states = self._input_control(states)
        u_rl = self._input_control(u_rl)
        batch_size = states.shape[0]

        nabla_h = self.dh(states)
        B_dynamics = torch.as_tensor(self.env.get_attr('B')[0]).float()

        f_g = self._dynamics(states, u_rl)
        d_func = self._get_dist(states)

        if self.n_gen == 1:
            A = torch.matmul(nabla_h, B_dynamics).float()
            b = (-torch.einsum('bi, bi -> b', nabla_h, (f_g + d_func)) - self.kappa * self.h(states)).float()
            u_opt, = self.cvxpylayer(A, b, solver_args={'solve_method': 'Clarabel'})
        elif self.n_gen == 3:
            nabla_1, nabla_2, nabla_3 = torch.zeros_like(nabla_h), torch.zeros_like(nabla_h), torch.zeros_like(nabla_h)
            nabla_1[:, 3] = nabla_h[:, 3]
            nabla_2[:, 4] = nabla_h[:, 4]
            nabla_3[:, 5] = nabla_h[:, 5]
            
            h1, h2, h3 = torch.unbind(self.h(states), dim=1)
            h1 = h1 * (nabla_1[:, 3] != 0)
            h2 = h2 *(nabla_2[:, 4] != 0)
            h3 = h3 * (nabla_3[:, 5] != 0)

            A1 = torch.matmul(nabla_1, B_dynamics)
            A2 = torch.matmul(nabla_2, B_dynamics)
            A3 = torch.matmul(nabla_3, B_dynamics)
            b1 = (-torch.einsum('bi, bi -> b', nabla_1, (f_g + d_func)) - self.kappa * h1).float()
            b2 = (-torch.einsum('bi, bi -> b', nabla_2, (f_g + d_func)) - self.kappa * h2).float()
            b3 = (-torch.einsum('bi, bi -> b', nabla_3, (f_g + d_func)) - self.kappa * h3).float()
            
            u_opt, = self.cvxpylayer(A1, b1, A2, b2, A3, b3, solver_args={'solve_method': 'Clarabel'})

        return u_opt.reshape(batch_size, self.n_actions).float()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions) # replay_data.actions contains u_s

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
            # Min over all critic networks
            actions_safe = self.solve_qp(replay_data.observations, actions_pi)
            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi + actions_safe), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, train_freq: TrainFreq, replay_buffer: ReplayBuffer,
                         action_noise: Optional[ActionNoise] = None, learning_starts: int = 0, log_interval: Optional[int] = None) -> RolloutReturn:

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Store transition in GPR buffer
            if not self.gp_trained:
                s0 = self._last_obs
                s1 = new_obs
                x_dot = (s1 - s0) / self.dt
                f_g = self._dynamics(s0, actions).detach()
                actions = actions.reshape(-1, self.n_actions)
                self.noise_buf.store(torch.as_tensor(s0[:, self.n_gen:]),
                                     torch.as_tensor(x_dot[:, self.n_gen:]) - f_g[:, self.n_gen:])

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
    
    def predict(self, observation: Union[np.ndarray, dict[str, np.ndarray]], state: Optional[tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        
        u_rl, state = self.policy.predict(observation, state, episode_start, deterministic)
        u_s = self.solve_qp(observation, u_rl)
        u_star = (u_rl + u_s.detach().numpy()).reshape(-1, self.n_actions)
        return u_star, state