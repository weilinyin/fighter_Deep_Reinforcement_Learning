from stable_baselines3.ppo import PPO
import torch.nn as nn
from torch.nn import functional as F
import torch as th
from math import cos , exp
from MyEnvs import relative
import numpy as np
from MyEnvs import FighterEnv_2D
from typing import Optional , NamedTuple
from collections.abc import Generator
from stable_baselines3.common.buffers import RolloutBuffer

from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


from stable_baselines3.common.utils import obs_as_tensor , explained_variance
from gymnasium import spaces

class CustomRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    expert_actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    expert_log_probs: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class CustomRolloutBuffer(RolloutBuffer):
    #魔改了buffer，增加了专家策略
    expert_actions:np.ndarray
    expert_log_probs :np.ndarray


    def get(self, batch_size: Optional[int] = None) -> Generator[CustomRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "expert_actions",
                "values",
                "log_probs",
                "expert_log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.expert_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.expert_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        expert_action: np.ndarray ,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        expert_log_prob: th.Tensor,

    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.expert_actions[self.pos] = np.array(expert_action)  # Add expert action to the buffer
        self.expert_log_probs[self.pos] = expert_log_prob.clone().cpu().numpy()  # Add expert log probability to the buffer



        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> CustomRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.expert_actions[batch_inds],  # Include expert actions in the samples
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.expert_log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return CustomRolloutBufferSamples(*tuple(map(self.to_torch, data)))


















class expert_generator:
    def __init__(self ,env:FighterEnv_2D ,c_f=-0.002):
        self.env = env
        self.t_0 = env.t_0
        self.c_0 = self.env.FD.dq_z
        self.t_f = self.t_0 - self.env.FD.r / self.env.FD.dr -0.5
        self.c_f = c_f


    def generate(self ,t):
        
        if t < self.t_f:

            k_FD = self.env.FD.dr / self.env.FD.r
            r_FD = self.env.FD.r
            q_FD = self.env.FD.q_z
            psi_VF = self.env.FD.target.psi

            
            
            

            a_E1 = 2 * k_FD * r_FD * exp(k_FD *(self.t_f - t)) / (1-exp(2*k_FD *(self.t_f - self.t_0)))
            a_E2 = (self.c_0 * exp(k_FD * (self.t_f - self.t_0)) - self.c_f) / cos(q_FD - psi_VF)

            a_E = a_E1 * a_E2 / (9.81*2)

            return a_E
        else:
            
            self.env.FT.proportional_navigation()
            _ , a_E = self.env.FT.calculate_a()

            return a_E /(9.81*2)


    def sample(self):
        a = self.generate(self.env.t)

        return np.array([[a]] , dtype=np.float32)


        

        

        
        


class GAILDiscriminator(nn.Module):
    """状态-动作对判别器"""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states, actions):
        return self.net(th.cat((states , actions) , dim=-1))
    





class GAIL_PPO(PPO):
    rollout_buffer: CustomRolloutBuffer
    def __init__(self, expert_generator:expert_generator , N_gail, **kwargs):
        super().__init__(rollout_buffer_class=CustomRolloutBuffer, **kwargs)
        # 初始化判别器
        self.discriminator = GAILDiscriminator(
            self.observation_space, 
            self.action_space
        )
        self.disc_optimizer = th.optim.Adam(
            self.discriminator.parameters(), 
            lr=2e-3
        )
        self.expert_generator = expert_generator
        self.N_gail = N_gail
        

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: CustomRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            expert_actions = self.expert_generator.sample()
            expert_actions_tensor = th.from_numpy(expert_actions)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                _ , expert_log_probs , _ = self.policy.evaluate_actions(obs_tensor, expert_actions_tensor)

            actions = actions.cpu().numpy()



            



            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value


            

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                expert_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                expert_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    




    
    def train(self) -> None:
        """
        魔改了一下，加入了专家动作expert_actions

        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        episode = self.expert_generator.env.episode
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                
                if episode <= self.N_gail:
                    observations = rollout_data.observations
                    expert_actions = rollout_data.expert_actions

                    D_output_actor = th.zeros(actions.shape[0], actions.shape[1])
                    D_output_expert = th.zeros(actions.shape[0], actions.shape[1])

                    for steps in range(self.batch_size):
                        D_output_actor[steps] = self.discriminator(observations[steps] , actions[steps])
                        D_output_expert[steps] = self.discriminator(observations[steps] , expert_actions[steps])

                    
                    loss_D = -th.sum(th.log(D_output_expert) + th.log(1 - D_output_actor) , dim=0)/self.batch_size


                    self.disc_optimizer.zero_grad()
                    loss_D.backward()
                    th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                    self.disc_optimizer.step()


                

                



                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                



                


                

                if episode <= self.N_gail:
                   # GAIL loss calculation
                   _ , expert_log_probs , _ = self.policy.evaluate_actions(rollout_data.observations , expert_actions)
                   loss_gail = th.sum(expert_log_probs * th.log(1-D_output_actor.detach().flatten()) , dim=0)/self.batch_size
                   omega = 1/(1+np.exp(0.05*(episode-self.N_gail/2)))
                   loss = (1-omega)*loss +omega * loss_gail


                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
