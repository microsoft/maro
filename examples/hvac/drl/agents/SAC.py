
from .base_agent import BaseAgent
from .exploration import OU_Noise
from .replay_buffer import Replay_Buffer
from .model import create_NN
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
from shutil import copy2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 2
EPSILON = 1e-6

class SAC(BaseAgent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config, env, logger):
        BaseAgent.__init__(self, config, env, logger)

        self.hyperparameters = config.hyperparameters

        self.critic_local = create_NN(
            input_dim=self.state_size + self.action_size,
            output_dim=1,
            hyperparameters=self.hyperparameters["Critic"],
            seed=self.config.seed,
            device=self.device
        )
        self.critic_local_2 = create_NN(
            input_dim=self.state_size + self.action_size,
            output_dim=1,
            hyperparameters=self.hyperparameters["Critic"],
            seed=self.config.seed + 1,
            device=self.device
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.critic_optimizer_2 = torch.optim.Adam(
            self.critic_local_2.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )

        self.critic_target = create_NN(
            input_dim=self.state_size + self.action_size,
            output_dim=1,
            hyperparameters=self.hyperparameters["Critic"],
            seed=self.config.seed,
            device=self.device
        )
        self.critic_target_2 = create_NN(
            input_dim=self.state_size + self.action_size,
            output_dim=1,
            hyperparameters=self.hyperparameters["Critic"],
            seed=self.config.seed,
            device=self.device
        )
        BaseAgent.copy_model_over(self.critic_local, self.critic_target)
        BaseAgent.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.memory = Replay_Buffer(
            self.hyperparameters["Critic"]["buffer_size"],
            self.hyperparameters["batch_size"]
        )

        self.actor_local = create_NN(
            input_dim=self.state_size,
            output_dim=self.action_size * 2,
            hyperparameters=self.hyperparameters["Actor"],
            seed=self.config.seed,
            device=self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(),
            lr=self.hyperparameters["Actor"]["learning_rate"],
            eps=1e-4
        )

        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(
                self.action_size, self.config.seed,
                self.hyperparameters["mu"], self.hyperparameters["theta"], self.hyperparameters["sigma"]
            )

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def _save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.full_episode_total_rewards.extend([self.total_episode_reward_so_far])
            self.rolling_returns.append(np.mean(self.full_episode_total_rewards[-self.rolling_return_window:]))

        elif (self.episode_number + 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.full_episode_total_rewards.extend([self.total_episode_reward_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_returns.extend([np.mean(self.full_episode_total_rewards[-self.rolling_return_window:]) for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])

        self.max_episode_score_seen = max(self.max_episode_score_seen, self.full_episode_total_rewards[-1])

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        BaseAgent.reset_game(self)
        if self.add_extra_noise:
            self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = (self.episode_number+1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self._time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self._learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                self.memory.add_experience(self.state, self.action, self.reward, self.next_state, mask)
            self.state = self.next_state
            self.env_step_number += 1
            if self.env_step_number % 1000 == 0:
                print('action: ', self.action, eval_ep, self.reward, self.done, self.env_step_number, self.episode_number)
        if eval_ep:
            self.logger.info(f"Evaluation Episode Return: {self.total_episode_reward_so_far}")

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state
        if eval_ep: action = self._actor_pick_action(state=state, eval=True)
        elif self.env_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            print("Picking random action ", action)
        else: action = self._actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def _actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self._produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self._produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def _produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        # actor_output[:, self.action_size:] = torch.abs(actor_output[:, self.action_size:])
        # actor_output[:, :self.action_size] = torch.clamp(actor_output[:, :self.action_size], -1.0, 1.0)
        # actor_output = torch.clamp(actor_output, -10.0, 10.0)
        mean = torch.clamp(actor_output[:, :self.action_size], -3.0, 3.0)
        log_std = torch.clamp(actor_output[:, self.action_size:], -5.0, 2.0)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def _time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return (
            self.env_step_number > self.hyperparameters["min_steps_before_learning"]
            and len(self.memory) > self.hyperparameters["batch_size"]
            and self.env_step_number % self.hyperparameters["update_every_n_steps"] == 0
        )

    def _learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()
        qf1_loss, qf2_loss = self._calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self._update_critic_parameters(qf1_loss, qf2_loss)

        policy_loss, log_pi = self._calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning: alpha_loss = self._calculate_entropy_tuning_loss(log_pi)
        else: alpha_loss = None
        self._update_actor_parameters(policy_loss, alpha_loss)

    def _calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self._produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def _calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self._produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def _calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def _update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        self._take_optimization_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self._take_optimization_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self._soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self._soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def _update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self._take_optimization_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self._take_optimization_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def save_model(self):
        torch.save(
            self.actor_local.state_dict(),
            os.path.join(self.config.checkpoint_dir, f"actor_{self.episode_number}.pt")
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(self.config.checkpoint_dir, f"actor_optimizer_{self.episode_number}.pt")
        )

        copy2(
            os.path.join(self.config.checkpoint_dir, f"actor_{self.episode_number}.pt"),
            os.path.join(self.config.checkpoint_dir, "actor.pt")
        )
        copy2(
            os.path.join(self.config.checkpoint_dir, f"actor_optimizer_{self.episode_number}.pt"),
            os.path.join(self.config.checkpoint_dir, "actor_optimizer.pt")
        )

    def load_model(self):
        self.actor_local.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, "actor.pt")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, "actor_optimizer.pt")))
