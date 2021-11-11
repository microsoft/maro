from collections import namedtuple, deque
import random
import torch
import numpy as np

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data.
        The data saved in the order: (state, action, reward, next_state, done).
    """

    def __init__(self, buffer_size, batch_size, device=None):
        self.state = deque(maxlen=buffer_size)
        self.action = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.next_state = deque(maxlen=buffer_size)
        self.done = deque(maxlen=buffer_size)

        self.batch_size = batch_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert len(states) == len(actions) == len(rewards) == len(next_states) == len(dones)
            self.state.extend(states)
            self.action.extend(actions)
            self.reward.extend(rewards)
            self.next_state.extend(next_states)
            self.done.extend([int(done) for done in dones])
        else:
            self.state.append(states)
            self.action.append(actions)
            self.reward.append(rewards)
            self.next_state.append(next_states)
            self.done.append(int(dones))

    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        batch_size = num_experiences if num_experiences else self.batch_size
        indexes = np.random.choice(len(self.done), size=batch_size, replace=False)

        def get_data(q):
            return torch.from_numpy(np.vstack([q[i] for i in indexes])).float().to(self.device)

        return (
            get_data(self.state),
            get_data(self.action),
            get_data(self.reward),
            get_data(self.next_state),
            get_data(self.done)
        )

    def __len__(self):
        return len(self.done)
