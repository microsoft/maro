import os
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad

from maro.rl import AbsAlgorithm

from .utils import gnn_union

class ActorCritic(AbsAlgorithm):
    '''

    Actor-Critic algorithm in CIM problem.

    Args:
        model (nn.Module): A actor-critic module outputing both the policy network and the value network
        device (torch.device): A PyTorch device instance where the module is computed on.
        p2p_adj (numpy.array): The static port-to-port adjencency matrix.
        td_steps (int): The value 'n' in the n-step TD algorithm.
        gamma (float): The time decay.
        learning_rate (float): The learning rate for the module.
        entropy_factor (float): The weight of the policy's entropy to boost exploration.
    '''

    def __init__(self, model: nn.Module,
                 device: torch.device,
                    p2p_adj=None, 
                    td_steps=100, 
                    gamma=0.97, 
                    learning_rate=0.0003, 
                    entropy_factor=0.1):
        self._model = model

        self._gamma = gamma
        self._td_steps = td_steps
        self._value_discount = gamma**100
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._entropy_factor = entropy_factor
        self._device = device
        self._tot_batchs = 0
        self._p2p_adj = p2p_adj
        super().__init__(model_dict={"a&c": model}, optimizer_opt={"a&c": self._optimizer}, loss_func_dict={},
                         hyper_params=None)

    def choose_action(self, state: dict, p_idx: int, v_idx: int):
        '''
        Args:
            state (dict): A dictionary containing the input to the module. For example:
                {
                    'v': v,
                    'p': p,
                    'pe': {
                        'edge': pedge,
                        'adj': padj,
                        'mask': pmask,
                    },
                    've': {
                        'edge': vedge,
                        'adj': vadj,
                        'mask': vmask,
                    },
                    'ppe': {
                        'edge': ppedge,
                        'adj': p2p_adj,
                        'mask': p2p_mask,
                    },
                    'mask': seq_mask,
                }
            p_idx (int): The identity of the port doing the action.
            v_idx (int): The identity of the vessel doing the action.
        
        Returns:
            model_action (numpy.int64): The action returned from the module
        '''
        with torch.no_grad():
            prob, _ = self._model(state, a=True, p_idx=p_idx, v_idx=v_idx)
            distribution = Categorical(prob)
            model_action = distribution.sample().cpu().numpy()
            return model_action

    def train(self, batch, p_idx, v_idx):
        '''
        Args:
            batch (dict): The dictionary of a batch of experience. For example:
                {
                    's': the dictionary of state,
                    'a': model actions in numpy array,
                    'R': the n-step accumulated reward,
                    's'': the dictionary of the next state,
                }
            p_idx (int): The identity of the port doing the action.
            v_idx (int): The identity of the vessel doing the action.
        
        Returns:
            a_loss (float): action loss.
            c_loss (float): critic loss.
            e_loss (float): entropy loss.
            tot_norm (float): the L2 norm of the gradient.

        '''
        self._tot_batchs += 1
        item_a_loss, item_c_loss, item_e_loss = 0, 0, 0
        obs_batch = batch['s']
        action_batch = batch['a']
        return_batch = batch['R']
        next_obs_batch = batch['s_']

        obs_batch = gnn_union(obs_batch['p'], obs_batch['po'], obs_batch['pedge'], obs_batch['v'], 
                                        obs_batch['vo'], obs_batch['vedge'], self._p2p_adj, obs_batch['ppedge'], 
                                        obs_batch['mask'], self._device)
        action_batch = torch.from_numpy(action_batch).long().to(self._device)
        return_batch = torch.from_numpy(return_batch).float().to(self._device)
        next_obs_batch = gnn_union(next_obs_batch['p'], next_obs_batch['po'], next_obs_batch['pedge'], 
                                        next_obs_batch['v'], next_obs_batch['vo'], next_obs_batch['vedge'],
                                        self._p2p_adj, next_obs_batch['ppedge'], next_obs_batch['mask'], 
                                        self._device)

        # train actor network
        # self._actor_optimizer.zero_grad()
        # self._critic_optimizer.zero_grad()
        self._optimizer.zero_grad()

        # every port has a value
        # values.shape: (batch, p_cnt)

        probs, values = self._model(obs_batch, a=True, p_idx=p_idx, v_idx=v_idx, c=True)
        distribution = Categorical(probs)
        log_prob = distribution.log_prob(action_batch)
        entropy_loss = distribution.entropy()

        _, values_ = self._model(next_obs_batch, c=True)
        advantage = return_batch + self._value_discount * values_.detach() - values

        if self._entropy_factor != 0:
            # actor_loss = actor_loss* torch.log(entropy_loss + np.e)
            advantage[:, p_idx] += self._entropy_factor*entropy_loss.detach()

        actor_loss = - (log_prob*torch.sum(advantage, axis=-1).detach()).mean()

        # actor_loss.backward(retain_graph=True)
        # self._actor_optimizer.step()
        
        item_a_loss = actor_loss.item()
        item_e_loss = entropy_loss.mean().item()

        # train critic network
        critic_loss = torch.sum(advantage.pow(2), axis=1).mean()
        # critic_loss.backward()
        item_c_loss = critic_loss.item()
        # torch.nn.utils.clip_grad_norm_(self._critic_model.parameters(),0.5)
        # self._critic_optimizer.step()
        tot_loss = 0.1*actor_loss + critic_loss # - self._entropy_factor * entropy_loss
        tot_loss.backward()
        tot_norm = clip_grad.clip_grad_norm_(self._model.parameters(), 1)
        self._optimizer.step()
        return item_a_loss, item_c_loss, item_e_loss, float(tot_norm)

    def set_weights(self, weights):
        self._model.load_state_dict(weights)

    def get_weights(self):
        return self._model.state_dict()

    def _get_save_idx(self, fp_str):
        return int(fp_str.split('.')[0].split('_')[0])

    def save_model(self, pth, id):
        if not os.path.exists(pth):
            os.makedirs(pth)
        pth = os.path.join(pth, '%d_ac.pkl'%id)
        torch.save(self._model.state_dict(), pth)
    
    def _set_gnn_weights(self, weights):
        for key in weights:
            if key in self._model.state_dict().keys():
                self._model.state_dict()[key].copy_(weights[key])

    def load_model(self, folder_pth, idx=-1):
        if idx == -1:
            fps = os.listdir(folder_pth)
            fps = [f for f in fps if 'ac' in f]
            fps.sort(key=self._get_save_idx)
            ac_pth = fps[-1]
        else:
            ac_pth = '%d_ac.pkl'%idx
        pth = os.path.join(folder_pth, ac_pth)
        with open(pth, 'rb') as fp:
            weights = torch.load(fp, map_location=self._device)
        self._set_gnn_weights(weights)