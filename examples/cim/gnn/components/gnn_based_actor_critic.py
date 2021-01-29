import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad

from maro.rl import AbsAgent, AbsLearningModel
from maro.utils import DummyLogger

from examples.cim.gnn.components.numpy_store import Shuffler


class GNNBasedActorCriticConfig:
    """Configuration for the GNN-based Actor Critic algorithm.
    
    Args:
        p2p_adj (numpy.array): Adjencency matrix for static nodes.
        num_batches (int): number of batches to train the DQN model on per call to ``train``.
        batch_size (int): mini-batch size.
        td_steps (int): The value "n" in the n-step TD algorithm.
        gamma (float): The time decay.
        actor_loss_coefficient (float): Coefficient for actor loss in total loss.
        entropy_factor (float): The weight of the policy"s entropy to boost exploration.
    """
    __slots__ = [
        "p2p_adj", "num_batches", "batch_size", "td_steps", "reward_discount", "value_discount", 
        "actor_loss_coefficient", "entropy_factor"
    ]

    def __init__(
        self,
        p2p_adj: np.ndarray,
        num_batches: int,
        batch_size: int,
        td_steps: int = 100, 
        reward_discount: float = 0.97,
        actor_loss_coefficient: float = 0.1,
        entropy_factor: float = 0.1
    ):
        self.p2p_adj = p2p_adj
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.td_steps = td_steps
        self.reward_discount = reward_discount
        self.value_discount = reward_discount ** 100
        self.actor_loss_coefficient = actor_loss_coefficient
        self.entropy_factor = entropy_factor


class GNNBasedActorCritic(AbsAgent):
    """Actor-Critic algorithm in CIM problem.

    The vanilla ac algorithm.

    Args:
        model (AbsLearningModel): A actor-critic module outputing both the policy network and the value network.
        config (GNNBasedActorCriticConfig): Configuration for the GNN-based actor critic algorithm.
    """

    def __init__(
        self, name, model: AbsLearningModel, config: GNNBasedActorCriticConfig, experience_pool, logger=DummyLogger()
    ):
        super().__init__(name, model, config, experience_pool=experience_pool)
        self._batch_count = 0
        self._logger = logger

    def choose_action(self, state: dict):
        """Get action from the AC model.

        Args:
            state (dict): A dictionary containing the input to the module. For example:
                {
                    "v": v,
                    "p": p,
                    "pe": {
                        "edge": pedge,
                        "adj": padj,
                        "mask": pmask,
                    },
                    "ve": {
                        "edge": vedge,
                        "adj": vadj,
                        "mask": vmask,
                    },
                    "ppe": {
                        "edge": ppedge,
                        "adj": p2p_adj,
                        "mask": p2p_mask,
                    },
                    "mask": seq_mask,
                }

        Returns:
            model_action (numpy.int64): The action returned from the module.
        """
        model_state = self.union(
            state["p"], state["po"], state["pedge"], state["v"], state["vo"], state["vedge"],
            state["ppedge"], state["mask"]
        )
        action_prob, _ = self._model(
            model_state, p_idx=state["p_idx"], v_idx=state["v_idx"], use_actor=True, is_training=False
        )
        action_prob = Categorical(action_prob)
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0]) if len(action) == 1 else (action, log_p)

    def train(self):
        for (p_idx, v_idx), exp_pool in self._experience_pool.items():
            loss_dict = defaultdict(list)
            for _ in range(self._config.num_batches):
                shuffler = Shuffler(exp_pool, batch_size=self._config.batch_size)
                while shuffler.has_next():
                    batch = shuffler.next()
                    actor_loss, critic_loss, entropy_loss, tot_loss = self._train_on_batch(
                        batch["s"], batch["a"], batch["R"], batch["s_"], p_idx, v_idx
                    )
                    loss_dict["actor"].append(actor_loss)
                    loss_dict["critic"].append(critic_loss)
                    loss_dict["entropy"].append(entropy_loss)
                    loss_dict["tot"].append(tot_loss)

            if loss_dict:
                a_loss = np.mean(loss_dict["actor"])
                c_loss = np.mean(loss_dict["critic"])
                e_loss = np.mean(loss_dict["entropy"])
                tot_loss = np.mean(loss_dict["tot"])
                self._logger.debug(
                    f"code: {str(self._name)} \t actor: {float(a_loss)} \t critic: {float(c_loss)} \t entropy: {float(e_loss)} \
                    \t tot: {float(tot_loss)}")

            exp_pool.clear()
    
    def _train_on_batch(self, states, actions, returns, next_states, p_idx, v_idx):
        """Model training.

        Args:
            batch (dict): The dictionary of a batch of experience. For example:
                {
                    "s": the dictionary of state,
                    "a": model actions in numpy array,
                    "R": the n-step accumulated reward,
                    "s"": the dictionary of the next state,
                }
            p_idx (int): The identity of the port doing the action.
            v_idx (int): The identity of the vessel doing the action.

        Returns:
            a_loss (float): action loss.
            c_loss (float): critic loss.
            e_loss (float): entropy loss.
            tot_norm (float): the L2 norm of the gradient.
        """
        self._batch_count += 1
        states, actions, returns, next_states = self._preprocess(states, actions, returns, next_states)
        # Every port has a value.
        # values.shape: (batch, p_cnt)
        probs, values = self._model(states, p_idx=p_idx, v_idx=v_idx, use_actor=True, use_critic=True)
        distribution = Categorical(probs)
        log_prob = distribution.log_prob(actions)
        entropy_loss = distribution.entropy()

        _, values_ = self._model(next_states, use_critic=True)
        advantage = returns + self._config.value_discount * values_.detach() - values

        if self._config.entropy_factor != 0:
            # actor_loss = actor_loss* torch.log(entropy_loss + np.e)
            advantage[:, p_idx] += self._config.entropy_factor * entropy_loss.detach()

        actor_loss = -(log_prob * torch.sum(advantage, axis=-1).detach()).mean()
        critic_loss = torch.sum(advantage.pow(2), axis=1).mean()
        # torch.nn.utils.clip_grad_norm_(self._critic_model.parameters(),0.5)
        tot_loss = self._config.actor_loss_coefficient * actor_loss + critic_loss
        self._model.learn(tot_loss)
        tot_norm = clip_grad.clip_grad_norm_(self._model.parameters(), 1)
        return actor_loss.item(), critic_loss.item(), entropy_loss.mean().item(), float(tot_norm)

    def _get_save_idx(self, fp_str):
        return int(fp_str.split(".")[0].split("_")[0])

    def save_model(self, pth, id):
        if not os.path.exists(pth):
            os.makedirs(pth)
        pth = os.path.join(pth, f"{id}_ac.pkl")
        torch.save(self._model.state_dict(), pth)

    def _set_gnn_weights(self, weights):
        for key in weights:
            if key in self._model.state_dict().keys():
                self._model.state_dict()[key].copy_(weights[key])

    def store_experiences(self, experiences):
        for code, exp_list in experiences.items():
            self._experience_pool[code].put(exp_list)
    
    def load_model(self, folder_pth, idx=-1):
        if idx == -1:
            fps = os.listdir(folder_pth)
            fps = [f for f in fps if "ac" in f]
            fps.sort(key=self._get_save_idx)
            ac_pth = fps[-1]
        else:
            ac_pth = f"{idx}_ac.pkl"
        pth = os.path.join(folder_pth, ac_pth)
        with open(pth, "rb") as fp:
            weights = torch.load(fp, map_location=self._device)
        self._set_gnn_weights(weights)

    def union(self, p, po, pedge, v, vo, vedge, ppedge, seq_mask):
        """Union multiple graphs in CIM.

        Args:
            v: Numpy array of shape (seq_len, batch, v_cnt, v_dim).
            vo: Numpy array of shape (batch, v_cnt, p_cnt).
            vedge: Numpy array of shape (batch, v_cnt, p_cnt, e_dim).
        Returns:
            result (dict): The dictionary that describes the graph.
        """
        if len(p.shape) == 3:
            p, po, pedge = np.expand_dims(p, 1), np.expand_dims(po, 0), np.expand_dims(pedge, 0)
            v, vo, vedge = np.expand_dims(v, 1), np.expand_dims(vo, 0), np.expand_dims(vedge, 0)
            ppedge = np.expand_dims(ppedge, 0)
            seq_mask = np.expand_dims(seq_mask, 0)

        seq_len, batch, v_cnt, v_dim = v.shape
        _, _, p_cnt, p_dim = p.shape

        p = torch.from_numpy(p).float().to(self._device)
        po = torch.from_numpy(po).long().to(self._device)
        pedge = torch.from_numpy(pedge).float().to(self._device)
        v = torch.from_numpy(v).float().to(self._device)
        vo = torch.from_numpy(vo).long().to(self._device)
        vedge = torch.from_numpy(vedge).float().to(self._device)
        p2p = torch.from_numpy(self._config.p2p_adj).to(self._device)
        ppedge = torch.from_numpy(ppedge).float().to(self._device)
        seq_mask = torch.from_numpy(seq_mask).bool().to(self._device)

        batch_range = torch.arange(batch, dtype=torch.long).to(self._device)
        # vadj.shape: (batch*v_cnt, p_cnt*)
        vadj, vedge = self.flatten_embedding(vo, batch_range, vedge)
        # vmask.shape: (batch*v_cnt, p_cnt*)
        vmask = vadj == 0
        # vadj.shape: (p_cnt*, batch*v_cnt)
        vadj = vadj.transpose(0, 1)
        # vedge.shape: (p_cnt*, batch*v_cnt, e_dim)
        vedge = vedge.transpose(0, 1)

        padj, pedge = self.flatten_embedding(po, batch_range, pedge)
        pmask = padj == 0
        padj = padj.transpose(0, 1)
        pedge = pedge.transpose(0, 1)

        p2p_adj = p2p.repeat(batch, 1, 1)
        # p2p_adj.shape: (batch*p_cnt, p_cnt*)
        p2p_adj, ppedge = self.flatten_embedding(p2p_adj, batch_range, ppedge)
        # p2p_mask.shape: (batch*p_cnt, p_cnt*)
        p2p_mask = p2p_adj == 0
        # p2p_adj.shape: (p_cnt*, batch*p_cnt)
        p2p_adj = p2p_adj.transpose(0, 1)
        ppedge = ppedge.transpose(0, 1)

        return {
            "v": v,
            "p": p,
            "pe": {"edge": pedge, "adj": padj, "mask": pmask},
            "ve": {"edge": vedge, "adj": vadj, "mask": vmask},
            "ppe": {"edge": ppedge, "adj": p2p_adj, "mask": p2p_mask},
            "mask": seq_mask,
        }

    def _preprocess(self, states, actions, returns, next_states):
        states = self.union(
            states["p"], states["po"], states["pedge"], states["v"], states["vo"], states["vedge"],
            states["ppedge"], states["mask"]
        )
        actions = torch.from_numpy(actions).long().to(self._device)
        returns = torch.from_numpy(returns).float().to(self._device)
        next_states = self.union(
            next_states["p"], next_states["po"], next_states["pedge"],
            next_states["v"], next_states["vo"], next_states["vedge"],
            next_states["ppedge"], next_states["mask"]
        )
        return states, actions, returns, next_states

    @staticmethod
    def flatten_embedding(embedding, batch_range, edge=None):
        if len(embedding.shape) == 3:
            batch, x_cnt, y_cnt = embedding.shape
            addon = (batch_range * y_cnt).view(batch, 1, 1)
        else:
            seq_len, batch, x_cnt, y_cnt = embedding.shape
            addon = (batch_range * y_cnt).view(seq_len, batch, 1, 1)

        embedding_mask = embedding == 0
        embedding += addon
        embedding[embedding_mask] = 0
        ret = embedding.reshape(-1, embedding.shape[-1])
        col_mask = ret.sum(dim=0) != 0
        ret = ret[:, col_mask]
        if edge is None:
            return ret
        else:
            edge = edge.reshape(-1, *edge.shape[2:])[:, col_mask, :]
            return ret, edge
