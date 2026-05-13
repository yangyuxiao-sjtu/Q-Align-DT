# Code are largely from https://github.com/Dragon-Zhuang/BPPO.git
import d4rl

import gym
import numpy as np
import torch

import argparse
import pickle
import random
import sys
import os
import pathlib
import time
import torch.nn.functional as F

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.training.ql_trainer import Trainer
from decision_transformer.models.ql_DT import DecisionTransformer, Critic
from decision_transformer.models.decision_transformer_conv import DecisionTransformer_conv
import wandb
from d4rl import infos
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
USE_BPPO=False

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.Mish()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Mish())
    layers.append(nn.Linear(hidden_dim, output_dim))
    # if final_activation == 'relu':
    #     layers.append(nn.ReLU())
    # elif final_activation == 'tanh':
    #     layers.append(nn.Tanh())

    return nn.Sequential(*layers)



class QMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, 
        state_dim: int, action_dim: int, hidden_dim: int, depth:int
    ) -> None:
        super().__init__()
        self._net = MLP((state_dim + action_dim), hidden_dim, depth, 1, 'relu')

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) :
        sa = torch.cat([s, a], dim=1)
        x=self._net(sa)
        return x,x
CONST_EPS = 1e-10
class OnlineReplayBuffer:
    _device: torch.device
    _state: np.ndarray
    _action: np.ndarray
    _reward: np.ndarray
    _next_state: np.ndarray
    _next_action: np.ndarray
    _not_done: np.ndarray
    _return: np.ndarray
    _size: int


    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:

        self._device = device

        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._not_done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))

        self._size = 0


    def store(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_p: np.ndarray,
        a_p: np.ndarray,
        not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1


    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]


    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)


    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )



class OfflineReplayBuffer(OnlineReplayBuffer):

    def __init__(
        self, device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:
        super().__init__(device, state_dim, action_dim, max_size)


    def load_dataset(
        self, dataset: dict
    ) -> None:
        self._state = dataset['observations'][:-1, :]
        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        self._not_done = 1. - (dataset['terminals'].reshape(-1, 1)[:-1, :] | dataset['timeouts'].reshape(-1, 1)[:-1, :])

        self._size = len(dataset['actions']) - 1


    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
class QLearner:
    _device: torch.device
    _Q: Critic
    _optimizer: torch.optim
    _target_Q: Critic
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float
    _batch_size: int

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        layernorm: bool,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        if USE_BPPO:
            self._Q=QMLP(state_dim,action_dim,256,depth).to(device)
        else:
            self._Q = Critic(
                state_dim,
                action_dim,
                hidden_dim=hidden_dim,
                n_hiddens=depth,
                activation="mish",
                layernorm=layernorm,
            ).to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )
        if USE_BPPO:
            self._target_Q=QMLP(state_dim,action_dim,256,depth).to(device)
        else:
            self._target_Q = Critic(
                state_dim,
                action_dim,
                hidden_dim=hidden_dim,
                n_hiddens=depth,
                activation="mish",
                layernorm=layernorm,
            ).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        raise NotImplementedError


    def update(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> float:
        Q_loss = self.loss(replay_buffer, pi)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._Q.state_dict(), path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')



class QSarsaLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        layernorm: bool,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int,
        env:str,
        sparse_reward_fix:bool,
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        layernorm = layernorm,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size
        )
        self.env=env
        self.sparse_reward_fix=sparse_reward_fix
    def loss(self, replay_buffer: OnlineReplayBuffer, pi) -> torch.Tensor:
        s, a, r, s_p, a_p, not_done, _, _ = replay_buffer.sample(self._batch_size)

        with torch.no_grad():
            target_q1, target_q2 = self._target_Q(s_p, a_p)


            target_q = torch.min(target_q1, target_q2)
            if 'maze' in self.env and not self.sparse_reward_fix:# In sparse reward seting (Maze) Q should not larger than 1
                target_q = torch.clamp(target_q, 0., 1.0)
            target_Q_value = r + not_done * self._gamma * target_q

        current_q1, current_q2 = self._Q(s, a)
        loss = F.mse_loss(current_q1, target_Q_value) + F.mse_loss(current_q2, target_Q_value)
        return loss
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(2e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(2e6), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e6), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=256, type=int)
    parser.add_argument("--q_depth", default=3, type=int)       
    parser.add_argument("--q_layernorm", action="store_true", default=False)
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) # try to reduce the bc/q/v step if it works poorly, 5e-4/2e-5/2e-5 for bc/q/v, for example
    parser.add_argument("--bc_hidden_dim", default=1024, type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(1e3), type=int)
    parser.add_argument("--bppo_hidden_dim", default=1024, type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0.0, type=float) # for ()-medium-() tasks, try to use the entropy loss, weight == 0.01
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=True, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=True, type=bool)
    parser.add_argument("--sparse_reward_fix",action="store_true",default = False)
    
    args = parser.parse_args()
    env = gym.make(args.env)
    # seed
    
    # dim of state and action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    # offline dataset to replay buffer
    dataset = env.get_dataset()
    if args.sparse_reward_fix :
        print("Using sparse_reward_fix!")
        dataset['rewards']=dataset['rewards'] -1.
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']))
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(args.gamma)
    
    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = 0., 1.

    comment = args.env + '_' + str(args.seed)

    wandb_name =f"{args.env}"
    if USE_BPPO:
        wandb_name+="-BPPO_Q"
    wandb.init(
        project="QT_pretrain_q",
        name=wandb_name,
        config=vars(args)
    )

    path = os.path.join(args.path, args.env)


    Q_bc = QSarsaLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_layernorm, args.q_lr, args.target_update_freq,
                         args.tau, args.gamma, args.q_batch_size,args.env,args.sparse_reward_fix)
    path =f"./saved_q_{args.env}/"
    if args.sparse_reward_fix :
        path =f"./saved_q_{args.env}_sparse_reward_fix/"
    
    if args.q_layernorm:
        model_name = "Q_bc_layernorm.pt"
    else:
        model_name = "Q_bc.pt"
    Q_bc_path = os.path.join(path, model_name)


    os.makedirs(path,exist_ok=True)

    for step in tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'): 
        Q_bc_loss = Q_bc.update(replay_buffer, pi=None)

        if step % 100 == 0:
            print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
            wandb.log({f"{args.env}/Q_bc_loss": Q_bc_loss})
    if USE_BPPO==False:
        Q_bc.save(Q_bc_path)
    
