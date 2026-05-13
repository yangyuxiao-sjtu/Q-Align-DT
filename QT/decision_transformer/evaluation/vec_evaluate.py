import gym
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import torch
from d4rl import get_normalized_score as get_d4rl_normalized_score
import torch.multiprocessing as mp
import pickle
def get_env_builder(seed, env_name, dataset,dversion,target_goal=None):

        def make_env_fn():
            import d4rl

            if env_name == 'hopper':
                env = gym.make('Hopper-v3')
            elif env_name == 'halfcheetah':
                env = gym.make('HalfCheetah-v3')
            elif env_name == 'walker2d':
                env = gym.make('Walker2d-v3')
            else:
                gym_name = f'{env_name}-{dataset}-v{dversion}'
                env = gym.make(gym_name)

                
            env.seed(seed)
         
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            if target_goal is not None:
                env.set_target_goal(target_goal)
                print(f"Set the target goal to be {env.target_goal}")
            return env

        return make_env_fn
def evaluate_episode_rtg_vec(
            vec_env,
            state_dim,
            act_dim,
            model,
            critic,
            max_ep_len,
            scale,
            target_return,
            mode,
            state_mean=0.,
            state_std=1.,
            device="cuda"
            ):
    model.eval()
    model.to(device=device)

    num_envs = vec_env.num_envs
    state_mean = torch.as_tensor(state_mean, device=device)
    state_std = torch.as_tensor(state_std, device=device)

    # reset environments
    states = vec_env.reset()  # (num_envs, state_dim)
    states = torch.as_tensor(states, device=device, dtype=torch.float32)

    # initialize history buffers: (num_envs, T, dim)
    state_hist = states.unsqueeze(1)  # (num_envs, 1, state_dim)
    actions = torch.zeros((num_envs, 0, act_dim), device=device)
    rewards = torch.zeros((num_envs, 0), device=device)

    # returns / timestep
    ep_return = target_return if target_return is not None else np.zeros(num_envs)
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).unsqueeze(1)
    timesteps = torch.zeros((num_envs, 1), device=device, dtype=torch.long)
    
    env_name = vec_env.get_attr("spec")[0].id
    is_antmaze = 'antmaze' in env_name
    is_maze2d = 'maze2d' in env_name

    if is_antmaze:
        min_distances = torch.full((num_envs,), float("inf"), device=device)
        goals = np.stack(vec_env.get_attr("target_goal"), axis=0)
        goals = torch.as_tensor(goals, device=device, dtype=torch.float32)
    elif is_maze2d:
        min_distances = torch.full((num_envs,), float("inf"), device=device)
        goals = np.stack(vec_env.env_method("get_target"), axis=0)
        goals = torch.as_tensor(goals, device=device, dtype=torch.float32)

    # episode stats
    episode_returns = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    dones = torch.zeros(num_envs, device=device, dtype=torch.bool)

    for t in range(max_ep_len):

        # add zero-padding placeholders along time dim
        actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

        # normalize state history
        norm_states = (state_hist - state_mean) / state_std

        # --- model predicts next action given the full trajectory ---
        action = model.get_action(
            states=norm_states,                 # (num_envs, T, state_dim)
            actions=actions,                    # (num_envs, T, act_dim)
            returns_to_go=target_return,        # (num_envs, T)
            timesteps=timesteps,
            critic=critic,
            rewards=None,
            batch_sz=num_envs,
        )  # (num_envs, act_dim)

        actions[:, -1] = action.detach()

        # --- env step ---
        next_states, reward, done, _ = vec_env.step(action.detach().cpu().numpy())
        next_states = torch.as_tensor(next_states, device=device, dtype=torch.float32)
        rewards[:, -1] = torch.as_tensor(reward, device=device, dtype=torch.float32)

        # --- update returns ---
        pred_return = target_return[:, -1] - (torch.as_tensor(reward, device=device) / scale)
        target_return = torch.cat([target_return, pred_return.unsqueeze(1)], dim=1)

        # --- update timestep ---
        timesteps = torch.cat([timesteps, (timesteps[:, -1] + 1).unsqueeze(1)], dim=1)

        # --- update states history ---
        state_hist = torch.cat([state_hist, next_states.unsqueeze(1)], dim=1)

        # --- accumulate stats ---
        episode_returns[~dones] += torch.as_tensor(reward, device=device)[~dones]
        episode_lengths[~dones] += 1
        
        if is_antmaze or is_maze2d:
            pos = next_states[:,:2]
            dist = torch.norm(pos - goals, dim=1)
            min_distances[~dones] = torch.minimum(dist, min_distances)[~dones]
        dones = dones | torch.as_tensor(done, device=device)

        if dones.all():
            break
    if is_antmaze or is_maze2d:
        dic = {}
        dic['min_distance'] = min_distances.cpu().numpy()
        return episode_returns.cpu().numpy(), episode_lengths.cpu().numpy(), dic
    
    return episode_returns.cpu().numpy(), episode_lengths.cpu().numpy(),None