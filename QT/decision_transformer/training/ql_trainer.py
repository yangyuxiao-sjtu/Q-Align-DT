import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:

    def __init__(self, 
                model, 
                critic,
                batch_size, 
                tau,
                discount,
                get_batch, 
                loss_fn, 
                eval_fns=None,
                max_q_backup=False,
                eta=1.0,
                eta2=1.0,
                ema_decay=0.995,
                step_start_ema=1000,
                update_ema_every=5,
                lr=3e-4,
                weight_decay=1e-4,
                lr_decay=False,
                lr_maxt=100000,
                lr_min=0.,
                grad_norm=1.0,
                scale=1.0,
                k_rewards=False,
                use_discount=True,
                **kwargs
            ):
        
        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=kwargs['critic_lr'], weight_decay=0.0)
        # self.vf=kwargs.get("vf",None)
        # if self.vf:
        #     self.vf_optimizer= torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)
        #     if self.vf:
        #         self.vf_lr_scheduler = CosineAnnealingLR(self.vf_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.eta = eta
        self.eta2 = eta2
        self.final_eta2=eta2
        self.lr_decay = lr_decay
        self.scale = scale
        self.k_rewards = k_rewards
        self.use_discount = use_discount
        self.noise_q=kwargs.get('noise_q',None)
        self.start_time = time.time()
        self.step = 0
        self.rtg_scale = kwargs['rtg_scale']
        self.rtg_init_scale= kwargs['rtg_scale']
        self.rtg_final_scale =1.
        self.max_iter=lr_maxt
        self.alg =kwargs['alg']
        self.rs_loss =kwargs['rs_loss']
        self.weight_method =kwargs['weight_method']
        self.bc_iter=kwargs.get('bc_iter',0)
        self.iter_num=0
        self.Q_update_BC = kwargs['Q_update_BC']
        self.alignment_func=kwargs['alignment_function']
        self.clip_range =kwargs.get('clip_range',None)
        self.env_name=kwargs['env_name']
        self.critic_update_every=kwargs['critic_update_every']
        self.norm_q=kwargs['norm_q']
        self.target_rtg=kwargs['target_rtg']
        self.sparse_reward_fix=kwargs['sparse_reward_fix']
        self.update_Q=kwargs['update_Q']
        self.state_mean=kwargs['state_mean']
        self.state_std=kwargs['state_std']
        self.margin_beta = kwargs.get('margin',0.)
        self.min_noise =kwargs.get("min_noise",0.1)
        self.stop_Q= kwargs.get("stop_Q",None)
        self.start_update_Q= kwargs.get("start_update_Q",0)
        self.critic_update= kwargs.get("critic_update",1)
        self.use_target_q_stop = kwargs.get("use_target_q_stop", True)
        self.target_q_stop_start_epoch = max(1, int(kwargs.get("target_q_stop_start_epoch", 30)))
        self.target_q_stop_window = max(1, int(kwargs.get("target_q_stop_window", 3)))
        self.target_q_stop_relative_drop = float(kwargs.get("target_q_stop_relative_drop", 0.005))
        self.target_q_stop_patience = max(1, int(kwargs.get("target_q_stop_patience", 1)))
        self.target_q_history = []
        self.target_q_stop_best_window_mean = None
        self.target_q_stop_bad_count = 0
        self.target_q_stop_last_metrics = {}
    def step_ema(self):
        if self.Q_update_BC:
            return 
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)
    def step_rtg_scale(self,step):
        self.rtg_scale= self.rtg_final_scale + 0.5 * (self.rtg_init_scale - self.rtg_final_scale) * (1 + np.cos(np.pi * step /  self.max_iter))
        # if self.final_eta2>10:
        #     self.eta2=  self.final_eta2 * (1 - np.exp(-step /  50))

    def _update_target_q_stop(self, iter_num, signal_mean):
        metrics = {
            'enabled': 1.0 if self.use_target_q_stop else 0.0,
            'start_epoch': float(self.target_q_stop_start_epoch),
            'window': float(self.target_q_stop_window),
            'relative_drop': float(self.target_q_stop_relative_drop),
            'patience': float(self.target_q_stop_patience),
            'bad_count': float(self.target_q_stop_bad_count),
            'current_mean': float(signal_mean),
            'history_size': float(len(self.target_q_history) + 1),
            'should_stop': 0.0,
        }

        self.target_q_history.append(float(signal_mean))

        if len(self.target_q_history) >= self.target_q_stop_window:
            recent_window_mean = float(np.mean(self.target_q_history[-self.target_q_stop_window:]))
        else:
            recent_window_mean = float(np.mean(self.target_q_history))
        metrics['recent_window_mean'] = recent_window_mean

        if self.target_q_stop_best_window_mean is None:
            self.target_q_stop_best_window_mean = recent_window_mean
        best_window_mean = float(self.target_q_stop_best_window_mean)
        metrics['best_window_mean'] = best_window_mean

        if (not self.use_target_q_stop) or iter_num < self.target_q_stop_start_epoch:
            self.target_q_stop_bad_count = 0
            metrics['bad_count'] = 0.0
            self.target_q_stop_last_metrics = metrics
            return False

        if len(self.target_q_history) < 2 * self.target_q_stop_window:
            self.target_q_stop_bad_count = 0
            metrics['bad_count'] = 0.0
            self.target_q_stop_last_metrics = metrics
            return False

        window = self.target_q_stop_window
        window_means = [
            float(np.mean(self.target_q_history[idx - window:idx]))
            for idx in range(window, len(self.target_q_history) + 1)
        ]
        window_slopes = [
            window_means[idx] - window_means[idx - 1]
            for idx in range(1, len(window_means))
        ]

        if not window_slopes:
            self.target_q_stop_bad_count = 0
            metrics['bad_count'] = 0.0
            self.target_q_stop_last_metrics = metrics
            return False

        recent_slope = float(np.mean(window_slopes[-window:]))
        reference_span = min(max(window, 1), len(window_slopes))
        reference_slopes = window_slopes[:reference_span]
        positive_reference_slopes = [slope for slope in reference_slopes if slope > 0.0]
        if positive_reference_slopes:
            reference_slope = float(np.mean(positive_reference_slopes))
        else:
            reference_slope = float(np.mean(reference_slopes))

        slope_threshold = 0.1 * max(reference_slope, 0.0)
        required_floor = (1.0 - self.target_q_stop_relative_drop) * best_window_mean

        metrics['recent_slope'] = recent_slope
        metrics['reference_slope'] = reference_slope
        metrics['slope_threshold'] = float(slope_threshold)
        metrics['required_floor'] = float(required_floor)

        has_reached_plateau_level = recent_window_mean >= required_floor
        is_growth_plateau = recent_slope <= slope_threshold
        metrics['has_reached_plateau_level'] = 1.0 if has_reached_plateau_level else 0.0
        metrics['is_growth_plateau'] = 1.0 if is_growth_plateau else 0.0

        if has_reached_plateau_level and is_growth_plateau:
            self.target_q_stop_bad_count += 1
        else:
            self.target_q_stop_bad_count = 0

        if recent_window_mean > self.target_q_stop_best_window_mean:
            self.target_q_stop_best_window_mean = recent_window_mean
        metrics['best_window_mean'] = float(self.target_q_stop_best_window_mean)
        metrics['bad_count'] = float(self.target_q_stop_bad_count)

        should_stop = self.target_q_stop_bad_count >= self.target_q_stop_patience
        metrics['should_stop'] = 1.0 if should_stop else 0.0
        self.target_q_stop_last_metrics = metrics
        return should_stop

    def _compute_guidance_metrics(
        self,
        sign,
        weight,
        q1_new_action,
        q2_new_action,
        q1_new_action_noise,
        q2_new_action_noise,
    ):
        if sign is None or weight is None:
            return {
                'guidance_margin_mean': 0.0,
                'guidance_agree_ratio': 0.0,
            }

        with torch.no_grad():
            gap1 = sign * weight * (q1_new_action_noise - q1_new_action)
            gap2 = sign * weight * (q2_new_action_noise - q2_new_action)
            paired_gap = torch.minimum(gap1, gap2)

            return {
                'guidance_margin_mean': F.relu(paired_gap).mean().item(),
                'guidance_agree_ratio': (paired_gap > 0).float().mean().item(),
            }

    def train_iteration(self, num_steps, logger, iter_num=0, log_writer=None):

        logs = dict()

        train_start = time.time()
        if self.stop_Q is not None and iter_num>=self.stop_Q:
            self.update_Q=False
        self.actor.train()
        self.critic.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
            'current_target_q_mean': [],
            'guidance_margin_mean': [],
            'guidance_agree_ratio': [],
        }
        self.iter_num=iter_num
        for _ in trange(num_steps):
            loss_metric = self.train_step(log_writer, loss_metric)
      
        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            # if self.vf:
            #     self.vf_lr_scheduler.step()
        self.step_rtg_scale(iter_num)
      
        bc_loss = np.mean(loss_metric['bc_loss'])
        ql_loss = np.mean(loss_metric['ql_loss'])
        actor_loss = np.mean(loss_metric['actor_loss'])
        critic_loss = np.mean(loss_metric['critic_loss'])
        target_q_mean = np.mean(loss_metric['target_q_mean'])
        current_target_q_mean = np.mean(loss_metric['current_target_q_mean'])
        guidance_margin_mean = np.mean(loss_metric['guidance_margin_mean'])
        guidance_agree_ratio = np.mean(loss_metric['guidance_agree_ratio'])
        logger.record_tabular('BC Loss', bc_loss)
        logger.record_tabular('QL Loss', ql_loss)
        logger.record_tabular('Actor Loss', actor_loss)
        logger.record_tabular('Critic Loss', critic_loss)
        logger.record_tabular('Target Q Mean', target_q_mean)
        logger.record_tabular('Current Target Q Mean', current_target_q_mean)
        logger.record_tabular('Guidance Margin Mean', guidance_margin_mean)
        logger.record_tabular('Guidance Agree Ratio', guidance_agree_ratio)
        logger.dump_tabular()

        logs['time/training'] = time.time() - train_start
        logs['training/ql_loss']=float(ql_loss)
        logs['training/actor_loss']=float(actor_loss)
        logs['training/critic_loss']=float(critic_loss)
        logs['training/target_q_mean']=float(target_q_mean)
        logs['training/current_target_q_mean']=float(current_target_q_mean)
        logs['training/guidance_margin_mean']=float(guidance_margin_mean)
        logs['training/guidance_agree_ratio']=float(guidance_agree_ratio)
        if self.update_Q and self._update_target_q_stop(iter_num=iter_num, signal_mean=current_target_q_mean):
            self.update_Q = False
        if self.target_q_stop_last_metrics:
            for key, value in self.target_q_stop_last_metrics.items():
                logs[f'training/target_q_stop_{key}'] = float(value)
        eval_start = time.time()


        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            logger.record_tabular(k, float(v))
        logger.record_tabular('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        logger.record_tabular('Current critic learning rate', self.critic_optimizer.param_groups[0]['lr'])
        logger.dump_tabular()

        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        
        logs['rtg_scale']=self.rtg_scale
        return logs
    
    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_
    def clip_rtg(self, rtg, min_val=None, max_val=None):
        if self.clip_range is None and min_val is None and max_val is None:
            return rtg

        default_min, default_max = (self.clip_range or (None, None))
        min_val = default_min if min_val is None else min_val
        max_val = default_max if max_val is None else max_val
        assert not (min_val is None) ^ (max_val is None), \
            f"Both min_val and max_val must be provided (got min_val={min_val}, max_val={max_val})"
        rtg0 = rtg[:, 0]
        rtg0_clipped = torch.clamp(rtg0, min=min_val, max=max_val)
        delta = rtg0_clipped - rtg0
        # import sys
        # new_rtg=rtg + delta.unsqueeze(1)
        # print("new_rtg,",new_rtg[:3][:20])
        # print("old_rtg",rtg[:3][:20])
        
        # sys.exit()
        return rtg + delta.unsqueeze(1)
   
    def train_step(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''

        T = states.shape[1]
        if self.update_Q and self.iter_num>=self.start_update_Q and self.step % self.critic_update==0:
            current_q1, current_q2 = self.critic.forward(states, actions)

            if self.Q_update_BC:
                next_action =action_target
            else:
                if  "alignment" in  self.alg:
                    new_rtg=rtg+self.target_rtg
                    #new_rtg=rtg+5.

                    new_rtg= self.clip_rtg(new_rtg)
                    _, next_action, _ = self.ema_model(
                            states, actions, rewards, action_target,new_rtg[:,:-1] , timesteps, attention_mask=attention_mask,
                        )
                else:
                    _, next_action, _ = self.ema_model(
                        states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
                    )  
            if  self.sparse_reward_fix:
                q_rewards=rewards-1.
            else:
                q_rewards=rewards
            
            if self.k_rewards:     
            
                critic_next_states = states[:, -1]
                next_action = next_action[:, -1]
                with torch.no_grad():
                    target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
                    current_target_q1, current_target_q2 = self.critic(critic_next_states, next_action)
                target_q = torch.min(target_q1, target_q2) # [B, 1]
                current_target_q = torch.min(current_target_q1, current_target_q2)
                if 'antmaze' in self.env_name and not self.sparse_reward_fix:
                    target_q = torch.clamp(target_q, 0., 1.0)
                    current_target_q = torch.clamp(current_target_q, 0., 1.0)
                not_done =(1 - dones[:, -1]) # [B, 1]
                if self.use_discount:
                
              
                    target_q_last=target_q
                    rewards_no_last = q_rewards.clone()
                    rewards_no_last[:, -1] = target_q_last

                    discount_factors = self.discount ** torch.arange(T, device=device).float().view(1, T, 1)  # [1, T, 1]
                    not_done_mask = (1.0 - dones).float()  # [B, T, 1]
                    discounted = rewards_no_last * discount_factors * not_done_mask
                    k_rewards = torch.cumsum(discounted.flip([1]), dim=1).flip([1]) #left padding
                    k_rewards = k_rewards / (discount_factors + 1e-8)

                    td_target = k_rewards.clone()
                    last_mask = ((attention_mask[:, -1] > 0) & (dones[:, -1, 0] == 1)).view(batch_size,1)
                    # if 'antmaze' in self.env_name:
                    #     last_mask = torch.zeros_like(last_mask, dtype=torch.bool)
                    td_target[:, -1] =q_rewards[:, -1]*last_mask
                    critic_mask = attention_mask.bool()  # [B, T]
                    critic_mask[:, -1] = critic_mask[:, -1] & last_mask.squeeze(-1)

                    
                else:
                    assert NotImplementedError # this will conflict with sparse_reward_fix, so plz go discount branch with factor =1 
                    
                    # k_rewards = (rtg[:,:-1] - rtg[:, -2:-1])* self.scale # [B, T, 1]
                    # target_q = (k_rewards + (not_done * target_q).unsqueeze(-1)).detach() # [B, T, 1]
            else:
                # if self.vf:
                #     value_pred = self.vf(states)
                #     with torch.no_grad():
                #         target_pred1,target_pred2 = self.critic_target(states,next_action)
                #         target_pred = torch.min(target_pred1,target_pred2)
                #     v_loss= F.mse_loss(value_pred[attention_mask>0],target_pred[attention_mask>0])
                #     self.vf_optimizer.zero_grad()
                #     v_loss.backward()
                #     self.vf_optimizer.step()
                #     target_q=value_pred[:,1:].detach()
                # else:
                with torch.no_grad():
                    next_actions = next_action[:, 1:, :]  # [B, T-1, A]
                    next_states = states[:, 1:, :]
                    target_q1, target_q2 = self.critic_target(next_states, next_actions)
                    current_target_q1, current_target_q2 = self.critic(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2)  # [B, T-1, 1]
                    current_target_q = torch.min(current_target_q1, current_target_q2)  # [B, T-1, 1]
                    if self.noise_q is not None:
                        noise = self.noise_q * torch.randn_like(target_q)
                        target_q = target_q + noise

                if 'antmaze' in self.env_name and not self.sparse_reward_fix:
                    target_q = torch.clamp(target_q, 0., 1.0)
                    current_target_q = torch.clamp(current_target_q, 0., 1.0)
                td_target = q_rewards[:, :-1] + self.discount * (1 - dones[:, :-1]) * target_q  # [B, T-1, 1]

                last_mask = (attention_mask[:, -1] > 0) & (dones[:, -1, 0] == 1)  # [B]
                last_td = q_rewards[:, -1:]  # [B, 1, 1]
                # if 'antmaze' in self.env_name:
                #     last_td = torch.zeros_like(last_td)
                #     last_mask = torch.zeros_like(last_mask, dtype=torch.bool)
                td_target = torch.cat([td_target, last_td], dim=1)  # [B, T, 1]

                # ======== Critic Loss ========
                critic_mask = attention_mask.bool()  # [B, T]
                critic_mask[:, -1] = critic_mask[:, -1] & last_mask
                # if 'maze' in self.env_name:
                #     critic_mask=critic_mask & (td_target.squeeze(-1)>0.5)
            
            critic_loss= F.mse_loss(current_q1[critic_mask], td_target[critic_mask]) + F.mse_loss(current_q2[critic_mask], td_target[critic_mask])
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()
        else:
            critic_grad_norms = torch.tensor([0.0], device=device)  
            critic_loss = torch.tensor(0.0, device=device)
            target_q = torch.tensor(0.0, device=device)
            current_target_q = torch.tensor(0.0, device=device)

        '''Policy Training'''        
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1]>0].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = F.mse_loss(reward_preds, reward_target)
        else:
            rewards_loss = 0
        if self.rs_loss ==False:
            states_loss=0.
            rewards_loss=0.
        bc_loss = F.mse_loss(action_preds_, action_target_) + states_loss + rewards_loss

        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        guidance_metrics = {
            'guidance_margin_mean': 0.0,
            'guidance_agree_ratio': 0.0,
        }
        # q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        if "alignment" in  self.alg:
            with torch.no_grad():
                q1_new_action, q2_new_action = self.critic(
                    actor_states,
                    action_preds_.detach()
                )
            
            if 'sequence' in self.alg:
                noise = self.rtg_scale * torch.randn(rtg.size(0), 1, 1, device=rtg.device)
                #min_abs = 1.
                min_abs = 0.2  # e.g. 0.1
                if 'antmaze' in self.env_name:
                    min_abs = 1. 
                    noise = noise.abs()
                    

                noise = torch.where(noise.abs() < min_abs, 
                                    min_abs * noise.sign(), 
                                    noise)
                rtg_noise = rtg + noise.expand_as(rtg)                
                rtg_noise=self.clip_rtg(rtg_noise)
  
                    # else: rtg_noise = 1.0 - rtg

                    #rtg_noise = torch.ones_like(rtg)

            elif 'step' in self.alg:
                rtg_noise = rtg +self.rtg_scale* torch.randn_like(rtg)
                rtg_noise=self.clip_rtg(rtg_noise)
            __, action_preds_noise, _ = self.actor.forward(
                states, actions, rewards, action_target, rtg_noise[:,:-1], timesteps, attention_mask=attention_mask,
            )
            action_preds_noise_ = action_preds_noise.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
            
            q1_new_action_noise, q2_new_action_noise = self.critic(actor_states, action_preds_noise_)
            delta_rtg = (rtg_noise[:, :-1] - rtg[:, :-1])
            if 'antmaze' in self.env_name :
                eps=1e-3
                success_orig = (rtg > eps)
                success_noisy = (rtg_noise > eps)
                mask_diff = success_orig ^ success_noisy 
                delta_rtg = torch.where(mask_diff, rtg_noise - rtg, torch.zeros_like(rtg_noise))[:, :-1]


            sign = torch.sign(delta_rtg).reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            if self.weight_method =="sigmoid":
                weight = torch.sigmoid(delta_rtg.abs()).reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            elif self.weight_method =="tanh":
                weight = torch.tanh(delta_rtg.abs()).reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            elif self.weight_method  is None:
                weight = torch.ones_like(sign)

            if self.alignment_func =="softplus":
                order_loss1 = F.softplus(-sign * weight*(q1_new_action_noise - q1_new_action)).mean()
                order_loss2 = F.softplus(-sign * weight*(q2_new_action_noise - q2_new_action)).mean()
            elif self.alignment_func == "linear":
                # Linear encourages correct ordering magnitude (no clamp)
                # We *minimize* L = - mean(weight * sign * qdiff)
                order_loss1 =  (-sign * weight*(q1_new_action_noise - q1_new_action)).mean()
                order_loss2 =  (-sign * weight*(q2_new_action_noise - q2_new_action)).mean()
            elif self.alignment_func =='relu':
                if self.margin_beta>0.:
                    margin=target_q.mean().item()*self.margin_beta
                    order_loss1 = F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)-margin).mean()
                    order_loss2 = F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)-margin).mean()
                else:
                    order_loss1 = F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)).mean()
                    order_loss2 = F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)).mean()
            elif self.alignment_func =='relu_min':
                if self.margin_beta>0.:
                    margin=target_q.mean().item()*self.margin_beta
                    order_loss1 = torch.minimum(
                        F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)-margin),
                        F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)-margin),
                    ).mean()
                else:
                    order_loss1 = torch.minimum(
                        F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)),
                        F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)),
                    ).mean()
                order_loss2 = order_loss1
            elif self.alignment_func=="relu_cross_min": 
                if self.margin_beta>0.:
                    margin=target_q.mean().item()*self.margin_beta
                    order_loss1 = torch.minimum(
                        torch.minimum(
                            F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)-margin),
                            F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)-margin),
                        ),
                        torch.minimum(
                            F.relu(-sign * weight*(q1_new_action_noise - q2_new_action)-margin),
                            F.relu(-sign * weight*(q2_new_action_noise - q1_new_action)-margin),
                        ),
                    ).mean()
                else:
                    order_loss1 = torch.minimum(
                        torch.minimum(
                            F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)),
                            F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)),
                        ),
                        torch.minimum(
                            F.relu(-sign * weight*(q1_new_action_noise - q2_new_action)),
                            F.relu(-sign * weight*(q2_new_action_noise - q1_new_action)),
                        ),
                    ).mean()
                order_loss2 = order_loss1
            elif self.alignment_func =='silu':
                order_loss1 = F.silu(-sign * weight*(q1_new_action_noise - q1_new_action)).mean()
                order_loss2 = F.silu(-sign * weight*(q2_new_action_noise - q2_new_action)).mean()
            elif self.alignment_func =="softplus-relu":
                order_loss1 = F.softplus(F.relu(-sign * weight*(q1_new_action_noise - q1_new_action))).mean()
                order_loss2 = F.softplus(F.relu(-sign * weight*(q2_new_action_noise - q2_new_action))).mean()
            elif self.alignment_func=='square':
                order_loss1 = F.relu(-sign * weight*(q1_new_action_noise - q1_new_action)).pow(2).mean()
                order_loss2 = F.relu(-sign * weight*(q2_new_action_noise - q2_new_action)).pow(2).mean()

            guidance_metrics = self._compute_guidance_metrics(
                sign=sign,
                weight=weight,
                q1_new_action=q1_new_action,
                q2_new_action=q2_new_action,
                q1_new_action_noise=q1_new_action_noise,
                q2_new_action_noise=q2_new_action_noise,
            )
            q_loss = 0.5 * (order_loss1 + order_loss2)
            q_loss_scale =q_loss
            if self.norm_q:
                cur_mean = q_loss.detach().abs().mean().item()
                if not hasattr(self, "q_loss_ma"):
                    self.q_loss_ma = cur_mean
                else:
                    self.q_loss_ma = 0.99 * self.q_loss_ma + 0.01 * cur_mean
                q_loss = q_loss / (self.q_loss_ma + 1e-6)
        else:
            q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
            q_loss_scale =q1_new_action.mean()
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

        if "alignment" in self.alg :
            if self.iter_num<=self.bc_iter:
                q_loss = torch.tensor(0.0, device=bc_loss.device) 
            # if "entropy" in self.alg:
            #     q_loss+=5*F.mse_loss(action_preds_noise_, action_target_) 
            actor_loss = self.eta * bc_loss + self.eta2 * q_loss
        elif self.alg =="no_q_loss":
            q_loss = torch.tensor(0.0, device=bc_loss.device)  
           
            actor_loss = bc_loss
        else:
            actor_loss = self.eta2 * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        if self.update_Q:
            self.step_ema()
            if self.step %self.critic_update_every ==0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step += 1
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            if "alignment" in self.alg :
                x=sign * weight*(q1_new_action_noise - q1_new_action)
                x_pos = x[x > 0]
                if x_pos.numel() > 0:
                    self.diagnostics['training/delta_q_mean']= x_pos.mean().detach().cpu().item()
        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)
            log_writer.add_scalar('Current_Target_Q Mean', current_target_q.mean().item(), self.step)
            log_writer.add_scalar('Guidance_Margin Mean', guidance_metrics['guidance_margin_mean'], self.step)
            log_writer.add_scalar('Guidance_Agree Ratio', guidance_metrics['guidance_agree_ratio'], self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss_scale.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())
        loss_metric['current_target_q_mean'].append(current_target_q.mean().item())
        loss_metric['guidance_margin_mean'].append(guidance_metrics['guidance_margin_mean'])
        loss_metric['guidance_agree_ratio'].append(guidance_metrics['guidance_agree_ratio'])

        return loss_metric
