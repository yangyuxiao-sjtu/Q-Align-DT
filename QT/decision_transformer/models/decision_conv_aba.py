import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions import Normal, Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import transformers
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
)
import torch.nn.functional as F

from transformers.utils import logging
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import math
# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""


logger = logging.get_logger(__name__)


class Attention(nn.Module):
    def __init__(self, config, hidden_size, block_index, ctx=1024):
        super().__init__()
      
        self.remove_act_embs = False
        self.hidden_size = hidden_size
        self.idx = block_index
        
        # Convolution parameters.
        self.conv_kernel_size = 6 
        self.conv_padding = self.conv_kernel_size - 1 
        
      
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.q_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.conv_kernel_size, padding=self.conv_padding)
        self.k_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.conv_kernel_size, padding=self.conv_padding)
        self.v_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.conv_kernel_size, padding=self.conv_padding)

        self.n_head = config.n_head
        
        mask = torch.tril(torch.ones((ctx, ctx),dtype = torch.uint8), diagonal=0)
        mask = mask.clamp(min=0)
        self.register_buffer("bias", mask.view(1, 1,ctx, ctx))
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        
        extra_mask = torch.tril(torch.ones(ctx, ctx), diagonal=-3)
        self.register_buffer("extra_bias", extra_mask.view(1,1, ctx, ctx))
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        

    def _causal_conv(self, linear_output, conv_layer, T_original):
        # 1. Transpose: [B, T, D] -> [B, D, T]
        x_conv_in = linear_output.transpose(-1, -2) 
        
        # 2. Apply Conv: [B, D, T] -> [B, D, T + padding]
        conv_out = conv_layer(x_conv_in)
        
        # 3. Causal truncation keeps the original sequence length T.
        return conv_out[:, :, :T_original] # [B, D, T]


    def atten(self, qkv, x,add_extra):
        # qkv: [B, T, 3*D]
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        B, T, D = v.shape
        head_dim = D//self.n_head
        assert D == self.n_head * head_dim, f"hidden size {D} can not be dived by head_dim {head_dim} and n_head {self.n_head}!!!"

        v=v.view(B,T,self.n_head,head_dim).transpose(1,2)
        k=k.view(B,T,self.n_head,head_dim).transpose(1,2)
        q=q.view(B,T,self.n_head,head_dim).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask.
        mask = self.bias[:,:, :T, :T]
        attn_scores = torch.where(mask.bool(), attn_scores, self.masked_bias.to(attn_scores.dtype))

        w = torch.softmax(attn_scores, dim=-1)
        w = self.attn_dropout(w)
        
        out =torch.matmul(w, v)
        out=out.transpose(1,2).reshape(B,T,D)
        return out # [B, T, D]
    
    def forward(self, x,add_extra=True):
        
     
       
        q_lin = self.q_proj(x)
        k_lin = self.k_proj(x)
        v_lin = self.v_proj(x)

        T_original = x.shape[1] 
        
        # # 2. Apply causal Conv1D: [B, T, D] -> [B, D, T] -> [B, T, D]
        # q_conv = self._causal_conv(q_lin, self.q_conv, T_original)
        # k_conv = self._causal_conv(k_lin, self.k_conv, T_original)
        # v_conv = self._causal_conv(v_lin, self.v_conv, T_original)

        # # 3. Transpose back: [B, D, T] -> [B, T, D]
        # q_final = q_conv.transpose(-1, -2)
        # k_final = k_conv.transpose(-1, -2)
        # v_final = v_conv.transpose(-1, -2)

        # # 4. Concatenate Q, K, V for self.atten: [B, T, 3*D]
        # qkv_final = torch.cat([q_final, k_final, v_final], dim=-1)
        qkv_final = torch.cat([q_lin, k_lin, v_lin], dim=-1)
        out = self.atten(qkv_final, x, add_extra=add_extra)
    


        a = self.c_proj(out)
        a = self.resid_dropout(a)
        return a



class Block(nn.Module):
    def __init__(self, config, index, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.atten = Attention(config, hidden_size, index)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size, hidden_size),
            nn.Dropout(0.1),
        )
            
        self.index = index

    def forward(
            self,
            hidden_states,
            add_extra=True
    ):
        conv_output = self.atten(self.ln_1(hidden_states),add_extra=add_extra)
        hidden_states = conv_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        
        hidden_states = hidden_states + feed_forward_hidden_states
        
        return hidden_states


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index, scale=True) for index in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None,add_extra=True):
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]

        device = inputs_embeds.device

        hidden_states = inputs_embeds 
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states,add_extra=add_extra)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states
    


class DecisionTransformer_convaba(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            sar=False,
            scale=1.,
            rtg_no_q=False,
            infer_no_q=False,
            alg =None,
            **kwargs
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.rtg_no_q=rtg_no_q
        self.infer_no_q=infer_no_q
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            remove_act_embs=False,
            **kwargs
        )
        self.alg=alg
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.scale=scale
        self.remove_act_embs = False

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        
    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
            
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
        else:
            num_token_type = 3
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(inputs_embeds=stacked_inputs,add_extra=False)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:, 1])
        state_preds = self.predict_state(x[:, 2])
        rewards_preds = None


        return state_preds, action_preds, rewards_preds

    # def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
    #     # we don't care about the past rewards in this model

    #     states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=10, dim=0)
    #     actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=10, dim=0)
    #     rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=10, dim=0)
    #     timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=10, dim=0)

    #     returns_to_go = returns_to_go[0]
    #     returns_to_go = returns_to_go.reshape(1, -1, 1).repeat_interleave(repeats=10, dim=0)
    #     #returns_to_go = torch.cat([returns_to_go, torch.randn((50-returns_to_go.shape[0], returns_to_go.shape[1], 1), device=returns_to_go.device)], dim=0)
            

    #     if self.max_length is not None:
    #         states = states[:,-self.max_length:]
    #         actions = actions[:,-self.max_length:]
    #         returns_to_go = returns_to_go[:,-self.max_length:]
    #         rewards = rewards[:,-self.max_length:]
    #         timesteps = timesteps[:,-self.max_length:]

    #         # padding
    #         attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
    #         attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1).repeat_interleave(repeats=10, dim=0)
    #         states = torch.cat(
    #             [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
    #             dim=1).to(dtype=torch.float32)
    #         returns_to_go = torch.cat(
    #             [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
    #             dim=1).to(dtype=torch.float32)
    #         timesteps = torch.cat(
    #             [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
    #             dim=1
    #         ).to(dtype=torch.long)
    #         rewards = torch.cat(
    #             [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
    #             dim=1
    #         ).to(dtype=torch.float32)

    #         actions = torch.cat(
    #             [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
    #             dim=1).to(dtype=torch.float32)
    #     else:
    #         attention_mask = None
       
    #     if "sequence" in self.alg:
    #         noise = torch.randn(returns_to_go[1:].shape[0], 1,1, device=returns_to_go.device) * 0.1
    #         returns_to_go[1:, :] = returns_to_go[1:, :] + torch.abs(noise)
    #     else:
    #         returns_to_go[1:, -1] = returns_to_go[1:, -1] + torch.abs(torch.randn_like(returns_to_go[1:, -1]) * 0.1)
    #     if not self.rtg_no_q:
    #         returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale
    #     _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
    
        
    #     state_rpt = states[:, -1, :]
    #     action_preds = action_preds[:, -1, :]

    #     q_value = critic.q_min(state_rpt, action_preds).flatten()
    #     idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)

    #     if not self.infer_no_q:
    #         return action_preds[idx]
    #     else:
    #         return action_preds[0]
    def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, batch_sz = None,**kwargs):
        # we don't care about the past rewards in this model
        if batch_sz is None:
            batch_sz=states.shape[0]
        states = states.reshape(batch_sz, -1, self.state_dim)
        actions = actions.reshape(batch_sz, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(batch_sz, -1, 1)
        timesteps = timesteps.reshape(batch_sz, -1)

        states = states[:,-self.max_length:]
        actions = actions[:,-self.max_length:]
        returns_to_go = returns_to_go[:,-self.max_length:]
        timesteps = timesteps[:,-self.max_length:]

        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)

        _,action_preds,__ = self.forward(states=states, actions=actions, returns_to_go=returns_to_go, timesteps=timesteps)
        if batch_sz==1:
            return action_preds[0,-1]
        return action_preds[:,-1]
