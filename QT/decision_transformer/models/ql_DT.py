import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        n_hiddens=3,
        activation="mish",
        layernorm=False,
        state_mean=None,
        state_std=None,
        state_adapter="identity",
    ):
        super(Critic, self).__init__()
        self.q_mean =None
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_adapter = state_adapter
        if state_mean is None:
            state_mean = torch.zeros(self.state_dim, dtype=torch.float32)
        else:
            state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        if state_std is None:
            state_std = torch.ones(self.state_dim, dtype=torch.float32)
        else:
            state_std = torch.as_tensor(state_std, dtype=torch.float32)
        self.register_buffer("state_mean", state_mean.view(-1), persistent=False)
        self.register_buffer("state_std", state_std.view(-1), persistent=False)
        self.n_hiddens = int(n_hiddens)
        self.layernorm = bool(layernorm)
        self.activation = activation
        self._configure_network(
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            activation=self.activation,
            layernorm=self.layernorm,
        )

    def _activation_layer(self, activation: str) -> nn.Module:
        if activation == "mish":
            return nn.Mish()
        if activation == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported critic activation: {activation}")

    def _build_q_head(self, hidden_dim: int, n_hiddens: int, activation: str, layernorm: bool) -> nn.Sequential:
        input_dim = self.state_dim + self.action_dim
        layers = [nn.Linear(input_dim, hidden_dim), self._activation_layer(activation)]
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(max(0, n_hiddens - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._activation_layer(activation))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def _get_network_device_dtype(self):
        for module_name in ("q1_model", "q2_model"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            try:
                param = next(module.parameters())
                return param.device, param.dtype
            except StopIteration:
                continue
        return self.state_mean.device, self.state_mean.dtype

    def _configure_network(self, hidden_dim: int, n_hiddens: int, activation: str, layernorm: bool) -> None:
        device, dtype = self._get_network_device_dtype()
        self.hidden_dim = int(hidden_dim)
        self.n_hiddens = int(n_hiddens)
        self.activation = activation
        self.layernorm = bool(layernorm)
        self.q1_model = self._build_q_head(self.hidden_dim, self.n_hiddens, self.activation, self.layernorm)
        self.q2_model = self._build_q_head(self.hidden_dim, self.n_hiddens, self.activation, self.layernorm)
        self.q1_model = self.q1_model.to(device=device, dtype=dtype)
        self.q2_model = self.q2_model.to(device=device, dtype=dtype)

    def _infer_architecture_from_state_dict(self, state_dict):
        if any(key.startswith("q1_model.") for key in state_dict.keys()):
            q1_prefix = "q1_model."
            q2_prefix = "q2_model."
            activation = "mish"
        elif any(key.startswith("q1.") for key in state_dict.keys()):
            q1_prefix = "q1."
            q2_prefix = "q2."
            activation = "relu"
        else:
            return None

        linear_weights = []
        has_layernorm = False
        for key, value in state_dict.items():
            if not key.startswith(q1_prefix) or not key.endswith(".weight"):
                continue
            if value.ndim == 2:
                linear_weights.append((key, value))
            elif value.ndim == 1:
                has_layernorm = True

        if not linear_weights:
            raise RuntimeError("Unable to infer critic architecture from checkpoint: missing q1 linear weights")

        linear_weights.sort(key=lambda item: int(item[0][len(q1_prefix):].split(".", 1)[0]))
        input_dim = int(linear_weights[0][1].shape[1])
        expected_input_dim = self.state_dim + self.action_dim
        if input_dim != expected_input_dim:
            raise RuntimeError(
                f"Checkpoint critic input dim {input_dim} does not match current state+action dim {expected_input_dim}"
            )

        hidden_dim = int(linear_weights[0][1].shape[0])
        n_hiddens = len(linear_weights) - 1
        if n_hiddens < 1:
            raise RuntimeError("Checkpoint critic must contain at least one hidden layer")

        return {
            "q1_prefix": q1_prefix,
            "q2_prefix": q2_prefix,
            "hidden_dim": hidden_dim,
            "n_hiddens": n_hiddens,
            "layernorm": has_layernorm,
            "activation": activation,
        }

    def _remap_state_dict(self, state_dict, arch):
        if arch["q1_prefix"] == "q1_model." and arch["q2_prefix"] == "q2_model.":
            return state_dict

        remapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(arch["q1_prefix"]):
                new_key = "q1_model." + key[len(arch["q1_prefix"]):]
            elif key.startswith(arch["q2_prefix"]):
                new_key = "q2_model." + key[len(arch["q2_prefix"]):]
            else:
                new_key = key
            remapped[new_key] = value

        if hasattr(state_dict, "_metadata"):
            remapped._metadata = getattr(state_dict, "_metadata")
        return remapped

    def load_state_dict(self, state_dict, strict=True):
        arch = self._infer_architecture_from_state_dict(state_dict)
        if arch is not None:
            if (
                self.hidden_dim != arch["hidden_dim"]
                or self.n_hiddens != arch["n_hiddens"]
                or self.layernorm != arch["layernorm"]
                or self.activation != arch["activation"]
            ):
                self._configure_network(
                    hidden_dim=arch["hidden_dim"],
                    n_hiddens=arch["n_hiddens"],
                    activation=arch["activation"],
                    layernorm=arch["layernorm"],
                )
            state_dict = self._remap_state_dict(state_dict, arch)
        return super().load_state_dict(state_dict, strict=strict)

    def _adapt_state(self, state):
        if self.state_adapter == "identity":
            return state
        if self.state_adapter == "unnormalize":
            return state * self.state_std + self.state_mean
        raise ValueError(f"Unsupported critic state adapter: {self.state_adapter}")

    def forward(self, state, action):
        state = self._adapt_state(state)
        x = torch.cat([state, action], dim=-1)
        if self.q_mean is None:
            return self.q1_model(x), self.q2_model(x)
        else:
            return self.q1_model(x)/self.q_mean, self.q2_model(x)/self.q_mean

    def q1(self, state, action):
        state = self._adapt_state(state)
        x = torch.cat([state, action], dim=-1)
        if self.q_mean is None:
            return self.q1_model(x)
        else:
            return self.q1_model(x)/self.q_mean

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
    def set_mean(self,v):
        self.q_mean=v

class ValueNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.v(state)
class DecisionTransformer(TrajectoryModel):

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
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.config = config
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
 
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
      

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        if self.sar:
            action_preds = self.predict_action(x[:, 0])
            rewards_preds = self.predict_rewards(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
        else:
            action_preds = self.predict_action(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
            rewards_preds = None


        return state_preds, action_preds, rewards_preds

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
