import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformers, adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class TransformerForwardModel(nn.Module):
    """
    A Transformer-based forward dynamics model.
    Given (state, action) → predict next_state.
    We treat [state+action] as a single token or short sequence, feed it
    through a transformer encoder, and output next_state.
    """
    def __init__(self, state_dim, action_dim,
                 d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerForwardModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Project [state, action] from (state_dim + action_dim) → d_model
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        # Final output layer to project from d_model → state_dim
        self.fc_out = nn.Linear(d_model, state_dim)

    def forward(self, x, a):
        """
        x: (batch_size, state_dim)
        a: (batch_size, action_dim)
        Returns predicted next state: (batch_size, state_dim)
        """
        # Combine state and action into a single token
        xa = torch.cat([x, a], dim=-1)  # shape: (batch, state_dim + action_dim)
        xa_emb = self.input_projection(xa)  # (batch, d_model)

        # The PyTorch nn.Transformer modules by default expect
        # [sequence_length, batch_size, d_model].
        # So let's treat the single token as a sequence of length 1.
        xa_emb = xa_emb.unsqueeze(0)  # shape: (1, batch, d_model)

        # Positional encode
        xa_emb = self.pos_encoder(xa_emb)  # (1, batch, d_model)

        # Pass through the Transformer encoder
        encoded = self.transformer_encoder(xa_emb)  # (1, batch, d_model)
        encoded = encoded.squeeze(0)                # (batch, d_model)

        # Final projection to next state
        next_state = self.fc_out(encoded)  # (batch, state_dim)

        # Residual connection as in the original code: predict Δstate
        return next_state + x


class TransformerRewardModel(nn.Module):
    """
    A Transformer-based reward model.
    Given (state, action) → predict reward.
    """
    def __init__(self, state_dim, action_dim,
                 d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerRewardModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Project [state, action] to d_model
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        # Output dimension is 1 for scalar reward
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x, a):
        """
        x: (batch_size, state_dim)
        a: (batch_size, action_dim)
        Returns predicted reward: (batch_size,)
        """
        xa = torch.cat([x, a], dim=-1)   # (batch, state_dim + action_dim)
        xa_emb = self.input_projection(xa).unsqueeze(0)  # (1, batch, d_model)

        xa_emb = self.pos_encoder(xa_emb)                # (1, batch, d_model)
        encoded = self.transformer_encoder(xa_emb)       # (1, batch, d_model)
        encoded = encoded.squeeze(0)                     # (batch, d_model)

        reward = self.fc_out(encoded).squeeze(-1)        # (batch,)
        return reward


class TransformerDoneModel(nn.Module):
    """
    A Transformer-based done (termination) model.
    Given (state, action) → predict done (binary).
    """
    def __init__(self, state_dim, action_dim,
                 d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerDoneModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Project [state, action] to d_model
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        # Output dimension is 1 for a logit
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x, a):
        """
        x: (batch_size, state_dim)
        a: (batch_size, action_dim)
        Returns a scalar logit: (batch_size,) for BCELossWithLogits
        """
        xa = torch.cat([x, a], dim=-1)   # (batch, state_dim + action_dim)
        xa_emb = self.input_projection(xa).unsqueeze(0)  # (1, batch, d_model)

        xa_emb = self.pos_encoder(xa_emb)                # (1, batch, d_model)
        encoded = self.transformer_encoder(xa_emb)       # (1, batch, d_model)
        encoded = encoded.squeeze(0)                     # (batch, d_model)

        done_logits = self.fc_out(encoded).squeeze(-1)   # (batch,)
        return done_logits


class ModelBasedTransformer:
    """
    A model-based class similar to the original MLP-based one,
    but with Transformer-based dynamics, reward, and done models.
    """

    def __init__(self, state_dim, action_dim, learning_rate, weight_decay):
        self.dynamics_net = TransformerForwardModel(state_dim, action_dim)
        self.rewards_net = TransformerRewardModel(state_dim, action_dim)
        self.done_net = TransformerDoneModel(state_dim, action_dim)

        self.dyn_optimizer = optim.AdamW(self.dynamics_net.parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        self.reward_optimizer = optim.AdamW(self.rewards_net.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay)
        self.done_optimizer = optim.AdamW(self.done_net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

    def update(self, states, actions, next_states, rewards, masks, weights):
        """Updates model parameters using MSE for dynamics/reward and BCE for 'done'."""
        # ----- Dynamics update -----
        self.dyn_optimizer.zero_grad()
        pred_state = self.dynamics_net(states, actions)  # (batch, state_dim)
        dyn_loss = F.mse_loss(pred_state, next_states, reduction='none')
        dyn_loss = (dyn_loss * weights.unsqueeze(-1)).mean()
        dyn_loss.backward()
        self.dyn_optimizer.step()

        # ----- Reward update -----
        self.reward_optimizer.zero_grad()
        pred_rewards = self.rewards_net(states, actions)  # (batch,)
        reward_loss = F.mse_loss(pred_rewards, rewards, reduction='none')
        reward_loss = (reward_loss * weights).mean()
        reward_loss.backward()
        self.reward_optimizer.step()

        # ----- Done/termination update -----
        self.done_optimizer.zero_grad()
        pred_dones = self.done_net(states, actions)  # (batch,)
        done_loss = F.binary_cross_entropy_with_logits(pred_dones,
                                                       masks,
                                                       weight=weights)
        done_loss.backward()
        self.done_optimizer.step()

        return dyn_loss.item(), reward_loss.item(), done_loss.item()

    def estimate_returns(self, initial_states, weights, get_action, discount,
                         min_reward, max_reward, min_state, max_state,
                         clip=True, horizon=1000):
        """Compute returns via rollouts using the learned Transformer models."""
        returns = 0
        states = initial_states.clone()
        masks = torch.ones(initial_states.shape[0],
                           device=initial_states.device)

        for i in range(horizon):
            # Get actions from policy
            actions = get_action(states)

            # Predict rewards
            pred_rewards = self.rewards_net(states, actions)
            if clip:
                pred_rewards = pred_rewards.clamp(min=min_reward, max=max_reward)

            # Predict done
            logits = self.done_net(states, actions)
            mask_dist = torch.distributions.Bernoulli(logits=logits)
            new_dones = mask_dist.sample()
            masks *= (1.0 - new_dones)  # multiply by 0 if done, else 1

            # Update returns
            returns += (discount ** i) * masks * pred_rewards

            # Predict next states
            next_states = self.dynamics_net(states, actions)
            if clip:
                next_states = next_states.clamp(min=min_state, max=max_state)

            states = next_states

            # Early stopping if all done
            if masks.sum() < 1e-5:
                break

        # Weighted average returns (scaling by (1 - discount) often used for infinite-horizon)
        return (weights * returns).sum() / (weights.sum() + 1e-8) * (1 - discount)ß