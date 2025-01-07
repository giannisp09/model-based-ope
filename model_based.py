import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ForwardModel(nn.Module):
    """A class that implements a forward model."""

    def __init__(self, state_dim, action_dim):
        super(ForwardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, x, a):
        x_a = torch.cat([x, a], dim=-1)
        return self.model(x_a) + x


class RewardModel(nn.Module):
    """A class that implements a reward model."""

    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        x_a = torch.cat([x, a], dim=-1)
        return self.model(x_a).squeeze(1)


class ModelBased:
    """A class that learns models and estimates returns via rollouts."""

    def __init__(self, state_dim, action_dim, learning_rate, weight_decay):
        self.dynamics_net = ForwardModel(state_dim, action_dim)
        self.rewards_net = RewardModel(state_dim, action_dim)
        self.done_net = RewardModel(state_dim, action_dim)

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
        """Updates model parameters."""

        # Update dynamics model
        self.dyn_optimizer.zero_grad()
        pred_state = self.dynamics_net(states, actions)
        dyn_loss = F.mse_loss(pred_state, next_states, reduction='none')
        dyn_loss = (dyn_loss * weights.unsqueeze(-1)).mean()
        dyn_loss.backward()
        self.dyn_optimizer.step()

        # Update rewards model
        self.reward_optimizer.zero_grad()
        pred_rewards = self.rewards_net(states, actions)
        reward_loss = F.mse_loss(pred_rewards, rewards, reduction='none')
        reward_loss = (reward_loss * weights).mean()
        reward_loss.backward()
        self.reward_optimizer.step()

        # Update done model
        self.done_optimizer.zero_grad()
        pred_dones = self.done_net(states, actions)
        done_loss = F.binary_cross_entropy_with_logits(pred_dones, masks, weight=weights)
        done_loss.backward()
        self.done_optimizer.step()

        return dyn_loss.item(), reward_loss.item(), done_loss.item()

    def estimate_returns(self, initial_states, weights, get_action, discount,
                         min_reward, max_reward, min_state, max_state,
                         clip=True, horizon=1000):
        """Compute returns via rollouts."""
        returns = 0
        states = initial_states
        masks = torch.ones(initial_states.shape[0], device=initial_states.device)

        for i in range(horizon):
            actions = get_action(states)

            # Predict rewards
            pred_rewards = self.rewards_net(states, actions)
            if clip:
                pred_rewards = pred_rewards.clamp(min=min_reward, max=max_reward)

            # Predict done masks
            logits = self.done_net(states, actions)
            mask_dist = torch.distributions.Bernoulli(logits=logits)
            masks *= mask_dist.sample()

            # Update returns
            returns += (discount ** i) * masks * pred_rewards

            # Predict next states
            states = self.dynamics_net(states, actions)
            if clip:
                states = states.clamp(min=min_state, max=max_state)

        return (weights * returns).sum() / weights.sum() * (1 - discount)