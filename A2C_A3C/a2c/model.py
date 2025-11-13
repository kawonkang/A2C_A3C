import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc_pi = nn.Linear(128, act_dim)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc_pi(x)
        value = self.fc_v(x)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
