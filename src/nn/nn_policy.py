import torch
from torch import nn


class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, covariance, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.covariance = covariance
        assert covariance.shape == (action_dim, action_dim)

    def forward(self, state_batch):

        x = self.activation(self.fc1(state_batch))
        action_batch = torch.tanh(self.fc2(x)) * self.action_bound
        return action_batch

    def _get_distribution(self, state_batch):
        mean_action = self.forward(state_batch)
        return torch.distributions.MultivariateNormal(
            loc=mean_action, covariance_matrix=self.covariance
        )

    def log_prob(self, state_batch, action_batch):
        dist = self._get_distribution(state_batch)
        return dist.log_prob(action_batch)

    def prob(self, state_batch, action_batch):
        return torch.exp(self.log_prob(state_batch, action_batch))

    def sample_actions(self, state_batch):
        dist = self._get_distribution(state_batch)
        return dist.sample()

    def add_weights(self, weights_diff):

        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data += weights_diff[idx : idx + numel].view_as(p)
            idx += numel

    def save(self, path):
        dict_to_save = {
            "state_dict": self.state_dict(),
            "action_bound": self.action_bound,
            "covariance": self.covariance,
            "hidden_dim": self.fc1.out_features,
        }
        torch.save(dict_to_save, path)

    @classmethod
    def load(cls, path):
        dict_loaded = torch.load(path)
        model = cls(
            state_dim=dict_loaded["state_dict"]["fc1.weight"].shape[1],
            action_dim=dict_loaded["state_dict"]["fc2.weight"].shape[0],
            action_bound=dict_loaded["action_bound"],
            covariance=dict_loaded["covariance"],
            hidden_dim=dict_loaded["hidden_dim"],
        )
        model.load_state_dict(dict_loaded["state_dict"])
        return model
