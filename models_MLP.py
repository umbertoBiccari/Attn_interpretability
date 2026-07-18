import torch
import torch.nn as nn

def build_mlp(in_dim, hidden):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers), prev


# -----------------------------
# Model 1: TwoHeadLinearNet (outputs 3: [age_s, mets_s, sex_logit])
# -----------------------------
class TwoHeadLinearNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.head_am = nn.Linear(in_dim, 2)  # age_s, mets_s
        self.head_sex = nn.Linear(in_dim, 1)

    def forward(self, x):
        am = self.head_am(x)
        s = self.head_sex(x)
        return torch.cat([am, s], dim=1)


# -----------------------------
# Model 2: TwoHeadMLPNet (DeepSHAP-friendly) outputs 3
# -----------------------------
class TwoHeadMLPNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.trunk, trunk_out = build_mlp(in_dim, hidden)
        self.head_am = nn.Linear(trunk_out, 2)  # age_s, mets_s
        self.head_sex = nn.Linear(trunk_out, 1)

    def forward(self, x):
        h = self.trunk(x)
        am = self.head_am(h)
        s = self.head_sex(h)
        return torch.cat([am, s], dim=1)

class SingleOutMLPNet(nn.Module):
    """
    Single-output model that matches M2's trunk definition: build_mlp(in_dim, HIDDEN) + linear head -> 1 logit/value.
    This is the requested "single-output using model M2 (not M3)".
    """
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.trunk, trunk_out = build_mlp(in_dim, hidden)
        self.head = nn.Linear(trunk_out, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h)