import torch
import torch.nn as nn


class ProtoClassifier(nn.Module):
    """Learnable prototype vectors, shape [n_views, n_class * n_prototypes, hidden_dim]."""
    def __init__(self, hidden_dim, n_class, n_prototypes, n_views=2):
        super().__init__()
        self.n_class = n_class
        self.n_prototypes = n_prototypes
        self.n_views = n_views

        self.prototypes = nn.Parameter(
            torch.randn(n_views, n_class * n_prototypes, hidden_dim)
        )
        self.prototype_labels = torch.arange(n_class).repeat_interleave(n_prototypes)

    def get_prototypes(self):
        return self.prototypes
