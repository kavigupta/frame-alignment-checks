"""
Loads the LSSI models. Minimal amount of code necessary to do this. Largely sourced from the SAM repository, specifically

https://github.com/kavigupta/sam/blob/daad8967d85d7612f0487abf3b00e37c78049b22/spliceai/Canonical/modular_splicing/models/lssi.py
https://github.com/kavigupta/sam/blob/daad8967d85d7612f0487abf3b00e37c78049b22/spliceai/Canonical/modular_splicing/legacy/remapping_pickle.py
"""

import pickle

import torch
import torch.nn as nn


class renamed_symbol_unpickler(pickle.Unpickler):
    """
    Unpicler that renames modules and symbols as specified in the
    MODULE_RENAME_MAP and SYMBOL_RENAME_MAP dictionaries.
    """

    def find_class(self, module, name):
        if module == "splice_point_identifier":
            # refer back to this module
            module = "tests.models.lssi"
        try:
            return super(renamed_symbol_unpickler, self).find_class(module, name)
        except:
            print("Could not find", (module, name))
            raise


class remapping_pickle:
    """
    An instance of this class will behave like the pickle module, but
    will use the renamed_symbol_unpickler class instead of the default
    Unpickler class.
    """

    def __getattribute__(self, name):
        if name == "Unpickler":
            return renamed_symbol_unpickler
        return getattr(pickle, name)

    def __hasattr__(self, name):
        return hasattr(pickle, name)


def load_with_remapping_pickle(*args, **kwargs):
    """
    Behaves like torch.load, but re-maps modules.
    """
    return torch.load(*args, **kwargs, pickle_module=remapping_pickle())


class SplicePointIdentifier(nn.Module):
    def __init__(
        self,
        cl,
        asymmetric_cl,
        hidden_size,
        n_layers=3,
        starting_channels=4,
        input_size=4,
        sparsity=None,
    ):
        super().__init__()
        del input_size, sparsity
        assert cl % 2 == 0
        if asymmetric_cl is None:
            first_layer = nn.Conv1d(starting_channels, hidden_size, cl + 1)
        else:
            first_layer = AsymmetricConv(
                starting_channels, hidden_size, cl, *asymmetric_cl
            )
        conv_layers = [first_layer] + [
            nn.Conv1d(hidden_size, hidden_size, 1) for _ in range(n_layers)
        ]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.activation = nn.ReLU()
        self.last_layer = nn.Conv1d(hidden_size, 3, 1)

    def forward(self, x, collect_intermediates=False, collect_losses=False):
        if isinstance(x, dict):
            x = x["x"]
        x = x.transpose(2, 1)
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        x = x.transpose(2, 1)
        if collect_intermediates or collect_losses:
            return dict(output=x)
        return x


class AsymmetricConv(nn.Module):
    clipping = "cl-based"

    def __init__(self, in_channels, out_channels, cl, left, right):
        super().__init__()
        assert cl % 2 == 0
        assert max(left, right) <= cl // 2
        self.conv = nn.Conv1d(in_channels, out_channels, left + right + 1)
        self.cl = cl
        self.left = left
        self.right = right

    def forward(self, x):
        x = torch.cat(
            [
                torch.zeros(*x.shape[:-1], self.left).to(x.device),
                x,
                torch.zeros(*x.shape[:-1], self.right).to(x.device),
            ],
            dim=-1,
        )
        x = self.conv(x)
        if self.clipping == "cl-based":
            x = x[:, :, self.cl // 2 : x.shape[-1] - self.cl // 2]
        elif self.clipping == "natural":
            x = x[:, :, self.left : -self.right]
        elif self.clipping == "none":
            pass
        else:
            raise RuntimeError(f"bad value for self.clipping: {self.clipping}")
        return x
