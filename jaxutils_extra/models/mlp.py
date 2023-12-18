from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

Moduledef = Any
class mlp_mnist(nn.Module):
    dtype: Any = jnp.float32

    def setup(self):
        self.dense1 = nn.Dense(200, dtype=self.dtype)
        self.dense2 = nn.Dense(200, dtype=self.dtype)
        self.dense3 = nn.Dense(10, dtype=self.dtype)

    def __call__(self, x, train: bool = True):
        x = jnp.reshape(x, (x.shape[0], -1))

        x = self.dense1(x)
        x = nn.tanh(x)
        x = self.dense2(x)
        x = nn.tanh(x)
        x = self.dense3(x)
        x = jnp.asarray(x, self.dtype)

        return x
    

class mlp_fmnist(nn.Module):
    dtype: Any = jnp.float32

    def setup(self):
        self.dense1 = nn.Dense(200, dtype=self.dtype)
        self.dense2 = nn.Dense(200, dtype=self.dtype)
        self.dense3 = nn.Dense(10, dtype=self.dtype)

    def __call__(self, x, train: bool = True):
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.dense1(x)
        x = nn.tanh(x)
        x = self.dense2(x)
        x = nn.tanh(x)
        x = self.dense3(x)
        x = jnp.asarray(x, self.dtype)

        return x