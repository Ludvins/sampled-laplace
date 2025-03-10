"""Flax implementation of ResNet V1. Stolen from https://github.com/google/flax/blob/master/examples/imagenet/models.py"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    downsample: bool
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides, padding=((1, 1), (1, 1)))(x)
        # ^ Not using Flax default padding since it doesn't match PyTorch
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), padding=((1, 1), (1, 1)))(y)
        # ^ Not using Flax default padding since it doesn't match PyTorch

        # For pretrained bayesian-lottery-tickets models, don't init with 0 here.
        y = self.norm()(y)

        if self.downsample:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    downsample : bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides, padding=((1, 1), (1, 1)))(y)
        # ^ Not using Flax default padding since it doesn't match PyTorch
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if self.downsample:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    initial_conv: str = "1x3"

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        if self.initial_conv == "1x3":
            x = conv(
                self.num_filters,
                (3, 3),
                (1, 1),
                padding=[(1, 1), (1, 1)],
                name="conv_init",
            )(x)
            x = norm(name="bn_init")(x)
            x = nn.relu(x)
        elif self.initial_conv == "1x7":
            x = conv(
                self.num_filters,
                (7, 7),
                (2, 2),
                padding=[(3, 3), (3, 3)],
                name="conv_init",
            )(x)
            x = norm(name="bn_init")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        # TODO: Add (3x3) conv block option here.

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                downsample = i > 0 and j == 0
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    downsample = downsample,
                    norm=norm,
                    act=self.act,
                )(x)
        # Similar to PyTorch nn.AdaptiveAvgPool2d((1, 1))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        # x = nn.log_softmax(x)  # to match the Torch implementation at https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)

resnet20 = partial(ResNet, stage_sizes=[3, 3, 3], num_filters=16,
                   block_cls=ResNetBlock)

resnet32 = partial(ResNet, stage_sizes=[5, 5, 5], num_filters=16,
                   block_cls=ResNetBlock)

resnet44 = partial(ResNet, stage_sizes=[7, 7, 7], num_filters=16,
                   block_cls=ResNetBlock)

resnet56 = partial(ResNet, stage_sizes=[9, 9, 9], num_filters=16,
                   block_cls=ResNetBlock)