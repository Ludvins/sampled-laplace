from typing import Any

import torch
from absl import logging
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

PyTree = Any

FC_MAP = {"weight": "kernel", "bias": "bias"}
CONV_MAP = {"weight": "kernel", "bias": "bias"}
BN_MAP = {
    "weight": "scale",
    "bias": "bias",
    "running_mean": "mean",
    "running_var": "var",
}


def convert_resnet_param_keys(pytorch_name, model_name="resnet18"):
    """Convert ResNet models from bayesian-lottery-tickets into Flax models.

    blt models have a different naming convention for the first layer weights
    and BN layers, where conv1.0.weight -> conv1.weight and conv1.1 refers to
    the BN layers.
    """
    split = pytorch_name.split(".")

    if model_name == "resnet18":
        block_name = "ResNetBlock"
        layer_list = [2, 2, 2, 2]
    elif model_name == "resnet20":
        block_name = "ResNetBlock"
        layer_list = [3, 3, 3]
    elif model_name == "resnet32":
        block_name = "ResNetBlock"
        layer_list = [5, 5, 5]
    elif model_name == "resnet44":
        block_name = "ResNetBlock"
        layer_list = [7, 7, 7]
    elif model_name == "resnet56":
        block_name = "ResNetBlock"
        layer_list = [9, 9, 9]
    else:
        raise NotImplementedError("Only resnet18 implemented for now.")

    #################### First Layer Weights and BN Layers #################
    # conv1.0.weight -> (params, conv_init, kernel)
    if split[0] == "conv1":
        if split[1] in ["bias"]:
            logging.warning("conv1.0.bias is not a valid key for BLT models")
            return None

        return ("params", "conv_init", "kernel")

    # conv1.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
    if split[0] == "bn1":
        if split[1] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[1] in ["weight", "bias"] else "batch_stats",
            "bn_init",
            BN_MAP[split[1]],
        )

    ################### Conv Layer Weights and BN Layers ###################

    # layer{i}.{j}.conv{k}.weight -> (params, ResNetBlock_{i*n_layers + j}, Conv_{k-1}, kernel)
    if (
        len(split) == 4
        and split[0][:-1] == "layer"
        and split[1].isdigit()
        and split[2][:-1] == "conv"
    ):
        if split[3] in ["bias"]:
            logging.warning(f"{pytorch_name} is not a valid key for BLT models")
            return None

        return (
            "params",
            f"{block_name}_{str((int(split[0][-1])-1)*layer_list[int(split[0][-1])-1] + int(split[1]))}",
            f"Conv_{int(split[2][-1]) - 1}",
            "kernel",
        )

    # layer{i}.{j}.bn{k}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, ResNetBlock_{i*n_layers + j}, BatchNorm_{k-1}, {scale|bias|mean|var})
    if (
        len(split) == 4
        and split[0][:-1] == "layer"
        and split[1].isdigit()
        and split[2][:-1] == "bn"
    ):
        if split[3] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[3] in ["weight", "bias"] else "batch_stats",
            f"{block_name}_{str((int(split[0][-1])-1)*layer_list[int(split[0][-1])-1] + int(split[1]))}",
            f"BatchNorm_{int(split[2][-1]) - 1}",
            BN_MAP[split[3]],
        )

    ############################ Downsampling Layers #######################

    # layer{i}.{j}.downsample.0.weight -> (params, ResNetBlock_{i*n_layers + j}, conv_proj, kernel)
    if (
        len(split) == 5
        and split[0][:-1] == "layer"
        and split[1].isdigit()
        and split[2] == "downsample"
        and split[3] == "0"
    ):
        if split[4] in ["bias"]:
            logging.warning(f"{pytorch_name} is not a valid key for BLT models")
            return None

        return ("params", f"{block_name}_{str((int(split[0][-1])-1)*layer_list[int(split[0][-1])-1] + int(split[1]))}",
                 "conv_proj", "kernel")

    # layer{i}.{j}.downsample.1.{weight|bias|running_mean|running_var} -> 
    # ({params|batch_stats}, ResNetBlock_{{i*n_layers + j}, norm_proj, {scale|bias|mean|var})
    if (
        len(split) == 5
        and split[0][:-1] == "layer"
        and split[1].isdigit()
        and split[2] == "downsample"
        and split[3] == "1"
    ):
        if split[4] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[4] in ["weight", "bias"] else "batch_stats",
            f"{block_name}_{str((int(split[0][-1])-1)*layer_list[int(split[0][-1])-1] + int(split[1]))}",
            "norm_proj",
            BN_MAP[split[4]],
        )

    ############################# Output Layers ############################
    # fc.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if len(split) == 2 and split[0] == "fc":
        return ("params", "Dense_0", FC_MAP[split[1]])

def convert_mlp_param_keys(pytorch_name, model_name="mlp_mnist"):
    split = pytorch_name.split(".")
    if len(split) == 2 and split[0] == "0":
        return ("params", "dense1", FC_MAP[split[1]])

    elif len(split) == 2 and split[0] == "3":
        return ("params", "dense2", FC_MAP[split[1]])

    elif len(split) == 2 and split[0] == "6":
        return ("params", "dense3", FC_MAP[split[1]])



def convert_model(
    model_name: str, pytorch_model: torch.nn.Module, jax_params: PyTree
) -> PyTree:
    """Utility function to transfer params from Pytorch models to Flax."""
    # TODO: Add a check that params is frozen and not flat, and not repeat
    jax_params = flatten_dict(unfreeze(jax_params))
    if model_name in ["resnet18", "resnet20", "resnet32", "resnet44", "resnet56"]:
        convert_keys = convert_resnet_param_keys
    elif model_name in ["mlp_mnist", "mlp_fmnist"]:
        convert_keys = convert_mlp_param_keys
    else:
        raise NotImplementedError(f"{model_name} conversion not implemented yet.")

    jax_to_pytorch_keys, converted_jax_params = {}, {}
    for key, param in pytorch_model.state_dict().items():
        if convert_keys(key, model_name) is not None:
            jax_to_pytorch_keys[convert_keys(key, model_name)] = key

        if len(param.shape) != 4:
            converted_jax_params[key] = param.numpy().T
        else:
            # Pytorch # [outC, inC, kH, kW] -> Jax [kH, kW, inC, outC]
            converted_jax_params[key] = param.numpy().transpose((2, 3, 1, 0))
    new_jax_params = {
        key: converted_jax_params[jax_to_pytorch_keys[key]] for key in jax_params.keys()
    }
    new_jax_params = freeze(unflatten_dict(new_jax_params))

    return new_jax_params
