"""Training a LeNetSmall model on MNIST."""

import ml_collections

from jaxutils.data.pt_image import METADATA


def get_config():
    """Config for evaling CIFAR100 on CIFAR100-C."""
    config = ml_collections.ConfigDict()
    config.use_tpu = True
    config.global_seed = 0
    config.model_seed = 0

    # Dataset Configs
    config.dataset_type = "pytorch"
    config.eval_dataset = "corrupted"
    config.method = "sampled_laplace"# "sampled_laplace"  # "map"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "CIFAR10"
    config.dataset.try_gcs = True
    if config.dataset_type == "tf" and config.dataset.try_gcs:
        config.dataset.data_dir = None
    else:
        config.dataset.data_dir = "./data/"

    config.dataset.num_workers = 16

    config.dataset.cache = False
    config.dataset.repeat_after_batching = False
    config.dataset.shuffle_train_split = True
    config.dataset.shuffle_eval_split = False
    config.dataset.shuffle_buffer_size = 10_000
    config.dataset.prefetch_size = 4
    config.dataset.prefetch_on_device = None
    config.dataset.drop_remainder = True

    # Add METADATA information from jaxutils
    for key in METADATA:
        config.dataset[key] = METADATA[key][config.dataset.dataset_name]

    # Model Configs
    config.model_name = "resnet20"
    config.model = ml_collections.ConfigDict()
    config.model.num_classes = config.dataset.num_classes
    config.model.initial_conv = "1x3"

    config.checkpoint_dir = "./CIFAR10/" + config.model_name + "/" + "g1000_fn_8samples" 

    ##################### EM Step Configs #####################
    config.num_em_steps = 8

    ######################## Sample-then-Optimise Configs #####################
    config.sampling = ml_collections.ConfigDict()

    config.sampling.prediction_method = "gibbs"

    # Training Configs
    config.sampling.eval_process_batch_size = 200  # 10000/125

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "linearised-NNs"
    config.wandb.entity = "-"
    config.wandb.code_dir = "./linearised-NNs"
    return config
