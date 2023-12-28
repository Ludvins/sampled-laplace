"""Training a ResNet18 model on CIFAR100."""

import ml_collections

from jaxutils_extra.data.pt_image import METADATA


def get_config():
    """Config for training MLP on MNIST."""
    config = ml_collections.ConfigDict()

    config.use_tpu = True
    config.global_seed = 0
    config.model_seed = config.global_seed
    config.datasplit_seed = 0
    
    config.load_last_prior_prec = False

    # Dataset Configs
    config.dataset_type = "tf"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "MNIST"
    config.dataset.try_gcs = False
    if config.dataset_type == "tf" and config.dataset.try_gcs:
        config.dataset.data_dir = None
    else:
        config.dataset.data_dir = "./data/"

    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.0
    config.dataset.perform_augmentations = False
    config.dataset.num_workers = 2

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
    config.model_name = "mlp_mnist"
    config.model = ml_collections.ConfigDict()

    config.checkpoint_dir = "./converted_models/MLP/" + config.model_name + "/" + str(config.model_seed) 
    config.save_dir = "./MLP/" + config.model_name + "/" + str(config.model_seed) + "/samples"
    config.prior_save_dir = "./MLP/" + config.model_name + "/" + str(config.model_seed) + "/prior.txt"
    config.time_save_dir = "./MLP/" + config.model_name + "/" + str(config.model_seed) + "/time.txt"


    ##################### EM Step Configs #####################
    config.num_em_steps = 10

    ###################### Linear Mode Evaluation Configs ####################
    config.linear = ml_collections.ConfigDict()
    # Training Configs
    config.linear.process_batch_size = 1000
    config.linear.eval_process_batch_size = 1000  # 10000/125

    config.linear.n_epochs = 40
    config.linear.perform_eval = True
    config.linear.eval_interval = 5
    config.linear.save_interval = 10

    # Optimizer Configs
    config.linear.optim_name = "sgd"
    config.linear.optim = ml_collections.ConfigDict()
    config.linear.optim.lr = 1e-2
    config.linear.optim.nesterov = True
    config.linear.optim.momentum = 0.9

    config.linear.lr_schedule_name = "linear_schedule"
    config.linear.lr_schedule = ml_collections.ConfigDict()

    if config.linear.lr_schedule_name == "exponential_decay":
        config.linear.lr_schedule.decay_rate = 0.995
        config.linear.lr_schedule.transition_steps = 1
    linear_num_steps = int(
        config.linear.n_epochs
        * config.dataset.num_train
        * (1 - config.dataset.val_percent)
        / config.linear.process_batch_size
    )
    if config.linear.lr_schedule_name == "linear_schedule":
        config.linear.lr_schedule.decay_rate = 1 / 33
        config.linear.lr_schedule.transition_steps = int(
            linear_num_steps * 0.95
        )  # I set this to N steps * 0.75
        config.linear.lr_schedule.end_value = config.linear.optim.lr / 33

    ######################## Sample-then-Optimise Configs #####################
    config.sampling = ml_collections.ConfigDict()

    config.sampling.compute_exact_w_samples = False
    config.sampling.init_samples_at_prior = True
    config.sampling.recompute_ggn_matrix = True

    config.sampling.mackay_update = "function"

    config.sampling.use_g_prior = False
    config.sampling.use_new_objective = True

    if config.sampling.use_g_prior:
        config.prior_prec = 1.0
    else:
        config.prior_prec = 1000.0

    config.sampling.target_samples_path = (
        "./MLP/"  + config.model_name + "/" + str(config.model_seed) + "/target_samples.h5"
    )
    config.sampling.ggn_matrix_path = (
        "./MLP/"  + config.model_name + "/" + str(config.model_seed) + "/H.npy"
    )

    config.sampling.num_samples = 8

    config.sampling.H_L_jitter = 1e-6

    # Training Configs
    config.sampling.process_batch_size = 1000
    config.sampling.eval_process_batch_size = 1000  # 10000/125
    config.sampling.n_epochs = 20
    config.sampling.perform_eval = True
    config.sampling.eval_interval = 5
    config.sampling.save_interval = 10

    # Optimizer Configs
    config.sampling.optim_name = "sgd"
    config.sampling.optim = ml_collections.ConfigDict()
    if config.sampling.use_g_prior:
        config.sampling.optim.lr = 200.0
    else:
        config.sampling.optim.lr = 2e-1

    config.sampling.optim.nesterov = True
    config.sampling.optim.momentum = 0.9

    config.sampling.update_scheme = "polyak"  # polyak
    config.sampling.update_step_size = 0.01
    config.sampling.lr_schedule_name = None  # "exponential_decay"
    config.sampling.lr_schedule = ml_collections.ConfigDict()
    config.sampling.lr_schedule.decay_rate = 1
    config.sampling.lr_schedule.transition_steps = 1

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "linearised-NNs"
    config.wandb.entity = "-"
    config.wandb.code_dir = "./linearised-NNs"
    return config
