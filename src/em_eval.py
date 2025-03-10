import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections.config_flags
import torch
from tqdm import tqdm
from absl import app, flags
from tensorflow_probability.substrates.jax.stats import expected_calibration_error
from flax.training import checkpoints

import jaxutils_extra.models as models
import wandb
from jaxutils_extra.pt_ood import load_corrupted_dataset, load_rotated_dataset,load_ood_dataset
from jaxutils.data.utils import get_agnostic_iterator
from jaxutils_extra.train.classification import create_eval_step
from jaxutils.train.utils import eval_epoch
from jaxutils.utils import flatten_nested_dict, setup_training, update_config_dict
from src.sampling import SamplingPredictState, create_sampling_prediction_step
from src.sampling_train_utils import eval_sampled_laplace_epoch

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "./experiments/mnist_gcloud_eval.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        ####################### Refresh Config Dicts #########################
        # Update config file with run.config to update hparam_sweep values
        config.unlock()
        config.update_from_flattened_dict(run.config)
        # Add hparams that need to be computed from sweeped hparams to configs
        computed_configs = {}
        update_config_dict(config, run, computed_configs)
        config.lock()
        # Setup training flags and log to Wandb
        setup_training(run)

        ######################## Set up random seeds #########################
        seed = config.get("global_seed", 0)
        torch.manual_seed(seed)

        model_rng = random.PRNGKey(config.model_seed)

        ################ Create and initialise model ##########################
        model_cls = getattr(models, config.model_name)
        model = model_cls(**config.model.to_dict())
        print(model)

        dummy_init = jnp.expand_dims(jnp.ones(config.dataset.image_shape), 0)
        variables = model.init(model_rng, dummy_init)
        import flax
        model_state, params = flax.core.pop(variables, 'params')

        ################# Load from checkpoint ################################
        # checkpoint_dir = checkpoint_dir / "w_samples"
        em_step = 0
        checkpoint_dir = Path(config.checkpoint_dir).resolve()
        checkpoint_dir = checkpoint_dir / f"em_{em_step}" / "w_samples"
        print(checkpoint_dir)
        checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
        restored_state = checkpoints.restore_checkpoint(
            checkpoint_path, target=None
        )

        state = SamplingPredictState.create(
            apply_fn=model.apply,
            tx=None,
            params=restored_state["params"],
            model_state=restored_state["model_state"],
            w_lin=restored_state["w_lin"],
            prior_prec=restored_state["prior_prec"],
            w_samples=restored_state["w_samples"],
            avg_w_samples=restored_state["avg_w_samples"],
            scale_vec=restored_state["scale_vec"],
        )

        if config.eval_dataset == "corrupted":
            severity = [1, 2, 3, 4, 5]
            corruption_type = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
            load_dataset_fn = load_corrupted_dataset
        elif config.eval_dataset == "rotated":
            angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
            load_dataset_fn = load_rotated_dataset
        elif config.eval_dataset == "original":
            if config.dataset.dataset_name == "CIFAR10":
                severity = [0]
                corruption_type = [0]
                load_dataset_fn = load_corrupted_dataset
            else:
                angles = [0]
                load_dataset_fn = load_rotated_dataset


        if config.method == "sampled_laplace":
            predict_step = create_sampling_prediction_step(model, config.prediction_method)
        elif config.method == "map":
            predict_step = create_eval_step(
                model, num_classes=config.dataset.num_classes
            )

        # Create parallel version of the train and eval step
        p_predict_step = jax.pmap(predict_step, "device")
        state = state.replicate()
        metrics = {}

        if config.dataset.dataset_name == "CIFAR10":
            if config.eval_dataset == "original":
                split_loader, split_dataset = load_dataset_fn(
                            config.dataset.dataset_name,
                            0,
                            0,
                            config.dataset.data_dir,
                            batch_size=config.sampling.eval_process_batch_size,
                            num_workers=config.dataset.num_workers,
                        )

                n_dataset = len(split_dataset)
                steps_per_epoch = len(split_loader)

                if config.method == "sampled_laplace":
                    _, predict_metrics, preds, labels = eval_sampled_laplace_epoch(
                        predict_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        eval_log_prefix=f"original",
                        dataset_type=config.dataset_type,
                        aux_log_dict={"em_step": em_step},
                        rng=jax.random.PRNGKey(s),
                    )
                    l = []
                    p = []
                    for i in range(len(labels)):
                        for j in range(len(labels[0])):
                            l.append(labels[i][j])
                            p.append(preds[i][j])
                    labels = jax.numpy.concatenate(l)
                    preds = jax.numpy.concatenate(p)

                    ece = expected_calibration_error(15, preds, labels)
                    metrics = {
                        **metrics,
                        **{k: v.item() for k, v in predict_metrics.items()},
                        **{f"original/ece": ece}
                    }
                elif config.method == "map":
                    predict_metrics = eval_epoch(
                        eval_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        log_prefix=f"original",
                        dataset_type=config.dataset_type,
                    )
                    print(predict_metrics)
            else:
                with tqdm(total=(len(severity)-1)*len(corruption_type) + 1) as pbar:
                    for s in severity:
                        for t in corruption_type:
                            split_loader, split_dataset = load_dataset_fn(
                                config.dataset.dataset_name,
                                s,
                                t,
                                config.dataset.data_dir,
                                batch_size=config.sampling.eval_process_batch_size,
                                num_workers=config.dataset.num_workers,
                            )

                            n_dataset = len(split_dataset)
                            steps_per_epoch = len(split_loader)

                            if config.method == "sampled_laplace":
                                _, predict_metrics, preds, labels = eval_sampled_laplace_epoch(
                                    predict_step_fn=p_predict_step,
                                    data_iterator=get_agnostic_iterator(
                                        split_loader, config.dataset_type
                                    ),
                                    steps_per_epoch=steps_per_epoch,
                                    num_points=n_dataset,
                                    state=state,
                                    wandb_run=run,
                                    eval_log_prefix=f"severity_{s}_type_{t}",
                                    dataset_type=config.dataset_type,
                                    aux_log_dict={"severity": s, "type": t, "em_step": em_step},
                                    rng=jax.random.PRNGKey(s),
                                )
                                l = []
                                p = []
                                for i in range(len(labels)):
                                    for j in range(len(labels[0])):
                                        l.append(labels[i][j])
                                        p.append(preds[i][j])
                                labels = jax.numpy.concatenate(l)
                                preds = jax.numpy.concatenate(p)

                                ece = expected_calibration_error(15, preds, labels)
                                metrics = {
                                    **metrics,
                                    **{k: v.item() for k, v in predict_metrics.items()},
                                    **{f"severity_{s}_type_{t}/ece": ece}
                                }
                            elif config.method == "map":
                                predict_metrics = eval_epoch(
                                    eval_step_fn=p_predict_step,
                                    data_iterator=get_agnostic_iterator(
                                        split_loader, config.dataset_type
                                    ),
                                    steps_per_epoch=steps_per_epoch,
                                    num_points=n_dataset,
                                    state=state,
                                    wandb_run=run,
                                    log_prefix=f"severity_{s}_type_{t}",
                                    dataset_type=config.dataset_type,
                                )
                                print(predict_metrics)

                            pbar.update(1)

        elif config.dataset.dataset_name in ["MNIST", "FMNIST"]:
            if config.eval_dataset == "original":
                split_loader, split_dataset = load_dataset_fn(
                            config.dataset.dataset_name,
                            0,
                            config.dataset.data_dir,
                            batch_size=config.sampling.eval_process_batch_size,
                            num_workers=config.dataset.num_workers,
                        )

                n_dataset = len(split_dataset)
                steps_per_epoch = len(split_loader)

                if config.method == "sampled_laplace":
                    _, predict_metrics, preds, labels = eval_sampled_laplace_epoch(
                        predict_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        eval_log_prefix=f"original",
                        dataset_type=config.dataset_type,
                        aux_log_dict={"em_step": em_step},
                        rng=jax.random.PRNGKey(0),
                    )
                    l = []
                    p = []
                    for i in range(len(labels)):
                        for j in range(len(labels[0])):
                            l.append(labels[i][j])
                            p.append(preds[i][j])
                    labels = jax.numpy.concatenate(l)
                    preds = jax.numpy.concatenate(p)

                    ece = expected_calibration_error(15, preds, labels)
                    metrics = {
                        **metrics,
                        **{k: v.item() for k, v in predict_metrics.items()},
                        **{f"original/ece": ece}
                    }

                elif config.method == "map":
                    predict_metrics = eval_epoch(
                        eval_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        log_prefix=f"original",
                        dataset_type=config.dataset_type,
                    )
                    print(predict_metrics)
            else:
                with tqdm(total=(len(angles))) as pbar:
                    for angle in angles:
                        split_loader, split_dataset = load_dataset_fn(
                            config.dataset.dataset_name,
                            angle,
                            config.dataset.data_dir,
                            batch_size=config.sampling.eval_process_batch_size,
                            num_workers=config.dataset.num_workers,
                        )

                        n_dataset = len(split_dataset)
                        steps_per_epoch = len(split_loader)

                        if config.method == "sampled_laplace":
                            _, predict_metrics, preds, labels = eval_sampled_laplace_epoch(
                                predict_step_fn=p_predict_step,
                                data_iterator=get_agnostic_iterator(
                                    split_loader, config.dataset_type
                                ),
                                steps_per_epoch=steps_per_epoch,
                                num_points=n_dataset,
                                state=state,
                                wandb_run=run,
                                eval_log_prefix=f"angle_{angle}",
                                dataset_type=config.dataset_type,
                                aux_log_dict={"angle": angle, "em_step": em_step},
                                rng=jax.random.PRNGKey(angle),
                            )
                            l = []
                            p = []
                            for i in range(len(labels)):
                                for j in range(len(labels[0])):
                                    l.append(labels[i][j])
                                    p.append(preds[i][j])
                            labels = jax.numpy.concatenate(l)
                            preds = jax.numpy.concatenate(p)

                            ece = expected_calibration_error(15, preds, labels)
                            metrics = {
                                **metrics,
                                **{k: v.item() for k, v in predict_metrics.items()},
                                **{f"angle_{angle}/ece": ece}
                            }
                        elif config.method == "map":
                            predict_metrics = eval_epoch(
                                eval_step_fn=p_predict_step,
                                data_iterator=get_agnostic_iterator(
                                    split_loader, config.dataset_type
                                ),
                                steps_per_epoch=steps_per_epoch,
                                num_points=n_dataset,
                                state=state,
                                wandb_run=run,
                                log_prefix=f"angle_{angle}",
                                dataset_type=config.dataset_type,
                            )
                        pbar.update(1)

                    ood_loader, ood_dataset = load_ood_dataset(
                            config.dataset.dataset_name,
                            config.dataset.data_dir,
                            batch_size=config.sampling.eval_process_batch_size,
                            num_workers=config.dataset.num_workers,
                        )
                
                    n_dataset = len(ood_dataset)
                    steps_per_epoch = len(ood_loader)
                
                    _, _, preds, labels = eval_sampled_laplace_epoch(
                                predict_step_fn=p_predict_step,
                                data_iterator=get_agnostic_iterator(
                                    ood_loader, config.dataset_type
                                ),
                                steps_per_epoch=steps_per_epoch,
                                num_points=n_dataset,
                                state=state,
                                wandb_run=run,
                                eval_log_prefix=f"ood",
                                dataset_type=config.dataset_type,
                                aux_log_dict={"em_step": em_step},
                                rng=jax.random.PRNGKey(0),
                            )
                    l = []
                    p = []
                    for i in range(len(labels)):
                        for j in range(len(labels[0])):
                            l.append(labels[i][j])
                            p.append(preds[i][j])
                    labels = jax.numpy.concatenate(l)
                    preds = jax.numpy.concatenate(p)
                    probs = jax.nn.softmax(preds, -1)
                    # Compute Entropy
                    H = - jnp.sum(probs * jnp.log(probs), -1)   

                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(labels, H)
                    metrics = {
                                **metrics,
                                **{f"OOD AUC": auc}
                            }
                    

            import pandas as pd
            df = pd.DataFrame.from_dict(metrics, orient="index").transpose()

            print(df)
            if config.eval_dataset == "original":
                name = "results/" + config.model_name + "_original_" + str(config.model_seed) + ".csv"
            else:
                name = "results/" + config.model_name + "_" + str(config.model_seed) + ".csv"

            df.to_csv(
                path_or_buf=name,
                encoding="utf-8",
            )


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
