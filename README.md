# sampled-laplace
This repository includes Jax code and experiments for the paper [Sampling-based inference for large linear models, with application to linearised Laplace](https://arxiv.org/abs/2210.04994).

**Edit**: This fork contains changes to apply Sampled Laplace to Pre-trained Pytorch ResNet networks from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models). Changes to `jaxutils` are inside `jaxutils_extra` so the sub-repository is unchanged. Imports are changed accordingly.

The pretrained network maintain their exact MAP metrics after these changes. The desired ResNet to use can be chosen in `experiments/cifar10_gcloud_em.py` and `experiments/cifar10_gcloud_eval.py` under the flag `config.model_name = "ResNet20"`. Options are `ResNet20`, `ResNet32`, `ResNet44` and `ResNet56`.


## Run experiments

To run stochastic EM for a linearised Laplace model using the ResNet architecture on the Cifar10 dataset on a Google Cloud TPU VM, run the following command:

```bash
python src/em_trainer.py --config experiments/cifar10_gcloud_em.py
```

To compute the last EM step with an increased number of samples do:

```bash
python src/em_trainer.py --config experiments/cifar10_gcloud_last_em.py
```

For getting results metrics do:

```bash
python src/em_eval.py --config experiments/cifar10_gcloud_eval.py
```

## Cloning the Repository

Since the repository uses submodules, it is recommended to clone the repository with the following command:

```bash
git clone --recursive git@github.com:Ludvins/sampled-laplace.git
```

## Installation Instructions

The used and tested python version is `Python 3.9.5`. Do the following commands to create an environment and install the needed packages.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
