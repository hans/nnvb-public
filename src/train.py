from dataclasses import replace
from functools import partial
import logging 
from pathlib import Path
from typing import Optional

import datasets
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from ray import tune
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import transformers

from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.models import integrator

L = logging.getLogger(__name__)


def make_model_init(config: integrator.ContrastiveEmbeddingModelConfig, device="cpu"):
    def model_init(trial):
        if trial is not None:
            config_trial = replace(config,
                hidden_dim=trial["hidden_dim"],
                tau=trial.get("tau"),
                margin=trial.get("margin"))
        else:
            config_trial = config
        return integrator.ContrastiveEmbeddingModel(config_trial).to(device)  # type: ignore
    return model_init


def prepare_neg_dataset(equiv_dataset: SpeechEquivalenceDataset,
                        equiv_dataset_path: str,
                        hidden_states_path: str, **kwargs
                        ) -> tuple[int, datasets.IterableDataset, datasets.IterableDataset, int]:
    # Pick a max length that accommodates the majority of the samples,
    # excluding outlier lengths
    evident_lengths = equiv_dataset.lengths
    evident_lengths = evident_lengths[evident_lengths > 0]
    target_length = int(torch.quantile(evident_lengths.double(), 0.95).item())

    num_examples, train_dataset, eval_dataset = integrator.prepare_dataset(
        equiv_dataset, str(Path(equiv_dataset_path).resolve()),
        str(Path(hidden_states_path).resolve()),
        target_length, **kwargs)

    return num_examples, train_dataset, eval_dataset, target_length


def make_hyperparameter_space(fixed_hidden_dim: Optional[int] = None):
    def hyperparameter_space(trial):
        ret = {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
            "tau": tune.loguniform(1e-3, 1),
        }
        if fixed_hidden_dim is None:
            ret["hidden_dim"] = tune.choice([32, 64, 128, 256])
        else:
            ret["hidden_dim"] = fixed_hidden_dim
        return ret
    return hyperparameter_space

def make_hyperparameter_space_hinge(fixed_hidden_dim: Optional[int] = None):
    def hyperparameter_space_hinge(trial):
        ret = {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
            "margin": tune.loguniform(1e-3, 1),
        }
        if fixed_hidden_dim is None:
            ret["hidden_dim"] = tune.choice([32, 64, 128, 256])
        else:
            ret["hidden_dim"] = fixed_hidden_dim
        return ret
    return hyperparameter_space_hinge


HYPERPARAMETER_OBJECTIVE_DIRECTION = "maximize"
def hyperparameter_objective(metrics: dict[str, float]) -> float:
    from pprint import pprint
    pprint(metrics)
    return metrics["eval_mAP"]


def train(config: DictConfig):
    datasets.disable_caching()
    if config.device == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config.device = "cpu"
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir)
    assert not isinstance(dataset, datasets.DatasetDict), "should be a Dataset, not be a DatasetDict"

    hidden_states_path = Path(config.base_model.hidden_state_path).absolute()
    hidden_state_dataset = SpeechHiddenStateDataset.from_hdf5(hidden_states_path)

    equiv_dataset_path = config.equivalence.path
    equiv_dataset: SpeechEquivalenceDataset = torch.load(equiv_dataset_path)

    if config.get("smoke_test"):
        with equiv_dataset.modify_Q_ctx():
            equiv_dataset.Q[1000:] = -1

    # Prepare negative-sampling dataset
    if config.trainer.mode in ["train", "hyperparameter_search"]:
        total_num_examples, train_dataset, eval_dataset, max_length = prepare_neg_dataset(
            equiv_dataset, equiv_dataset_path, str(hidden_states_path))
        
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = eval_dataset.with_format("torch")
    elif config.trainer.mode == "no_train":
        total_num_examples = 0
        train_dataset, eval_dataset = None, None
        config.training_args.eval_strategy = None
        config.training_args.save_strategy = "no"
        max_length = equiv_dataset.lengths.max().item()
    else:
        raise ValueError(f"Invalid trainer mode: {config.trainer.mode}")

    max_training_steps = config.training_args.num_train_epochs * total_num_examples

    model_config = integrator.ContrastiveEmbeddingModelConfig(
        equivalence_classer=config.equivalence.equivalence_classer,
        num_classes=len(equiv_dataset.class_labels),
        max_length=max_length,
        input_dim=hidden_state_dataset.hidden_size,
        **OmegaConf.to_object(config.model))
    model_init = make_model_init(model_config, device=config.device)
    
    # Don't directly use `instantiate` with `TrainingArguments` or `Trainer` because the
    # type validation stuff is craaaaazy.
    # ^ can fix this with _recursive_ = False I think
    # We also have to use `to_object` to make sure the params are JSON-serializable
    
    model_learning_rate = config.model.get("learning_rate")
    if model_learning_rate is not None:
        L.warning("Overriding Trainer learning rate with config value from model config: %g", model_learning_rate)
        config.training_args.learning_rate = model_learning_rate

    # DEV
    L.warning("Overriding eval_steps, save_steps")
    config.training_args.eval_steps = 200
    config.training_args.save_steps = 200

    training_args = transformers.TrainingArguments(
        use_cpu=config.device == "cpu",
        output_dir=HydraConfig.get().runtime.output_dir,
        logging_dir=Path(HydraConfig.get().runtime.output_dir) / "logs",
        max_steps=max_training_steps,
        weight_decay=config.model.get("weight_decay", 0.0),
        **OmegaConf.to_object(config.training_args))

    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer_mode = trainer_config.pop("mode", "train")
    hparam_config = trainer_config.pop("hyperparameter_search", None)

    loss_form: integrator.ContrastiveLossSpec = getattr(config.model, "loss_form", "ratio")
    compute_metrics = partial(integrator.COMPUTE_METRICS[loss_form], model_config=model_config)

    trainer = transformers.Trainer(
        args=training_args,
        model=None, model_init=model_init,
        callbacks=callbacks,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        **trainer_config)

    if trainer_mode == "train":
        trainer.train()
    elif trainer_mode == "hyperparameter_search":
        fixed_hidden_dim = model_config.hidden_dim if model_config.num_layers == 0 else None
        hp_space = make_hyperparameter_space(fixed_hidden_dim)
        if loss_form == "hinge":
            hp_space = make_hyperparameter_space_hinge(fixed_hidden_dim)

        trainer.hyperparameter_search(
            direction=HYPERPARAMETER_OBJECTIVE_DIRECTION,
            backend="ray",
            n_trials=hparam_config.n_trials,
            hp_space=hp_space,
            compute_objective=hyperparameter_objective,
            resources_per_trial={"gpu": 0.32, "cpu": 1},
            scheduler=instantiate(hparam_config.scheduler,
                                  mode=HYPERPARAMETER_OBJECTIVE_DIRECTION[:3]),
        )
    elif trainer_mode == "no_train":
        checkpoint_dir = Path(training_args.output_dir) / "checkpoint-0"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        trainer.save_model(checkpoint_dir)

        # Save dummy trainer state
        trainer.state.best_model_checkpoint = str(checkpoint_dir)
        trainer.state.save_to_json(checkpoint_dir / "trainer_state.json")