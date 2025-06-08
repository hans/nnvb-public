"""
Defines methods for training and evaluating word recognition
classifiers on model embeddings.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
import transformers
from tqdm.auto import tqdm

from src.analysis import state_space as ss
from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.models import get_best_checkpoint


L = logging.getLogger(__name__)


def prepare_trajectories(embeddings, state_space_spec, config):
    trajectory = ss.prepare_state_trajectory(
        embeddings, state_space_spec, pad=np.nan)
    
    # aggregate the trajectory
    featurization = getattr(config.embeddings, "featurization", None)
    if featurization is not None:
        trajectory = ss.aggregate_state_trajectory(
            trajectory, state_space_spec, tuple(featurization)
        )
    flat_traj, flat_traj_src = ss.flatten_trajectory(trajectory)
    max_num_frames = flat_traj_src[:, 2].max() + 1

    item_idxs = np.zeros((flat_traj.shape[0],), dtype=int) - 1
    onset_idxs = np.zeros((flat_traj.shape[0],), dtype=int) - 1
    offset_idxs = np.zeros((flat_traj.shape[0],), dtype=int) - 1
    if state_space_spec.cuts is not None:
        # Retrieve item idxs and frame idxs so that we can reconstruct
        # precise stimulus from the model output data
        cuts_df = state_space_spec.cuts.xs("phoneme", level="level")
        cuts_df["label_idx"] = cuts_df.index.get_level_values("label").map({l: i for i, l in enumerate(state_space_spec.labels)})
        cuts_df["frame_idx"] = cuts_df.groupby(["label", "instance_idx"]).cumcount()
        cuts_df = cuts_df.reset_index().set_index(["label_idx", "instance_idx", "frame_idx"]).sort_index()

        for i, (label_idx, instance_idx, frame_idx) in enumerate(flat_traj_src):
            cut_row = cuts_df.loc[label_idx, instance_idx, frame_idx]
            item_idxs[i] = cut_row["item_idx"]
            onset_idxs[i] = cut_row["onset_frame_idx"]
            offset_idxs[i] = cut_row["offset_frame_idx"]

    # Group by frame
    flat_trajs_by_frame = []
    for frame in range(max_num_frames):
        mask = flat_traj_src[:, 2] == frame
        flat_trajs_by_frame.append((flat_traj[mask], flat_traj_src[mask, 0], flat_traj_src[mask, 1],
                                    item_idxs[mask], onset_idxs[mask], offset_idxs[mask]))

    return flat_trajs_by_frame


class MyDataset(Dataset):
    def __init__(self, idxs,
                 embeddings, labels, label_instance_idxs,
                 item_idxs, onset_frame_idxs, offset_frame_idxs):
        self.idxs = idxs

        self.embeddings = embeddings
        self.labels = labels
        self.label_instance_idxs = label_instance_idxs

        self.item_idxs = item_idxs
        self.onset_frame_idxs = onset_frame_idxs
        self.offset_frame_idxs = offset_frame_idxs

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"idxs": self.idxs[idx],
                "inputs": self.embeddings[idx],
                "labels": self.labels[idx],
                "label_instance_idxs": self.label_instance_idxs[idx],
                "item_idxs": self.item_idxs[idx],
                "onset_frame_idxs": self.onset_frame_idxs[idx],
                "offset_frame_idxs": self.offset_frame_idxs[idx]}
    

@dataclass
class MyModelOutput(transformers.utils.ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    class_pred: torch.Tensor = None
    

class MyModel(nn.Module):
    def __init__(self, input_dim, num_labels, return_logits=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)
        self.return_logits = return_logits

    def forward(self, inputs, labels=None,
                return_logits=None, **kwargs):
        return_logits = return_logits if return_logits is not None else self.return_logits

        logits = self.fc(inputs)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        class_pred = logits.argmax(dim=1)
        
        return MyModelOutput(
            loss=loss,
            logits=logits if return_logits else None,
            class_pred=class_pred,
        )
    

# Used in evaluations
class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def __str__(self):
        return f"EnsembleModel({len(self.models)} models)"
    
    __repr__ = __str__
    
    def forward(self, inputs, return_variance=False):
        logits_list = [model(inputs, return_logits=True).logits
                       for model in self.models]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        probabilities = nn.functional.softmax(avg_logits, dim=1)

        if not return_variance:
            return probabilities
        
        # compute variance over probabilities estimated under each model
        variance = torch.stack(logits_list).softmax(dim=2).var(dim=0)
        return probabilities, variance


def load_ensemble_models(frame_idxs: Union[int, list[int]], model_dir, device="cpu"):
    if isinstance(frame_idxs, int):
        frame_idxs = [frame_idxs]

    models = []
    for frame_idx in frame_idxs:
        for split_idx in range(3):  # Assuming 3 CV splits per frame
            model_path = Path(model_dir) / f"frame_{frame_idx}-split_{split_idx}"
            ckpt_path = get_best_checkpoint(model_path)
            if ckpt_path is None:
                continue
            ckpt_path = Path(ckpt_path) / "pytorch_model.bin"
            ckpt_data = torch.load(ckpt_path, map_location=device)

            num_labels, input_dim = ckpt_data["fc.weight"].shape
            model = MyModel(input_dim, num_labels).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            models.append(model)

    if not models:
        raise ValueError(f"No model found for frames {frame_idxs}")

    vocabulary_path = ckpt_path.parents[2] / "vocabulary.txt"
    with open(vocabulary_path, "r") as f:
        vocabulary = f.read().strip().splitlines()

    return EnsembleModel(models).to(device), vocabulary


def prepare_dataset(embeddings, labels, label_instance_idxs,
                    item_idxs, onset_frame_idxs, offset_frame_idxs,
                    num_splits=5) -> list[tuple[Dataset, Dataset]]:
    assert embeddings.shape[0] == labels.shape[0] == label_instance_idxs.shape[0]

    # l2 norm
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # do stratified k-fold split
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    datasets = []
    for train_idx, eval_idx in skf.split(embeddings, labels):
        train_embeddings, train_labels, train_label_instance_idxs = \
            embeddings[train_idx], labels[train_idx], label_instance_idxs[train_idx]
        train_item_idxs, train_onset_frame_idxs, train_offset_frame_idxs = \
            item_idxs[train_idx], onset_frame_idxs[train_idx], offset_frame_idxs[train_idx]

        eval_embeddings, eval_labels, eval_label_instance_idxs = \
            embeddings[eval_idx], labels[eval_idx], label_instance_idxs[eval_idx]
        eval_item_idxs, eval_onset_frame_idxs, eval_offset_frame_idxs = \
            item_idxs[eval_idx], onset_frame_idxs[eval_idx], offset_frame_idxs[eval_idx]

        datasets.append((MyDataset(torch.tensor(train_idx).long(),
                                   torch.tensor(train_embeddings).float(),
                                   torch.tensor(train_labels).long(),
                                   torch.tensor(train_label_instance_idxs).long(),
                                   torch.tensor(train_item_idxs).long(),
                                   torch.tensor(train_onset_frame_idxs).long(),
                                   torch.tensor(train_offset_frame_idxs).long()),
                         MyDataset(torch.tensor(eval_idx).long(),
                                   torch.tensor(eval_embeddings).float(),
                                   torch.tensor(eval_labels).long(),
                                   torch.tensor(eval_label_instance_idxs).long(),
                                   torch.tensor(eval_item_idxs).long(),
                                   torch.tensor(eval_onset_frame_idxs).long(),
                                   torch.tensor(eval_offset_frame_idxs).long())))

    return datasets


def compute_metrics(p: transformers.EvalPrediction):
    labels, idxs, instance_idxs = p.label_ids[:3]
    preds = p.predictions[-1]
    return {"accuracy": (preds == labels).mean()}


def train(config: DictConfig):
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            L.error("CUDA is not available. Falling back to CPU.")
            config["device"] = "cpu"

    hidden_states = SpeechHiddenStateDataset.from_hdf5(config.base_model.hidden_state_path)
    state_space_spec = ss.StateSpaceAnalysisSpec.from_hdf5(config.analysis.state_space_specs_path, "word")
    assert state_space_spec.is_compatible_with(hidden_states)
    embeddings = np.load(config.model.embeddings_path)

    # Subsample state space according to config
    L.info(f"Keeping top {config.recognition_model.evaluation.keep_top_k} labels (out of {len(state_space_spec.labels)})")
    state_space_spec = state_space_spec.keep_top_k(config.recognition_model.evaluation.keep_top_k)
    state_space_spec = state_space_spec.subsample_instances(config.recognition_model.evaluation.subsample_instances)
    L.info(f"Keeping labels with at least {config.recognition_model.evaluation.min_instances_per_label} instances")
    state_space_spec = state_space_spec.keep_min_frequency(config.recognition_model.evaluation.min_instances_per_label)
    L.info(f"Final number of labels: {len(state_space_spec.labels)}")

    trajectories = prepare_trajectories(embeddings, state_space_spec, config.recognition_model)
    datasets = {
        frame_idx: prepare_dataset(
            *traj,
            num_splits=config.recognition_model.evaluation.num_stratified_splits)
        for frame_idx, traj in enumerate(trajectories)
    }
    all_labels = state_space_spec.labels

    device = torch.device(config.device)
    def make_model():
        return MyModel(embeddings.shape[1], len(all_labels)).to(device)

    # Overrides -- hacky because we're pulling a config from the main model config
    config.training_args.per_device_train_batch_size = config.recognition_model.evaluation.train_batch_size
    config.training_args.num_train_epochs = config.recognition_model.evaluation.num_train_epochs
    config.training_args.label_names = ["labels", "idxs", "label_instance_idxs",
                                        "item_idxs", "onset_frame_idxs", "offset_frame_idxs"]
    config.training_args.learning_rate = config.recognition_model.optimizer.lr

    training_args = transformers.TrainingArguments(
        use_cpu=config.device == "cpu",
        output_dir=HydraConfig.get().runtime.output_dir,
        logging_dir=Path(HydraConfig.get().runtime.output_dir) / "logs",
        per_device_eval_batch_size=config.recognition_model.evaluation.eval_batch_size,
        eval_accumulation_steps=5,
        **OmegaConf.to_object(config.training_args))
    
    callbacks = []
    if "callbacks" in config.trainer:
        callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer_config = dict(config.trainer)
    trainer_config.pop("callbacks", None)
    trainer_mode = trainer_config.pop("mode", "train")
    hparam_config = trainer_config.pop("hyperparameter_search", None)

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Save vocabulary to output directory
    with open(output_dir / "vocabulary.txt", "w") as f:
        f.write("\n".join(all_labels))

    for frame_idx, datasets in tqdm(datasets.items(), unit="frame"):
        all_test_evaluations, all_test_outputs = [], []
        for split_idx, (train_dataset, test_dataset) in enumerate(datasets):
            model = make_model()
            model_dir = output_dir / f"frame_{frame_idx}-split_{split_idx}"
            training_args.output_dir = str(model_dir)
            training_args.logging_dir = str(model_dir / "logs")

            # create a validation dataset from 10% of the training dataset
            train_dataset, eval_dataset = random_split(
                train_dataset, [len(train_dataset) - len(train_dataset) // 10,
                                len(train_dataset) // 10])

            trainer = transformers.Trainer(
                args=training_args,
                model=model,
                callbacks=callbacks,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                **trainer_config)
    
            trainer.train()

            all_test_evaluations.append(trainer.evaluate(test_dataset))

            trainer.model.return_logits = True
            test_output = trainer.predict(test_dataset)
            trainer.model.return_logits = False

            assert test_output.predictions is not None
            assert test_output.label_ids is not None
            model_logits: np.ndarray = test_output.predictions[0]
            predicted_label_idx = test_output.predictions[1]

            model_probabilities = torch.nn.functional.softmax(torch.tensor(model_logits), dim=1).numpy()
            model_entropy = -np.sum(model_probabilities * np.log(model_probabilities), axis=1)
            # probability of top label
            predicted_probability = model_probabilities[np.arange(model_probabilities.shape[0]), predicted_label_idx]
            # probability of GT label
            gt_label_probability = model_probabilities[np.arange(model_probabilities.shape[0]), test_output.label_ids[0]]

            all_test_outputs.append({
                "predicted_label_idx": predicted_label_idx,
                "predicted_probability": predicted_probability,

                "gt_label_probability": gt_label_probability,

                "entropy": model_entropy,

                "label_idx": test_output.label_ids[0],
                "label_instance_idx": test_output.label_ids[2],
                "example_idx": test_output.label_ids[1],

                "item_idx": test_output.label_ids[3],
                "onset_frame_idx": test_output.label_ids[4],
                "offset_frame_idx": test_output.label_ids[5],
            })

        predictions_df = pd.concat([pd.DataFrame(e) for e in all_test_outputs], ignore_index=True) \
            .sort_values("example_idx")
        predictions_df["label"] = predictions_df.label_idx.map(dict(enumerate(all_labels)))
        predictions_df["predicted_label"] = predictions_df.predicted_label_idx.map(dict(enumerate(all_labels)))
        predictions_df["correct"] = predictions_df.label_idx == predictions_df.predicted_label_idx
        predictions_df.to_parquet(output_dir / f"predictions-frame_{frame_idx}.parquet", index=False)