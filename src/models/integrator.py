from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
from pathlib import Path
import random
from typing import Callable, Optional, Iterator, Literal, TypeAlias

from datasets import Dataset, IterableDataset
from IsoScore.IsoScore import IsoScore
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction
from transformers.file_utils import ModelOutput
from tqdm.auto import tqdm, trange

from src.datasets.speech_equivalence import SpeechHiddenStateDataset, SpeechEquivalenceDataset


L = logging.getLogger(__name__)


class RNNModel(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, type="lstm"):
        super(RNNModel, self).__init__()
        if num_layers == 0:
            assert hidden_dim == input_dim, f"Hidden dim {hidden_dim} must match input dim {input_dim} if num_layers is 0"
            self.rnn = nn.Identity()
        else:
            fn = nn.LSTM if type == "lstm" else nn.RNN
            self.rnn = fn(num_layers=num_layers, input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc: nn.Module = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        lengths = lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        if isinstance(self.rnn, nn.Identity):
            packed_output = packed_input
        else:
            packed_output, _ = self.rnn(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output
    

class ContrastiveEmbeddingObjective(nn.Module):
    def __init__(self, tau=0.1, batch_soft_negatives=False,
                 regularization=None):
        super(ContrastiveEmbeddingObjective, self).__init__()
        self.tau = tau
        self.batch_soft_negatives = batch_soft_negatives

        regularization_type = regularization[0] if isinstance(regularization, (tuple, list)) else regularization
        if regularization_type not in [None, "covariance", "spectral_norm"]:
            raise ValueError(f"Unknown regularization {regularization}")
        self.regularization_type = regularization_type
        self.regularization = regularization

    def forward(self, embeddings, pos_embeddings, neg_embeddings,
                reduction="mean",
                embeddings_class=None,
                return_soft_negatives=False):
        pos_dist = F.cosine_similarity(embeddings, pos_embeddings, dim=1)

        if neg_embeddings is None:
            neg_dist = torch.tensor(0.).to(embeddings)
        else:
            neg_dist = F.cosine_similarity(embeddings, neg_embeddings, dim=1)

        pos_loss = -torch.log(torch.exp(pos_dist / self.tau))
        neg_loss = -torch.log(torch.exp(-neg_dist / self.tau))

        if reduction == "mean":
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()

        if self.batch_soft_negatives:
            if embeddings_class is None:
                raise ValueError("Must provide embeddings_class if using batch_soft_negatives")

            # Compute pairwise cosine similarity matrix
            anchors = embeddings
            soft_negatives = embeddings  # TODO could also include hard positives/negatives of other examples
            pairwise_cosine_sim = F.cosine_similarity(anchors.unsqueeze(1), soft_negatives.unsqueeze(0), dim=2)

            # Evaluate upper triangle
            mask = torch.triu(embeddings_class.unsqueeze(1) != embeddings_class.unsqueeze(0), diagonal=1)
            pairwise_cosine_sim = pairwise_cosine_sim[mask]

            soft_neg_loss = -torch.log(torch.exp(-pairwise_cosine_sim / self.tau)).mean()
            # Guard for NaN (will happen if there are no soft negatives)
            if torch.isnan(soft_neg_loss):
                soft_neg_loss = torch.tensor(0.0, device=soft_neg_loss.device)

            neg_loss += soft_neg_loss
        
        train_loss = pos_loss + neg_loss

        ### Regularization

        if self.training:
            regularization_loss = torch.tensor(0.0, device=embeddings.device)
            if self.regularization_type == "covariance":
                scaler = float(self.regularization[1]) if isinstance(self.regularization, (list, tuple)) else 1.0
                if embeddings.shape[0] >= 2 and embeddings.shape[1] >= 2:
                    # Regularize by penalizing divergence of embedding covariance matrix from identity
                    # i.e., encourage axes to be decorrelated
                    embedding_cov = torch.cov(embeddings.T)

                    # Only penalize covariance off diagonal
                    embedding_cov = embedding_cov[~torch.eye(embedding_cov.shape[0], dtype=bool)]
                    target = torch.zeros_like(embedding_cov)

                    # TODO tune the scale of this relative to training loss
                    regularization_loss = scaler * F.mse_loss(embedding_cov, target)

                    if torch.isnan(regularization_loss):
                        import ipdb; ipdb.set_trace()
                
            elif self.regularization_type == "spectral_norm":
                # Do nothing -- this happens outside the loss function, which only has access to embeddings
                pass

            train_loss += regularization_loss

        if return_soft_negatives:
            return train_loss, (soft_negatives, mask, pairwise_cosine_sim)
        return train_loss
    

class ContrastiveEmbeddingHingeObjective(nn.Module):
    # TODO redundancy with ContrastiveEmbeddingObjective
    def __init__(self, margin=0.1, batch_soft_negatives=False,
                 regularization=None, **kwargs):
        super(ContrastiveEmbeddingHingeObjective, self).__init__()
        self.margin = torch.tensor(margin)
        self.batch_soft_negatives = batch_soft_negatives

        regularization_type = regularization[0] if isinstance(regularization, (tuple, list)) else regularization
        if regularization_type not in [None, "covariance", "spectral_norm"]:
            raise ValueError(f"Unknown regularization {regularization}")
        self.regularization_type = regularization_type
        self.regularization = regularization

    def forward(self, embeddings, pos_embeddings, neg_embeddings,
                reduction="mean",
                embeddings_class=None,
                return_soft_negatives=False):
        # torch.autograd.set_detect_anomaly(True)
        if reduction is not None and reduction != "mean":
            raise NotImplementedError()

        pos_sim = F.cosine_similarity(embeddings, pos_embeddings, dim=1)

        assert neg_embeddings is None, "not supported"
        assert self.batch_soft_negatives

        if embeddings_class is None:
            raise ValueError("Must provide embeddings_class if using batch_soft_negatives")

        # Compute pairwise cosine similarity matrix
        anchors = embeddings
        soft_negatives = embeddings  # TODO could also include hard positives/negatives of other examples
        neg_sim = F.cosine_similarity(anchors.unsqueeze(1), soft_negatives.unsqueeze(0), dim=2)

        # prepare to zero out the diagonal
        mask = embeddings_class.unsqueeze(1) != embeddings_class.unsqueeze(0)

        pairwise_sim = pos_sim[None, :] - neg_sim
        pairwise_sim = pairwise_sim[mask]

        loss = self.margin - pairwise_sim
        loss = F.relu(loss)
        if reduction == "mean":
            loss = loss.mean()

        ### Regularization

        if self.training and self.regularization_type is not None:
            regularization_loss = torch.tensor(0.0, device=embeddings.device)
            if self.regularization_type == "covariance":
                scaler = float(self.regularization[1]) if isinstance(self.regularization, (list, tuple)) else 1.0
                if embeddings.shape[0] >= 2 and embeddings.shape[1] >= 2:
                    # Regularize by penalizing divergence of embedding covariance matrix from identity
                    # i.e., encourage axes to be decorrelated
                    embedding_cov = torch.cov(embeddings.T)

                    # Only penalize covariance off diagonal
                    embedding_cov = embedding_cov[~torch.eye(embedding_cov.shape[0], dtype=bool)]
                    target = torch.zeros_like(embedding_cov)

                    # TODO tune the scale of this relative to training loss
                    regularization_loss = scaler * F.mse_loss(embedding_cov, target)

                    if torch.isnan(regularization_loss):
                        import ipdb; ipdb.set_trace()
                
            elif self.regularization_type == "spectral_norm":
                # Do nothing -- this happens outside the loss function, which only has access to embeddings
                pass

            loss += regularization_loss

        if return_soft_negatives:
            return loss, (soft_negatives, mask, pairwise_sim)
        return loss



ContrastiveLossSpec: TypeAlias = Literal["ratio", "hinge"]


@dataclass
class ContrastiveEmbeddingModelConfig(PretrainedConfig):
    base_model_ref: str = "facebook/wav2vec2-base"
    base_model_layer: int = 6

    # equivalence-classing config
    equivalence_classer: str = "phoneme_within_word_prefix"
    num_classes: int = 5

    # NN config
    max_length: int = 20
    input_dim: int = 4
    num_layers: int = 1
    hidden_dim: int = 256
    output_dim: int = 4

    # Loss config
    loss_form: ContrastiveLossSpec = "ratio"
    # only relevant for ratio objective
    tau: float = 0.1
    # only relevant for hinge objective
    margin: float = 0.1

    in_batch_soft_negatives: bool = True
    """
    If True, use all other examples in the batch as soft negatives unless they have
    the same class. If False, only use the hard negative example.
    """

    regularization: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.base_model_ref == dataset.model_name and \
               self.input_dim == dataset.hidden_size


@dataclass
class ContrastiveEmbeddingModelOutput(ModelOutput):
    loss: torch.Tensor

    embeddings: Optional[torch.Tensor] = None
    embeddings_hard_positive: Optional[torch.Tensor] = None
    embeddings_hard_negative: Optional[torch.Tensor] = None

    embeddings_soft_negative: Optional[torch.Tensor] = None
    soft_negative_mask: Optional[torch.Tensor] = None
    soft_negative_pairwise_sim: Optional[torch.Tensor] = None


class ContrastiveEmbeddingModel(PreTrainedModel):
    config_class = ContrastiveEmbeddingModelConfig
    main_input_name = "example"

    def __init__(self, config):
        super().__init__(config)
        self.rnn = RNNModel(config.num_layers,
                            config.input_dim,
                            config.hidden_dim,
                            config.output_dim)
        
        if self.config.regularization == "spectral_norm":
            # Register spectral norm parameterization on RNN output layer.
            self.rnn.fc = spectral_norm(self.rnn.fc)

        self.fc = None

    def __setstate__(self, state):
        if "fc" not in state:
            state["fc"] = None
        super().__setstate__(state)
        
    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.config.is_compatible_with(dataset)

    def forward(self, example, example_length, pos, pos_length,
                neg=None, neg_length=None,
                loss_reduction="mean",
                return_loss=True, return_embeddings=True,
                in_batch_soft_negatives=None,
                **kwargs):
        in_batch_soft_negatives = (in_batch_soft_negatives if in_batch_soft_negatives is not None
                                   else self.config.in_batch_soft_negatives)

        if neg is None and not in_batch_soft_negatives:
            raise ValueError("Must provide negative examples if not using in_batch_soft_negatives")

        embeddings, pos_embeddings, neg_embeddings = self.compute_batch_embeddings(
            example, example_length, pos, pos_length, neg, neg_length)
        
        if self.fc is not None:
            embeddings = self.fc(embeddings)

        loss_kwargs = dict(batch_soft_negatives=in_batch_soft_negatives,
                           regularization=self.config.regularization)
        if self.config.loss_form == "ratio":
            loss_fn = ContrastiveEmbeddingObjective(
                tau = self.config.tau, **loss_kwargs)
        elif self.config.loss_form == "hinge":
            loss_fn = ContrastiveEmbeddingHingeObjective(
                margin=self.config.margin, **loss_kwargs)
        else:
            raise ValueError(f"Unknown loss form {self.config.loss_form}")

        ret = loss_fn(embeddings, pos_embeddings, neg_embeddings,
                      reduction=loss_reduction,
                      embeddings_class=kwargs.get("example_class"),
                      return_soft_negatives=in_batch_soft_negatives)
        
        return self._prepare_output(ret, embeddings, pos_embeddings, neg_embeddings,
                                    has_soft_negatives=in_batch_soft_negatives,
                                    return_embeddings=return_embeddings)

    def compute_embeddings(self, example, example_length, return_all_states=False):
        embeddings = self.rnn(example, example_length)

        if return_all_states:
            # Mask states beyond the length of each example.
            max_batch_length = example_length.max().item()
            mask = torch.arange(max_batch_length).expand(example.shape[0], -1).to(example.device) >= example_length.unsqueeze(1)
            embeddings[mask] = 0
            return embeddings
        else:
            # Gather final embedding of each sequence
            embeddings = torch.gather(embeddings, 1, (example_length - 1).reshape(-1, 1, 1).expand(-1, 1, embeddings.shape[-1])).squeeze(1)
            return embeddings
        
    def compute_batch_embeddings(self, example, example_length, pos, pos_length,
                                 neg=None, neg_length=None):
        return self.compute_embeddings(example, example_length), \
                self.compute_embeddings(pos, pos_length), \
                self.compute_embeddings(neg, neg_length) if neg is not None else None
    
    def _prepare_output(self, loss_ret, embeddings, pos_embeddings, neg_embeddings,
                        has_soft_negatives=False,
                        return_embeddings=True) -> ContrastiveEmbeddingModelOutput:
        if has_soft_negatives:
            loss = loss_ret[0]
            soft_neg_embeddings, mask, pairwise_sim = loss_ret[1]
        else:
            loss = loss_ret
            soft_neg_embeddings, mask, pairwise_sim = None, None, None

        if not return_embeddings:
            return ContrastiveEmbeddingModelOutput(loss=loss)
        else:
            return ContrastiveEmbeddingModelOutput(
                loss=loss,
                embeddings=embeddings,
                embeddings_hard_positive=pos_embeddings,
                embeddings_hard_negative=neg_embeddings,
                embeddings_soft_negative=soft_neg_embeddings,
                soft_negative_mask=mask,
                soft_negative_pairwise_sim=pairwise_sim
            )


def get_sequence(F, start_index, end_index, max_length, layer=0):
    if end_index - start_index + 1 > max_length:
        start_index = end_index - max_length + 1
    sequence = F[start_index:end_index + 1]

    if F.ndim == 3:
        # index into layer dimension
        sequence = sequence[:, layer]

    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence)
    
    # Pad on right if necessary
    if len(sequence) < max_length:
        pad_size = max_length - len(sequence)
        padding = torch.zeros(pad_size, F.shape[-1]).to(sequence)
        sequence = torch.cat((sequence, padding), dim=0)
    
    return sequence


def iter_dataset(equiv_dataset_path: str,
                 hidden_states_path: str,
                 max_length: int,
                 num_examples: Optional[int] = None,
                 layer: Optional[int] = None,
                 select_idxs: Optional[list[int]] = None,
                 smoke_test=False,
                 infinite=True) -> Iterator[dict]:
    # Implementation note: because this function is invoked from the Datasets generator
    # pipeline, we pass paths rather than objects so that the relevant inputs don't get
    # needlessly re-serialized every time this function is used.
    equiv_dataset = torch.load(equiv_dataset_path)
    hidden_state_dataset = SpeechHiddenStateDataset.from_hdf5(hidden_states_path)

    class_to_frames = None
    if smoke_test:
        equiv_dataset.Q = equiv_dataset.Q[:10000]
        if select_idxs is not None:
            select_idxs = select_idxs[select_idxs < equiv_dataset.Q.shape[0]]
        equiv_dataset.Q = torch.clamp(equiv_dataset.Q, min=-1, max=5)
        class_to_frames = {cl: [f for f in equiv_dataset.class_to_frames[cl] if f < equiv_dataset.Q.shape[0]]
                           for cl in range(6)}

    if layer is None:
        if hidden_state_dataset.num_layers > 1:
            raise ValueError("Must specify layer if there are multiple layers")
        layer = 0
    F = hidden_state_dataset.states

    lengths = torch.minimum(equiv_dataset.lengths, torch.tensor(max_length))

    if select_idxs is not None:
        assert (equiv_dataset.Q[select_idxs] != -1).all()
        non_null_frames = torch.tensor(select_idxs)
        select_idxs = np.array(select_idxs)

        with equiv_dataset.modify_Q_ctx():
            equiv_dataset.Q[np.setdiff1d(np.arange(len(equiv_dataset.Q)), select_idxs)] = -1
    else:
        non_null_frames = (equiv_dataset.Q != -1).nonzero(as_tuple=True)[0]
        if num_examples is not None:
            non_null_frames = np.random.choice(non_null_frames.numpy(), num_examples, replace=False)

    # load class_to_frames now that dataset subsetting is done
    if class_to_frames is None:
        class_to_frames = equiv_dataset.class_to_frames

    # infinite generation
    while True:
        for i in non_null_frames:
            # Sanity checks
            assert lengths[i] > 0

            pos_indices = class_to_frames[equiv_dataset.Q[i].item()]

            if len(pos_indices) > 1:
                # get non-identical positive example
                pos_idx = i.item()
                while pos_idx == i.item():
                    pos_idx = random.choice(pos_indices)

                neg_idx = None  # random.choice(neg_indices)

                # Extract sequences
                example_seq = get_sequence(F, equiv_dataset.S[i], i, max_length, layer=layer)
                pos_seq = get_sequence(F, equiv_dataset.S[pos_idx], pos_idx, max_length, layer=layer)
                neg_seq = None  # get_sequence(F, equiv_dataset.S[neg_idx], neg_idx, max_length)

                # Sanity checks
                assert lengths[pos_idx] > 0
                # assert lengths[neg_idx] > 0
                assert equiv_dataset.Q[i] != -1
                assert equiv_dataset.Q[pos_idx] != -1
                # assert equiv_dataset.Q[neg_idx] != -1

                yield {
                    "example": example_seq,
                    "example_idx": i,
                    "example_class": equiv_dataset.Q[i],
                    "example_length": lengths[i],

                    "pos": pos_seq,
                    "pos_idx": pos_idx,
                    "pos_class": equiv_dataset.Q[pos_idx],
                    "pos_length": lengths[pos_idx],

                    "neg": neg_seq,
                    "neg_idx": neg_idx,
                    "neg_class": None,  # equiv_dataset.Q[neg_idx],
                    "neg_length": None,  # lengths[neg_idx],
                }

        if not infinite:
            break


def prepare_dataset(equiv_dataset: SpeechEquivalenceDataset,
                    equiv_dataset_path: str,
                    hidden_states_path: str,
                    max_length: int,
                    layer: Optional[int] = None,
                    eval_fraction=0.1,
                    max_eval_size=10000,
                    **kwargs) -> tuple[int, IterableDataset, Dataset]:
    """
    Prepare a negative-sampling dataset for contrastive embedding learning.

    Returns train and test split datasets.
    """

    all_idxs = (equiv_dataset.Q != -1).nonzero(as_tuple=True)[0].numpy()
    all_idxs = np.random.permutation(all_idxs)
    
    eval_num_samples = int(eval_fraction * len(all_idxs))
    if max_eval_size is not None and eval_num_samples > max_eval_size:
        L.info(f"Reducing eval set size from {eval_num_samples} to {max_eval_size}")
        eval_num_samples = max_eval_size

    test_idxs, train_idxs = all_idxs[:eval_num_samples], all_idxs[eval_num_samples:]

    dataset_kwargs = {
        "equiv_dataset_path": equiv_dataset_path,
        "hidden_states_path": hidden_states_path,
        "max_length": max_length,
        "layer": layer,
        **kwargs
    }

    train_dataset = IterableDataset.from_generator(
        iter_dataset, gen_kwargs={**dataset_kwargs, "select_idxs": train_idxs,
                                  "infinite": True})
    test_dataset: Dataset = Dataset.from_generator(
        iter_dataset, keep_in_memory=True,
        gen_kwargs={**dataset_kwargs, "select_idxs": test_idxs,
                    "infinite": False})

    return len(all_idxs), train_dataset, test_dataset


def compute_embeddings(model: ContrastiveEmbeddingModel,
                       equiv_dataset: SpeechEquivalenceDataset,
                       hidden_state_dataset: SpeechHiddenStateDataset,
                       batch_size=16,
                       max_num_frames=None,
                       device=None) -> torch.Tensor:
    """
    Compute integrator embeddings for a given model on a speech
    equivalence classing dataset.
    """
    assert model.is_compatible_with(hidden_state_dataset)
    if device is not None:
        model = model.to(device)
    device = model.device

    model_representations = []

    F = hidden_state_dataset.states
    
    lengths = equiv_dataset.lengths.to(device)

    max_num_frames = max_num_frames or hidden_state_dataset.num_frames
    for batch_start in trange(0, max_num_frames, batch_size):
        batch_idxs = torch.arange(batch_start, min(batch_start + batch_size, hidden_state_dataset.num_frames))
        batch = torch.stack([get_sequence(F, equiv_dataset.S[idx], idx, model.config.max_length, layer=0)
                             for idx in batch_idxs])
        batch = batch.to(device)
        
        lengths_batch = torch.minimum(lengths[batch_idxs], torch.tensor(model.config.max_length))
        # HACK
        lengths_batch[lengths_batch <= 0] = 1

        with torch.no_grad():
            model_representations.append(model.compute_embeddings(batch, lengths_batch))

    return torch.cat(model_representations, dim=0)


def load_or_compute_embeddings(model, equiv_dataset, model_dir, equiv_dataset_path,
                               embedding_cache_dir="out/embedding_cache", **kwargs):
    embedding_cache_path = Path(embedding_cache_dir) / f"{model_dir.replace('/', '-')}-{equiv_dataset_path.replace('/', '-')}.npy"
    
    if Path(embedding_cache_path).exists():
        model_representations = np.load(embedding_cache_path)
    else:
        model_representations = compute_embeddings(model, equiv_dataset,
                                                   **kwargs)
        model_representations = model_representations.numpy()
        np.save(embedding_cache_path, model_representations)
    return model_representations


def compute_embedding_loss(embeddings, pos_embeddings, neg_embeddings, tau=0.1):
    pos_dist = cdist(embeddings, pos_embeddings, metric="cosine")
    neg_dist = cdist(embeddings, neg_embeddings, metric="cosine")

    pos_loss = -np.log(np.exp(pos_dist / tau))
    neg_loss = -np.log(np.exp(-neg_dist / tau))

    return pos_loss.mean() + neg_loss.mean()


def compute_embedding_alignment(embeddings, pos_embeddings, metric="euclidean"):
    """
    Compute average Euclidean/cosine distance between embeddings and their positive anchors.
    """
    if metric == "cosine":
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        pos_embeddings /= np.linalg.norm(pos_embeddings, ord=2, axis=1, keepdims=True)
        return 1 - (embeddings * pos_embeddings).sum(axis=1).mean()
    elif metric == "euclidean":
        return np.linalg.norm(embeddings - pos_embeddings, ord=2, axis=1).mean()
    else:
        raise ValueError(f"Unknown metric {metric}")


def compute_embedding_uniformity(embeddings: np.ndarray, metric="euclidean"):
    """
    Compute uniformity a la Wang & Isola (2020)
    """
    distances = pdist(embeddings, metric=metric)
    return distances.mean()


def compute_embedding_axis_correlation(embeddings: np.ndarray):
    corrs = np.corrcoef(embeddings.T)
    return np.abs(corrs[np.triu_indices_from(corrs, k=1)]).mean()


def compute_mean_average_precision(embeddings: np.ndarray, classes: np.ndarray):
    """
    estimate classification performance by learning a classifier and calculating mean
    average precision
    """
    # Binarize the labels for the mAP calculation
    lb = LabelBinarizer()
    binarized_labels = lb.fit_transform(classes)

    # Train a simple logistic regression classifier on the embeddings
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(embeddings, classes)

    # Get the classifier's prediction probabilities
    pred_probs = classifier.predict_proba(embeddings)

    # Calculate the mean average precision (mAP)
    map_score = average_precision_score(binarized_labels, pred_probs, average="macro")

    return map_score


def compute_metrics(p: EvalPrediction, model_config: ContrastiveEmbeddingModelConfig):
    hard_negative_embeddings, soft_negative_embeddings = None, None
    soft_negative_distances = None
    if len(p.predictions) == 3:
        # hard negatives, no soft negatives
        embeddings, hard_positive_embeddings, hard_negative_embeddings = p.predictions
    elif len(p.predictions) == 5:
        # soft negatives, no hard negatives
        embeddings, hard_positive_embeddings = p.predictions[:2]
        soft_negative_embeddings, _, soft_negative_distances = p.predictions[2:]
    elif len(p.predictions) == 6:
        # hard and soft negatives
        embeddings, hard_positive_embeddings, hard_negative_embeddings = p.predictions[:3]
        soft_negative_embeddings, _, soft_negative_distances = p.predictions[3:]

    example_idxs, example_classes = p.label_ids
    assert embeddings.shape[0] == example_classes.shape[0]

    eval_soft_loss, eval_hard_loss = None, None
    if soft_negative_distances is not None:
        eval_soft_loss = -np.log(np.exp(-soft_negative_distances / 0.1)).mean()
    if hard_negative_embeddings is not None:
        eval_hard_loss = compute_embedding_loss(embeddings, hard_positive_embeddings, hard_negative_embeddings)
    assert eval_hard_loss is not None or eval_soft_loss is not None

    return {
        "eval_loss": eval_hard_loss if eval_hard_loss is not None else eval_soft_loss,
        "eval_soft_loss": eval_soft_loss,
        "eval_embedding_norm": np.linalg.norm(embeddings, ord=2, axis=1).mean(),
        "eval_embedding_alignment": compute_embedding_alignment(embeddings, hard_positive_embeddings, metric="euclidean"),
        "eval_embedding_alignment_cosine": compute_embedding_alignment(embeddings, hard_positive_embeddings, metric="cosine"),
        "eval_embedding_uniformity": compute_embedding_uniformity(embeddings),
        "eval_embedding_corr": compute_embedding_axis_correlation(embeddings),
        "eval_embedding_isoscore": IsoScore(embeddings),
        "eval_mAP": compute_mean_average_precision(embeddings, example_classes),
    }


def compute_metrics_hinge(p: EvalPrediction, model_config: ContrastiveEmbeddingModelConfig):
    # only designed for soft negative case right now
    assert model_config.in_batch_soft_negatives
    embeddings, embeddings_hard_positive, _, _, soft_negative_sims = p.predictions

    assert model_config.margin is not None

    # pos_sim = F.cosine_similarity(torch.tensor(embeddings), torch.tensor(embeddings_hard_positive), dim=1)
    # neg_sim = F.cosine_similarity(torch.tensor(embeddings), torch.tensor(embeddings_soft_negative), dim=1)

    eval_loss = (model_config.margin - soft_negative_sims).clip(0, None).mean()

    example_idxs, example_classes = p.label_ids
    assert embeddings.shape[0] == example_classes.shape[0]

    return {
        "eval_loss": eval_loss,
        "eval_embedding_norm": np.linalg.norm(embeddings, ord=2, axis=1).mean(),
        "eval_embedding_alignment": compute_embedding_alignment(embeddings, embeddings_hard_positive, metric="euclidean"),
        "eval_embedding_alignment_cosine": compute_embedding_alignment(embeddings, embeddings_hard_positive, metric="cosine"),
        "eval_embedding_uniformity": compute_embedding_uniformity(embeddings),
        "eval_embedding_corr": compute_embedding_axis_correlation(embeddings),
        "eval_embedding_isoscore": IsoScore(embeddings),
        "eval_mAP": compute_mean_average_precision(embeddings, example_classes),
    }


def compute_metrics_classification(p: EvalPrediction, model_config: ContrastiveEmbeddingModelConfig):
    example_idx, example_class = p.label_ids
    logits, embeddings = p.predictions
    preds = np.argmax(logits, axis=1)
    return {
        "eval_accuracy": (preds == example_class).mean(),
        "eval_mAP": compute_mean_average_precision(embeddings, example_class),
    }


COMPUTE_METRICS: dict[ContrastiveLossSpec, Callable[[EvalPrediction, ContrastiveEmbeddingModelConfig], dict]] = {
    "ratio": compute_metrics,
    "hinge": compute_metrics_hinge,
}