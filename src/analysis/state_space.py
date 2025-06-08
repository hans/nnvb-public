"""
State space analysis tools for integrator models.
"""

from copy import deepcopy
from functools import cached_property, wraps
from dataclasses import dataclass
import logging
from typing import Literal, Optional, Union, Any, Callable, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.datasets.speech_equivalence import SpeechHiddenStateDataset, SpeechEquivalenceDataset


L = logging.getLogger(__name__)


@dataclass
class StateSpaceAnalysisSpec:

    # Number of frames in the dataset associated with this state space spec.
    # Used for validation.
    total_num_frames: int

    labels: list[str]

    # Analyze K categories of N state space trajectories.
    # Tuples are start and end indices, inclusive.
    target_frame_spans: list[list[tuple[int, int]]]

    # Optional representation of frame subdivisions at lower levels of representation.
    # For example, a word-level state space trajectory may retain information here
    # about phoneme-level subdivisions.
    # 
    # DataFrame with index levels (label, instance_idx, level)
    # and columns (description, frame_idx, onset_frame_idx, offset_frame_idx).
    # This refers to the frame span `target_frame_spans[labels.index(label)][instance_idx]`
    cuts: Optional[pd.DataFrame] = None

    def __post_init__(self):
        assert len(self.target_frame_spans) == len(self.labels)

        if self.cuts is not None:
            assert self.cuts.index.names == ["label", "instance_idx", "level"]
            assert set(self.cuts.columns) >= {"description", "onset_frame_idx", "offset_frame_idx"}

            assert set(self.cuts.index.get_level_values("label")) <= set(self.labels)

            assert (self.cuts.onset_frame_idx < self.total_num_frames).all()
            assert (self.cuts.offset_frame_idx < self.total_num_frames).all()

            # check consistency of cuts + target frame spans by merge-and-compare. faster than a for loop! :)
            cuts_validity_check = pd.merge(self.cuts.reset_index("level", drop=True), self.target_frame_spans_df,
                                           left_index=True, right_on=["label", "instance_idx"])
            assert (cuts_validity_check.onset_frame_idx >= cuts_validity_check.start_frame).all()
            assert (cuts_validity_check.offset_frame_idx <= cuts_validity_check.end_frame).all()

    def to_hdf5(self, path: str, key=None):
        with h5py.File(path, "a") as f:
            group = f
            if key is not None:
                group = f.create_group(key)

            group.attrs["total_num_frames"] = self.total_num_frames

            # if we have non-string labels, serialize with repr and annotate
            labels_are_repr = False
            labels = self.labels
            if any(not isinstance(label, str) for label in self.labels):
                labels_are_repr = True
                labels = [repr(label) for label in self.labels]

            group["labels"] = [label.encode("utf-8") for label in labels]
            group.attrs["labels_are_repr"] = labels_are_repr

            # flatten target_frame_spans, retaining original indices
            target_frame_spans = np.concatenate([np.array(spans_i) for spans_i in self.target_frame_spans])
            target_frame_span_boundaries = np.cumsum([len(spans_i) for spans_i in self.target_frame_spans])
            group.create_dataset("target_frame_spans", data=target_frame_spans)
            group.create_dataset("target_frame_span_boundaries", data=target_frame_span_boundaries)

        if self.cuts is not None:
            cuts_key = "cuts" if key is None else f"{key}/cuts"
            self.cuts.to_hdf(path, key=cuts_key, mode="a")

    @classmethod
    def from_hdf5(cls, path: str, key=None):
        with h5py.File(path, "r") as f:
            group = f
            if key is not None:
                group = f[key]

            total_num_frames = group.attrs["total_num_frames"]
            labels = [label.decode("utf-8") for label in group["labels"]]

            if group.attrs.get("labels_are_repr", False):
                labels = [eval(label) for label in labels]

            target_frame_spans = group["target_frame_spans"][()]
            target_frame_span_boundaries = group["target_frame_span_boundaries"][()]
            target_frame_spans = np.split(target_frame_spans, target_frame_span_boundaries[:-1])

            has_cuts = "cuts" in group

        cuts = None
        if has_cuts:
            cuts = pd.read_hdf(path, key="cuts" if key is None else f"{key}/cuts")

            # `nan` will get re-coded as np.nan; fix this
            cuts = cuts.reset_index("label")
            cuts["label"] = cuts.label.fillna("nan")
            cuts = cuts.set_index("label", append=True)
            cuts = cuts.reorder_levels(["label", "instance_idx", "level"])

        return cls(total_num_frames=total_num_frames, labels=labels,
                    target_frame_spans=target_frame_spans, cuts=cuts)

    @property
    def target_frame_spans_df(self) -> pd.DataFrame:
        """
        Return a dataframe representation of target frame spans. The keys `label` and `instance_idx`
        are comparable to the keys in `cuts`.
        """
        return pd.DataFrame([
            (label, instance_idx, start, end)
            for label, frame_spans in zip(self.labels, self.target_frame_spans)
            for instance_idx, (start, end) in enumerate(frame_spans)
        ], columns=["label", "instance_idx", "start_frame", "end_frame"])

    @property
    def label_counts(self):
        return pd.Series([len(spans) for spans in self.target_frame_spans], index=self.labels)

    def is_compatible_with(self, dataset: Union[SpeechHiddenStateDataset, np.ndarray]) -> bool:
        if isinstance(dataset, SpeechHiddenStateDataset):
            return self.total_num_frames == dataset.num_frames
        else:
            return self.total_num_frames == dataset.shape[0]
    
    def drop_labels(self, drop_idxs=None, drop_names=None):
        if drop_idxs is None and drop_names is None:
            raise ValueError("Must provide either drop_idxs or drop_names")
        
        if drop_idxs is None:
            drop_idxs = [i for i, label in enumerate(self.labels) if label in drop_names]
        
        labels = [label for i, label in enumerate(self.labels) if i not in drop_idxs]
        target_frame_spans = [span for i, span in enumerate(self.target_frame_spans) if i not in drop_idxs]

        new_cuts = None
        if self.cuts is not None:
            mask = self.cuts.index.get_level_values("label").isin(labels)
            new_cuts = self.cuts.loc[mask]

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=labels,
            target_frame_spans=target_frame_spans,
            cuts=new_cuts,
        )
    
    def subsample_instances(self, max_instances_per_label: int, random=True):
        """
        Return a copy of the current state space analysis spec with at most
        `max_instances_per_label` instances per label.
        """
        new_labels = self.labels
        converting_labels = False
        if any(isinstance(label, tuple) for label in self.labels):
            # concat will fail if labels are tuples; pandas will try to turn this into
            # a multiindex.
            # HACK for now is to turn into a string representation.
            # this may produce inconsistency
            converting_labels = True
            L.warning("Subsampling instances with tuple labels; this may produce inconsistent results")
            new_labels = [" ".join(label) for label in self.labels]

        new_target_frame_spans, new_cuts = [], {}
        for label, frame_spans in zip(self.labels, self.target_frame_spans):
            if len(frame_spans) <= max_instances_per_label:
                keep_idxs = np.arange(len(frame_spans))
            else:
                if random:
                    keep_idxs = np.random.choice(len(frame_spans), max_instances_per_label, replace=False)
                else:
                    keep_idxs = np.arange(max_instances_per_label)

            new_target_frame_spans.append([frame_spans[i] for i in keep_idxs])
            if self.cuts is not None:
                new_cuts_i = self.cuts.loc[label]
                new_cuts_i = new_cuts_i[new_cuts_i.index.get_level_values("instance_idx").isin(keep_idxs)]

                # Relabel instance_idx
                new_cuts_i = new_cuts_i.reset_index("instance_idx")
                new_cuts_i["instance_idx"] = new_cuts_i.instance_idx.map({old_idx: new_idx for new_idx, old_idx in enumerate(keep_idxs)})
                new_cuts_i = new_cuts_i.set_index("instance_idx", append=True).reorder_levels(["instance_idx", "level"])
                new_cuts[label] = new_cuts_i

        if self.cuts is not None:
            if converting_labels:
                new_cuts = pd.concat(new_cuts.values(), keys=[" ".join(label) for label in new_cuts.keys()],
                                     names=["label"]).sort_index()
            else:
                new_cuts = pd.concat(new_cuts, names=["label"]).sort_index()
        else:
            new_cuts = None

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=new_labels,
            target_frame_spans=new_target_frame_spans,
            cuts=new_cuts,
        )

    def groupby(self, grouper) -> Iterable[tuple[Any, "StateSpaceAnalysisSpec"]]:
        if self.cuts is None:
            raise ValueError("Cannot groupby without cuts")

        for group_key, group_df in self.cuts.groupby(grouper):
            new_target_frame_spans = []
            new_labels = []
            new_cut_idxs = []
            for label, label_df in group_df.groupby("label"):
                label_idx = self.labels.index(label)
                new_labels.append(label)
                new_label_spans = []
                for instance_idx, instance_df in label_df.groupby("instance_idx"):
                    new_label_spans.append(self.target_frame_spans[label_idx][instance_idx])
                    new_cut_idxs.append((label, instance_idx))
                new_target_frame_spans.append(new_label_spans)

            cut_indexer = pd.DataFrame(new_cut_idxs, columns=["label", "instance_idx"])
            cut_indexer["new_instance_idx"] = cut_indexer.groupby("label").cumcount()
            new_cuts = pd.merge(self.cuts.reset_index(), cut_indexer, on=["label", "instance_idx"])
            # relabel instance_idx
            new_cuts["instance_idx"] = new_cuts.new_instance_idx
            new_cuts = new_cuts.drop(columns=["new_instance_idx"]).set_index(["label", "instance_idx", "level"])
            
            yield group_key, StateSpaceAnalysisSpec(
                total_num_frames=self.total_num_frames,
                labels=new_labels,
                target_frame_spans=new_target_frame_spans,
                cuts=new_cuts,
            )
    
    def keep_top_k(self, k=100):
        """
        Return a copy of the current state space analysis spec with only the top `k`
        labels by instance count.
        """
        if k >= len(self.labels):
            return deepcopy(self)
        top_k_labels = self.label_counts.sort_values(ascending=False).head(k).index
        return self.drop_labels(drop_names=set(self.labels) - set(top_k_labels))
    
    def keep_min_frequency(self, freq=50):
        """
        Return a copy of the current state space analysis spec with only labels
        that have at least `freq` instances.
        """
        keep_labels = self.label_counts[self.label_counts >= freq].index
        return self.drop_labels(drop_names=set(self.labels) - set(keep_labels))
    
    def expand_by_cut_index(self, cut_level: str) -> "StateSpaceAnalysisSpec":
        """
        Expand the state space analysis spec to include information about
        the given cut index within each class instance.
        """
        if self.cuts is None:
            raise ValueError("No cuts available to expand")

        cuts_df = self.cuts.xs(cut_level, level="level")
        cuts_df["idx_in_level"] = cuts_df.groupby(["label", "instance_idx"]).cumcount()
        new_target_frame_spans = []
        new_labels = []

        for (label, idx_in_level), cuts_group in cuts_df.groupby(["label", "idx_in_level"]):
            new_labels.append((label, idx_in_level))
            new_target_frame_spans.append(list(zip(cuts_group.onset_frame_idx, cuts_group.offset_frame_idx)))

        return StateSpaceAnalysisSpec(
            total_num_frames=self.total_num_frames,
            labels=new_labels,
            target_frame_spans=new_target_frame_spans,
            cuts=None,
        )
    
    @cached_property
    def flat(self) -> np.ndarray:
        """
        Return a "flat" representation indexing into state space trajectories
        by frame index rather than by label and instance index.

        Returns a `total_num_frames` x 4 array, where each row is a reference
        to the start of a state trajectory instance, with columns:
        - start frame index
        - end frame index
        - label index
        - instance index
        """
        flat_references = []
        for i, (label, frame_spans) in enumerate(zip(self.labels, self.target_frame_spans)):
            for j, (start, end) in enumerate(frame_spans):
                flat_references.append((start, end, i, j))

        return np.array(sorted(flat_references))

    def get_trajectories_in_span(self, span_left, span_right) -> np.ndarray:
        """
        Return the state space trajectories that intersect with the given
        frame span (inclusive).
        """
        return self.flat[
            (self.flat[:, 0] <= span_right) & (self.flat[:, 1] >= span_left)
        ]


def prepare_word_trajectory_spec(
        dataset: SpeechEquivalenceDataset,
        target_words: list[tuple[str, ...]],
) -> StateSpaceAnalysisSpec:
    """
    Retrieve a list of frame spans describing a speech perception
    trajectory matching the given target words.
    """

    # We expect the dataset class labels to correspond to words
    assert all(type(label) == tuple for label in dataset.class_labels)

    target_labels = [dataset.class_labels.index(word) for word in target_words]
    frame_bounds = []
    for label_idx in target_labels:
        final_frames = torch.where(dataset.Q == label_idx)[0]
        start_frames = dataset.S[final_frames]
        frame_bounds.append(list(zip(start_frames.numpy(), final_frames.numpy())))

    return StateSpaceAnalysisSpec(
        target_frame_spans=frame_bounds,
        labels=target_words,
        total_num_frames=dataset.hidden_state_dataset.num_frames,
    )


class LabeledStateTrajectory:

    def __init__(self,
                 embeddings: np.ndarray,
                 metadata: pd.DataFrame,
                 cut_names: Optional[list[str]] = None):
        self.embeddings = embeddings
        self.metadata = metadata
        self.cut_names = cut_names or []

        self._check_metadata()

    def _check_metadata(self):
        # assert self.trajectories.shape[0] == self.metadata.shape[0]

        expected_columns = {"label", "label_idx", "instance_idx", "frame_idx",
                            "span_onset_frame_idx", "span_offset_frame_idx",
                            "relative_frame_idx"}
        for cut_name in self.cut_names:
            expected_columns.add(f"{cut_name}_idx")
            expected_columns.add(f"{cut_name}_label")
            expected_columns.add(f"{cut_name}_onset_frame_idx")
        assert set(self.metadata.columns) <= expected_columns

    @staticmethod
    def _make_cuts_metadata(embeddings: np.ndarray, spec: StateSpaceAnalysisSpec,
                            include_cuts: Optional[list[str]] = None) -> pd.DataFrame:
        assert spec.cuts is not None
        if include_cuts is None:
            include_cuts = list(spec.cuts.index.get_level_values("level").unique())
        else:
            assert set(include_cuts) <= set(spec.cuts.index.get_level_values("level").unique())

        frame_series = pd.Series(np.arange(embeddings.shape[0]), name="frame_idx")
        tdf = spec.target_frame_spans_df

        all_cuts_md = []
        for i, level in enumerate(tqdm(include_cuts, leave=False, desc="Preparing metadata")):
            cuts = spec.cuts.xs(level, level="level").copy()
            cuts[f"{level}_idx"] = cuts.groupby(["label", "instance_idx"]).cumcount()
            cuts = cuts.sort_values("onset_frame_idx")

            # include label and instance_idx information from first pass through cuts
            retain_columns = ["onset_frame_idx", "offset_frame_idx", f"{level}_idx", "description"]
            if i == 0:
                cuts = cuts.reset_index()
                retain_columns += ["label", "instance_idx"]

                # Add in start/end of state space span
                cuts = pd.merge(cuts, tdf.rename(columns={"start_frame": "span_onset_frame_idx",
                                                          "end_frame": "span_offset_frame_idx"}),
                                on=["label", "instance_idx"])
                retain_columns += ["span_onset_frame_idx", "span_offset_frame_idx"]
            cuts = cuts[retain_columns]

            cuts_md = pd.merge_asof(frame_series, cuts,
                                    left_on="frame_idx", right_on="onset_frame_idx",
                                    direction="backward")
            
            # Drop rows where the frame is after the cut offset
            cuts_md = cuts_md[cuts_md.frame_idx < cuts_md.offset_frame_idx]
            
            type_spec = {"onset_frame_idx": int,
                         f"{level}_idx": int}
            if i == 0:
                type_spec.update({"instance_idx": int,
                                  "span_onset_frame_idx": int,
                                  "span_offset_frame_idx": int})
            cuts_md = cuts_md.set_index("frame_idx").dropna() \
                .drop(columns="offset_frame_idx") \
                .astype(type_spec) \
                .rename(columns={"onset_frame_idx": f"{level}_onset_frame_idx",
                                 "description": f"{level}_label"})

            all_cuts_md.append(cuts_md)

        ret = all_cuts_md[0]
        for cuts_md in all_cuts_md[1:]:
            ret = pd.merge(ret, cuts_md, left_index=True, right_index=True, how="outer")

        # add final metadata columns
        ret["label_idx"] = ret["label"].map({label: i for i, label in enumerate(spec.labels)})
        ret["relative_frame_idx"] = ret.index - ret[f"span_onset_frame_idx"]

        return ret

    @classmethod
    def from_embeddings(cls,
                        embeddings: np.ndarray,
                        spec: StateSpaceAnalysisSpec,
                        expand_window: Optional[tuple[int, int]] = None,
                        include_cuts: Optional[list[str]] = None,
                        pad: Union[str, float] = "last") -> "LabeledStateTrajectory":
        """
        Prepare a labeled state trajectory from the given embeddings, following
        the given state space representation.
        """
        if include_cuts is not None:
            assert spec.cuts is not None
            assert set(include_cuts) <= set(spec.cuts.index.get_level_values("level"))
        else:
            include_cuts = list(spec.cuts.index.get_level_values("level").unique())

        # traj_data = prepare_state_trajectory(embeddings, spec, expand_window=expand_window, pad=pad)
        metadata = cls._make_cuts_metadata(embeddings, spec, include_cuts=include_cuts)

        return cls(embeddings, metadata, cut_names=include_cuts)

    def get_frames_by_relative_idx(self, relative_idx: int, include_labels=False
                                   ):
        """
        Get embeddings by position relative to word onset.
        
        Args:
            relative_idx: Position relative to word onset, where 0 is the onset frame.
                Negative values are allowed; in this case, returned labels will still
                correspond to the focus word, and not the preceding word from which
                the frames are drawn.
        """
        draw_idx = 0 if relative_idx < 0 else relative_idx
        frame_rows = self.metadata[self.metadata.relative_frame_idx == draw_idx]

        idx_rows = frame_rows.index
        if relative_idx < 0:
            idx_rows += relative_idx

        if include_labels:
            return idx_rows, self.embeddings[idx_rows], frame_rows.label_idx.values
        return idx_rows, self.embeddings[idx_rows]


def prepare_state_trajectory(
        embeddings: Union[np.ndarray, h5py.Dataset],
        spec: StateSpaceAnalysisSpec,
        expand_window: Optional[tuple[int, int]] = None,
        pad: Union[str, float] = "last",
        agg_fn_spec: Optional[Union[str, tuple[str, Any]]] = None,
        agg_fn_dimension: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Prepare the state trajectory for the given dataset and model embeddings.

    If `expand_window` is not None, add `expand_window[0]` frames to the left
    of each trajectory and `expand_window[1]` frames to the right.
    """
    agg_fn = None
    if agg_fn_spec is not None:
        if agg_fn_dimension is None:
            raise ValueError("Must provide agg_fn_dimension when using agg_fn_spec")
    
        agg_fn = _get_agg_fn(agg_fn_spec)

    max_num_frames = max(max(end - start + 1 for start, end in trajectory_spec)
                         for trajectory_spec in spec.target_frame_spans)
    if expand_window is not None:
        max_num_frames += expand_window[0] + expand_window[1]
    ret = []

    if embeddings.ndim == 3:
        assert embeddings.shape[1] == 1
    embedding_dim = embeddings.shape[-1]

    ret_num_timesteps = max_num_frames if agg_fn_dimension is None else agg_fn_dimension

    for i, frame_spec in enumerate(tqdm(spec.target_frame_spans)):
        num_instances = len(frame_spec)

        trajectory_frames = np.zeros((num_instances, ret_num_timesteps, embedding_dim))
        skip_idxs = []
        for j, (start, end) in enumerate(frame_spec):
            if expand_window is not None:
                if start < expand_window[0]:
                    L.warning(f"Skipping instance {j} of label {i} due to insufficient context")
                    skip_idxs.append(j)
                    continue
                start = max(0, start - expand_window[0])
                end = min(spec.total_num_frames - 1, end + expand_window[1])

            embedding_ij = embeddings[start:end + 1]
            
            if embedding_ij.ndim == 3:
                embedding_ij = embedding_ij.squeeze(1)

            if agg_fn is not None:
                embedding_ij = agg_fn(embedding_ij.T, state_space_spec=spec,
                                      label_idx=i).T
                trajectory_frames[j] = embedding_ij
            else:
                trajectory_frames[j, :end - start + 1] = embedding_ij

                # Pad on right
                if pad == "last":
                    pad_value = embeddings[end]
                    if pad_value.ndim == 2:
                        pad_value = pad_value.squeeze(0)
                elif isinstance(pad, str):
                    raise ValueError(f"Invalid pad value {pad}; use `last` or a float")
                else:
                    pad_value = pad

                if end - start + 1 < max_num_frames:
                    trajectory_frames[j, end - start + 1:] = pad_value

        trajectory_frames = np.delete(trajectory_frames, skip_idxs, axis=0)
        ret.append(trajectory_frames)

    return ret


def get_trajectory_lengths(trajectory: list[np.ndarray]) -> np.ndarray:
    return [np.isnan(traj_i[:, :, 0]).any(axis=1) * np.isnan(traj_i[:, :, 0]).argmax(axis=1) + \
            ~np.isnan(traj_i[:, :, 0]).any(axis=1) * traj_i.shape[1]
            for traj_i in trajectory]


def make_simple_agg_fn(fn):
    @wraps(fn)
    def agg_fn(xs, *args, **kwargs):
        return fn(xs)
    return agg_fn

def make_agg_fn_mean_last_k(k):
    def agg_fn(xs, *args, **kwargs):
        nan_onset = np.isnan(xs[:, :, 0]).argmax(axis=1)
        # if there are no nans, set nan_onset to len
        nan_onset[~np.isnan(xs[:, :, 0]).any(axis=1)] = xs.shape[1]
        return np.stack([
            np.mean(xs[i, np.maximum(0, nan_onset[i] - k) : nan_onset[i]], axis=0, keepdims=True)
            for i in range(xs.shape[0])
        ])
    return agg_fn

def make_agg_fn_mean_first_k(k):
    def agg_fn(xs, *args, **kwargs):
        nan_onset = np.isnan(xs[:, :, 0]).argmax(axis=1)
        # if there are no nans, set nan_onset to len
        nan_onset[~np.isnan(xs[:, :, 0]).any(axis=1)] = xs.shape[1]
        return np.stack([
            np.mean(xs[i, 0:min(nan_onset[i],k)], axis=0, keepdims=True)
            for i in range(xs.shape[0])
        ])
    return agg_fn

class AggMeanWithinCut:
    def __init__(self, cut_level: str):
        self.cut_level = cut_level
    def __call__(self, trajectory_i: np.ndarray, state_space_spec: StateSpaceAnalysisSpec,
                 label_idx: int, pad: Union[Literal["last"], float] = np.nan) -> np.ndarray:
        if state_space_spec.cuts is None:
            raise ValueError("No cuts available to aggregate within")
        
        label = state_space_spec.labels[label_idx]
        cuts_df = state_space_spec.cuts.loc[label]
        try:
            cuts_df = cuts_df.xs(self.cut_level, level="level")
        except KeyError:
            raise ValueError(f"Cut level {self.cut_level} not found in cuts")

        assert set(cuts_df.index.get_level_values("instance_idx")) == set(np.arange(len(trajectory_i)))

        max_num_cuts: int = cuts_df.groupby("instance_idx").size().max()  # type: ignore
        new_trajs = np.zeros((len(trajectory_i), max_num_cuts, trajectory_i.shape[2]), dtype=float)
        
        for instance_idx, instance_cuts in cuts_df.groupby("instance_idx"):
            instance_frame_start, _ = state_space_spec.target_frame_spans[label_idx][instance_idx]
            for cut_idx, (_, cut) in enumerate(instance_cuts.iterrows()):
                # get index of cut relative to instance onset frame
                cut_start: int = cut.onset_frame_idx - instance_frame_start
                cut_end: int = cut.offset_frame_idx - instance_frame_start

                new_trajs[instance_idx, cut_idx] = np.mean(trajectory_i[instance_idx, cut_start:cut_end], axis=0, keepdims=True)

            if pad == "last":
                pad_value = new_trajs[instance_idx, cut_idx]
            elif isinstance(pad, str):
                raise ValueError(f"Invalid pad value {pad}; use `last` or a float")
            else:
                pad_value = pad

            if cut_idx < max_num_cuts - 1:
                new_trajs[instance_idx, cut_idx + 1:] = pad_value

        return new_trajs
        

TRAJECTORY_AGG_FNS: dict[str, Callable] = {
    "mean": lambda xs: np.nanmean(xs, axis=1, keepdims=True),
    "max": lambda xs: np.nanmax(xs, axis=1, keepdims=True),
    "last_frame": lambda xs: xs[np.arange(xs.shape[0]), np.isnan(xs[:, :, 0]).argmax(axis=1) - 1][:, None, :],
}
TRAJECTORY_AGG_FNS = {k: make_simple_agg_fn(v) for k, v in TRAJECTORY_AGG_FNS.items()}

TRAJECTORY_META_AGG_FNS: dict[str, Callable] = {
    "mean_last_k": make_agg_fn_mean_last_k,
    "mean_first_k": make_agg_fn_mean_first_k,
    "mean_within_cut": AggMeanWithinCut,
}


def _get_agg_fn(agg_fn_spec: Union[str, tuple[str, Any]]) -> Callable:
    if isinstance(agg_fn_spec, tuple):
        agg_fn_name, agg_fn_args = agg_fn_spec
        agg_fn = TRAJECTORY_META_AGG_FNS[agg_fn_name](agg_fn_args)
    else:
        agg_fn = TRAJECTORY_AGG_FNS[agg_fn_spec]
    return agg_fn


def aggregate_state_trajectory(trajectory: list[np.ndarray],
                               state_space_spec: StateSpaceAnalysisSpec,
                               agg_fn_spec: Union[str, tuple[str, Any]],
                               keepdims=False) -> list[np.ndarray]:
    """
    Aggregate over time in the state trajectories returned by `prepare_state_trajectory`.
    """
    agg_fn = _get_agg_fn(agg_fn_spec)

    ret = [agg_fn(traj, state_space_spec=state_space_spec,
                  label_idx=idx)
           for idx, traj in enumerate(tqdm(trajectory, unit="label", desc="Aggregating", leave=False))]
    if not keepdims and ret[0].shape[1] == 1:
        ret = [traj.squeeze(1) for traj in ret]

    return ret


def flatten_trajectory(trajectory: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a flattened representation of all state space trajectories.

    Returns:
    - all_trajectories: (N, D) array of all state space trajectories
    - all_trajectories_src: (N, 3) array
        describes, for each row of `all_trajectories`, the originating label, instance idx, and frame idx
    """

    all_trajectories = np.concatenate([
        traj_i.reshape(-1, traj_i.shape[-1]) for traj_i in trajectory
    ])
    all_trajectories_src = np.concatenate([
        np.array([(label_idx, instance_idx, frame_idx) for label_idx, traj_i in enumerate(trajectory)
                 for instance_idx, frame_idx in np.ndindex(traj_i.shape[:2])])
    ])
    assert all_trajectories.shape[0] == all_trajectories_src.shape[0]
    assert all_trajectories_src.shape[1] == 3
    assert all_trajectories_src[:, 0].max() == len(trajectory) - 1

    # TODO this assumes NaN padding
    retain_idxs = ~np.isnan(all_trajectories).any(axis=1)
    all_trajectories = all_trajectories[retain_idxs]
    all_trajectories_src = all_trajectories_src[retain_idxs]

    return all_trajectories, all_trajectories_src