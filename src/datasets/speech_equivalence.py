from collections import Counter, defaultdict
from contextlib import contextmanager
from beartype import beartype
from dataclasses import dataclass, replace
from functools import cached_property
import itertools
from pathlib import Path
from typing import TypeAlias, Callable, Any, Hashable, Optional, Union

import datasets
import h5py
from jaxtyping import Float, Int64
import numpy as np
import torch
from torch import Tensor as T
from tqdm.auto import trange, tqdm


# defines the critical logic by which frames are equivalence-classed.
# Functions accept grouped word phonemic annotations, word utterances and
# frame index, and return arbitrary grouping value.
EquivalenceClasser: TypeAlias = Callable[[Any, str, int], Hashable]
equivalence_classers: dict[str, EquivalenceClasser] = {
    "phoneme_within_word_prefix": 
        lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[:i+1]),
    "phoneme": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i]["phone"],
    "next_phoneme": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i+1]["phone"] if i + 1 < len(word_phonemic_detail) else None,
    "phoneme_within_word_suffix": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[i:]),
    "word_suffix": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail[i + 1:]),
    "word": lambda word_phonemic_detail, word_str, i: tuple(phone["phone"] for phone in word_phonemic_detail),
    "word_broad": lambda word_phonemic_detail, word_str, i: word_str,

    "biphone_recon": lambda word_phonemic_detail, word_str, i: (word_phonemic_detail[i-1]["phone"] if i > 0 else "#", word_phonemic_detail[i]["phone"]),
    "biphone_pred": lambda word_phonemic_detail, word_str, i: (word_phonemic_detail[i]["phone"], word_phonemic_detail[i+1]["phone"] if i + 1 < len(word_phonemic_detail) else "#"),

    "phoneme_fixed": lambda word_phonemic_detail, word_str, i: word_phonemic_detail[i]["phone"],

    "syllable": lambda word_phonemic_detail, word_str, i: tuple(word_phonemic_detail[i]["syllable_phones"]) if word_phonemic_detail[i]["syllable_phones"] else None,
}

def syllable_anti_grouper(word_phonemic_detail, i):
    # mismatch: current syllable
    mismatch_key = tuple(word_phonemic_detail[i]["syllable_phones"]) if word_phonemic_detail[i]["syllable_phones"] else None
    if mismatch_key is None:
        return None, None

    # match: preceding syllable
    current_syllable_idx = word_phonemic_detail[i]["syllable_idx"]
    if current_syllable_idx == 0:
        return None, None
    for j in range(i - 1, -1, -1):
        if word_phonemic_detail[j]["syllable_idx"] < current_syllable_idx:
            return tuple(word_phonemic_detail[j]["syllable_phones"]) if word_phonemic_detail[j]["syllable_phones"] else None, None

    return None, None


# Define subequivalence classing logic. Subequivalence classes define the conjunction of 1) a mismatch
# on an equivalence class and 2) a match on a second equivalence class. This can be used to define
# hard negative samples, for example -- phonemes which mismatch but appear in the same context,
# or syllables which mismatch but contain a substantial number of similar segments.
#
# For a given word annotation and phoneme index, return a grouping value (on which a frame should be
# mismatched with other frames) and an anti-grouping value (on which a frame should match with other
# frames).
subequivalence_classers: dict[str, Callable] = {
    "phoneme": lambda word, i: (word[i]["phone"], word[i - 1]["phone"] if i > 0 else "#"),

    # anti-grouping: preceding syllable
    "syllable": syllable_anti_grouper,
}

# Each equivalence classer also defines a function to compute the start of the
# event which is a sufficient statistic for the class.
start_references = {
    "phoneme_within_word_prefix": "word",
    "phoneme": "word",
    "next_phoneme": "word",
    "phoneme_within_word_suffix": "word",
    "word_suffix": "word",
    "word": "word",
    "word_broad": "word",

    "biphone_recon": "word",
    "biphone_pred": "word",

    "phoneme_fixed": "fixed",

    "syllable": "word",
}


@dataclass
class SpeechHiddenStateDataset:
    model_name: str

    # num_frames * num_layers * hidden_size
    # states: Float[T, "num_frames num_layers hidden_size"]
    states: h5py.Dataset

    # ratio of num_frames to original audio num_samples. useful for 
    # aligning frames with annotations in the original audio.
    compression_ratios: dict[int, float]

    # mapping from flattened frame index to (item index, frame index)
    flat_idxs: list[tuple[int, int]]

    _file_handle: Optional[h5py.File] = None

    def __post_init__(self):
        assert self.states.ndim == 3
        assert len(self.flat_idxs) == self.states.shape[0]
        # each item has at least one frame; and we have a compression ratio stored
        # for each item
        assert set(np.unique(np.array(self.flat_idxs)[:, 0])) == set(np.arange(len(self.compression_ratios)))

    def get_layer(self, layer: int) -> Float[T, "num_frames hidden_size"]:
        return torch.tensor(self.states[:, layer, :][()])

    def __repr__(self):
        return f"SpeechHiddenStateDataset({self.model_name}, {self.num_items} items, {self.num_frames} frames, {self.num_layers} layers, {self.states.shape[2]} hidden size)"
    __str__ = __repr__

    def __getitem__(self, slice_) -> "SpeechHiddenStateDataset":
        if not isinstance(slice_, slice):
            raise ValueError()

        new_states = self.states[slice_]
        new_flat_idxs = self.flat_idxs[slice_.start:slice_.stop]

        new_item_idxs = sorted(set(np.array(new_flat_idxs)[:, 0]))
        new_compression_ratios = {idx: self.compression_ratios[idx] for idx in new_item_idxs}

        return replace(self, states=new_states, flat_idxs=new_flat_idxs, compression_ratios=new_compression_ratios)

    def to_hdf5(self, path: str):
        # need contiguous keys in compression_ratios
        assert set(self.compression_ratios.keys()) == set(range(len(self.compression_ratios)))
        compression_ratios = np.array([self.compression_ratios[idx] for idx in range(len(self.compression_ratios))])

        with h5py.File(path, "w") as f:
            f.attrs["model_name"] = self.model_name
            f.create_dataset("states", data=self.states.numpy())
            f.create_dataset("compression_ratios", data=compression_ratios, dtype=np.float32)
            f.create_dataset("flat_idxs", data=self.flat_idxs, dtype=np.int32)
    
    @classmethod
    def from_hdf5(cls, path: Union[str, Path]):
        f = h5py.File(path, "r")
        model_name = f.attrs["model_name"]
        states = f["states"]  # NB not loading into memory

        compression_ratios = f["compression_ratios"][:]
        compression_ratios = {idx: ratio for idx, ratio in enumerate(compression_ratios)}

        flat_idxs = f["flat_idxs"][:]

        return cls(model_name=model_name, states=states,
                   compression_ratios=compression_ratios,
                   flat_idxs=flat_idxs,
                   _file_handle=f)

    @property
    def num_frames(self) -> int:
        return len(self.flat_idxs)
    
    @property
    def num_items(self) -> int:
        return len(self.frames_by_item)
    
    @property
    def num_layers(self) -> int:
        return self.states.shape[1]
    
    @property
    def hidden_size(self) -> int:
        return self.states.shape[2]

    @cached_property
    def frames_by_item(self) -> dict[int, tuple[int, int]]:
        """Mapping from item number to flat idx frame start, end (exclusive)"""
        item_idxs, flat_idx_starts = np.unique(np.array(self.flat_idxs)[:, 0], return_index=True)
        flat_idx_ends = np.concatenate([flat_idx_starts[1:], [len(self.flat_idxs)]])
        return {item: (start, end) for item, start, end in zip(item_idxs, flat_idx_starts, flat_idx_ends)}


@beartype
@dataclass
class SpeechEquivalenceDataset:
    """
    Represents an equivalence classing over a `SpeechHiddenStateDataset`.

    Each frame is annotated by
        1) a class, and
        2) a start frame. An ideal model should be able to aggregrate hidden states from the
           start frame to the current frame and predict this class. In this sense the start
           frame defines a sufficient statistic for the class on an instance level.

    These annotations are represented in the vectors `Q` and `S`, respectively.
    """

    name: str

    Q: Int64[T, "num_frames"]
    S: Int64[T, "num_frames"]

    class_to_frames: dict[int, list[int]]
    """
    For each equivalence class index, a list of frame indices.
    This is a redundant inverse representation of `Q`, which maps from frame index to class.
    """

    class_labels: list[Hashable]
    """
    For each equivalence class, a description of its content.
    """

    def __post_init__(self):
        assert self.Q.shape[0] == self.S.shape[0]

        assert self.Q.max() < len(self.class_labels)
        assert self.Q.min() >= -1

        # If Q is set, then S should be set
        assert (self.Q == -1).logical_or(self.S != -1).all()

        assert set(self.class_to_frames.keys()) == set(range(len(self.class_labels)))
        # check consistency between class_to_frames and Q
        for class_idx, frames in tqdm(self.class_to_frames.items(), desc="Checking consistency"):
            assert (self.Q[frames] == class_idx).all()

    def __repr__(self):
        return f"SpeechEquivalenceDataset({self.name}, {self.num_classes} classes, {self.num_instances} instances)"

    def is_compatible_with(self, dataset: SpeechHiddenStateDataset):
        return self.Q.shape[0] == dataset.num_frames

    @property
    def num_instances(self) -> int:
        return (self.Q != -1).long().sum().item()

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)
    
    @cached_property
    def lengths(self):
        """
        For each frame, the distance between the right edge of this frame and the start of the
        class instance. Integer on [1, num_frames) if this frame is part of a class instance,
        and 0 otherwise.
        """
        lengths = torch.arange(self.S.shape[0]) - self.S + 1
        lengths[self.S == -1] = 0
        return lengths

    @contextmanager
    def modify_Q_ctx(self):
        """
        Use this context manager if you want to modify Q; auxiliary data structures
        will be updated accordingly afterwards.
        """
        yield
        self._recompute_class_to_frames()

    def _recompute_class_to_frames(self):
        class_idx_to_frames = {idx: [] for idx, class_key in enumerate(self.class_labels)}
        for frame_idx in tqdm((self.Q != -1).nonzero(as_tuple=True)[0], desc="Building inverse lookup"):
            class_idx = int(self.Q[frame_idx].item())
            class_idx_to_frames[class_idx].append(frame_idx.item())
        self.class_to_frames = class_idx_to_frames
    
    def drop_lengths(self, max_length: int) -> "SpeechEquivalenceDataset":
        """
        Drop all frames with length greater than `max_length`.
        """
        mask = self.lengths > max_length

        new_Q = self.Q.clone()
        new_Q[mask] = -1
        new_S = self.S.clone()
        new_S[mask] = -1

        # update class_to_frames
        new_class_to_frames = {class_idx: [frame_idx for frame_idx in frames if not mask[frame_idx]]
                               for class_idx, frames in self.class_to_frames.items()}

        return replace(self, Q=new_Q, S=new_S, class_to_frames=new_class_to_frames)
    

def make_timit_equivalence_dataset(name: str,
                                   dataset: datasets.Dataset,
                                   hidden_states: SpeechHiddenStateDataset,
                                   equivalence_classer: str,
                                   minimum_frequency_percentile: float = 0.,
                                   minimum_frequency_count: int = 0,
                                   max_length: Optional[int] = 100,
                                   start_reference: Optional[str] = None,
                                   num_frames_per_phoneme=None) -> SpeechEquivalenceDataset:
    """
    TIMIT-specific function to prepare an equivalence-classed frame dataset
    from a TIMIT dataset and a speech model.

    NB that equivalence classing is not specific to models and could be
    decoupled in principle.
    """
    assert equivalence_classer in equivalence_classers

    frame_groups = defaultdict(list)
    # count number of WORDS which contain instances of each group
    # NB this under-counts phoneme instances. we only care about this for filtering
    # the open-class units (e.g. words) so it's okay. (See usage below in
    # `minimum_frequency_percentile` filtering.)
    group_counts = Counter()

    frames_by_item = hidden_states.frames_by_item

    # Align with TIMIT annotations
    def process_item(item, idx):
        compression_ratio = hidden_states.compression_ratios[idx]
        seen_groups = set()

        for i, word in enumerate(item["word_phonemic_detail"]):
            if len(word) == 0:
                continue

            word_str = item["word_detail"]["utterance"][i]
            word_start = int(word[0]["start"] * compression_ratio)
            word_end = int(word[-1]["stop"] * compression_ratio)

            for j, phone in enumerate(word):
                phone_str = phone["phone"]
                phone_start = int(phone["start"] * compression_ratio)
                phone_end = int(phone["stop"] * compression_ratio)

                ks = list(range(phone_start, phone_end + 1))
                if num_frames_per_phoneme is not None and len(ks) > num_frames_per_phoneme:
                    # Sample uniformly spaced frames within the span of the phoneme
                    ks = np.linspace(phone_start, phone_end, num_frames_per_phoneme).round().astype(int)
                for k in ks:
                    class_label = equivalence_classers[equivalence_classer](word, word_str, j)
                    if class_label is not None:
                        frame_groups[class_label].append((idx, k))
                        seen_groups.add(class_label)

        for group in seen_groups:
            group_counts[group] += 1

    dataset.map(process_item, with_indices=True, desc="Aligning metadata")

    if len(group_counts) > 0 and minimum_frequency_percentile > 0:
        # Filter out low-frequency classes
        min_count = np.percentile(np.array(list(group_counts.values())), minimum_frequency_percentile)

        len_before = len(frame_groups)
        frame_groups = {class_key: group for class_key, group in frame_groups.items() if group_counts[class_key] >= min_count}
        len_after = len(frame_groups)
        print(f"Filtered out {len_before - len_after} classes with fewer than {min_count} frames")
        print(f"Remaining classes: {len_after}")
    
    if len(group_counts) > 0 and minimum_frequency_count > 0:
        if minimum_frequency_percentile > 0 and min_count >= minimum_frequency_count:
            print("Skipping minimum frequency count filtering because minimum frequency percentile set a more stringent criterion")
        else:
            len_before = len(frame_groups)
            frame_groups = {class_key: group for class_key, group in frame_groups.items() if group_counts[class_key] >= minimum_frequency_count}
            len_after = len(frame_groups)
            print(f"Filtered out {len_before - len_after} classes with fewer than {minimum_frequency_count} frames")
            print(f"Remaining classes: {len_after}")

    # Now run equivalence classing.
    Q = torch.zeros(len(hidden_states.flat_idxs), dtype=torch.long) - 1
    class_labels = sorted(frame_groups.keys())
    class_label_to_idx = {class_key: idx for idx, class_key in enumerate(class_labels)}
    flat_idx_rev = {tuple(idx): i for i, idx in enumerate(hidden_states.flat_idxs)}
    # Prepare a redundant representation, mapping from class idx to list of flat idxs.
    class_idx_to_frames = {idx: [] for idx, class_key in enumerate(class_labels)}
    for class_key, group in tqdm(frame_groups.items(), desc="Building forward lookup"):
        for idx, frame in group:
            Q[flat_idx_rev[idx, frame]] = class_label_to_idx[class_key]
    for frame_idx in tqdm((Q != -1).nonzero(as_tuple=True)[0], desc="Building inverse lookup"):
        class_key = class_labels[Q[frame_idx].item()]
        class_idx = class_label_to_idx[class_key]
        class_idx_to_frames[class_idx].append(frame_idx.item())

    # Compute start frames.
    S = torch.zeros(len(hidden_states.flat_idxs), dtype=torch.long) - 1
    if start_reference is None:
        start_reference = start_references[equivalence_classer]

    if start_reference == "word":
        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = hidden_states.compression_ratios[idx]

            for word in item["word_phonemic_detail"]:
                if len(word) == 0:
                    continue
                word_str = tuple(phone["phone"] for phone in word)
                word_start = int(word[0]["start"] * compression_ratio)
                word_end = int(word[-1]["stop"] * compression_ratio)

                for j in range(word_start, word_end + 1):
                    S[flat_idx_offset + j] = flat_idx_offset + word_start
    elif start_reference == "phoneme":
        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = hidden_states.compression_ratios[idx]

            for word in item["word_phonemic_detail"]:
                for phone in word:
                    phone_str = phone["phone"]
                    phone_start = int(phone["start"] * compression_ratio)
                    phone_end = int(phone["stop"] * compression_ratio)

                    for j in range(phone_start, phone_end + 1):
                        S[flat_idx_offset + j] = flat_idx_offset + phone_start
    elif start_reference[0] == "fixed":
        fixed_length = int(start_reference[1])
        if max_length is not None and fixed_length > max_length:
            raise ValueError(f"Fixed length {fixed_length} exceeds max length {max_length}")

        def compute_start(item, idx):
            flat_idx_offset, flat_idx_end = frames_by_item[idx]
            num_frames = flat_idx_end - flat_idx_offset
            compression_ratio = hidden_states.compression_ratios[idx]

            for word in item["word_phonemic_detail"]:
                for phone in word:
                    phone_str = phone["phone"]
                    phone_start = int(phone["start"] * compression_ratio)
                    phone_end = int(phone["stop"] * compression_ratio)

                    for j in range(phone_start, phone_end + 1):
                        S[flat_idx_offset + j] = max(flat_idx_offset, flat_idx_offset + j - fixed_length)
    else:
        raise ValueError(f"Unknown start reference {start_reference}")

    dataset.map(compute_start, with_indices=True, desc="Computing start frames")

    ret = SpeechEquivalenceDataset(name=name,
                                   Q=Q,
                                   S=S,
                                   class_to_frames=class_idx_to_frames,
                                   class_labels=class_labels)
    if max_length is not None:
        ret = ret.drop_lengths(max_length)
    return ret