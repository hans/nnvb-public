import itertools
from typing import Optional, TypeAlias
import logging

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.analysis.analogy import nxn_cos_sim

L = logging.getLogger(__name__)


PredictionEquivalenceKey: TypeAlias = tuple
PredictionEquivalenceCollection: TypeAlias = dict[PredictionEquivalenceKey, dict[str, set[int]]]


def prepare_prediction_equivalences(cuts_df, cut_phonemic_forms, cohort, next_phoneme):
    """
    Equivalence-class vocabulary for evaluating predictions on the given
    cohort + next phoneme.
    """

    cohort_length = cohort.count(" ") + 1 if cohort != "" else 0

    # next-phoneme strict: match next phoneme
    matches_next_phoneme = cuts_df.xs(cohort_length, level="frame_idx") \
        .query("description == @next_phoneme")

    # next-phoneme weak: match next phoneme, but allow predicting that phoneme frame
    # or any frame of the word
    matches_next_phoneme_weak = cuts_df.join(
        matches_next_phoneme.traj_flat_idx.rename("next_phoneme_flat_idx"),
        how="inner"
    )

    # matches cohort
    matches_cohort = cut_phonemic_forms[cut_phonemic_forms.str.match(f"^{cohort}")].index
    matches_cohort = pd.merge(cuts_df.reset_index(),
                              matches_cohort.to_frame(index=False),
                              how="inner",
                              on=["label", "instance_idx"]) \
        .query("frame_idx >= @cohort_length")
    
    matches_cohort_and_next_phoneme = matches_next_phoneme[
        matches_next_phoneme.traj_flat_idx.isin(matches_cohort.traj_flat_idx)]
    
    matches_cohort_and_next_phoneme_weak = matches_next_phoneme_weak[
        matches_next_phoneme_weak.traj_flat_idx.isin(matches_cohort.traj_flat_idx)]

    ret = {
        "matches_next_phoneme": matches_next_phoneme.traj_flat_idx.values,
        "matches_next_phoneme_weak": matches_next_phoneme_weak.traj_flat_idx.values,
        "matches_cohort": matches_cohort.traj_flat_idx.values,
        "matches_cohort_and_next_phoneme": matches_cohort_and_next_phoneme.traj_flat_idx.values,
        "matches_cohort_and_next_phoneme_weak": matches_cohort_and_next_phoneme_weak.traj_flat_idx.values,
    }

    return ret


def iter_equivalences(
        config, all_cross_instances, agg_src: np.ndarray,
        num_samples=100, max_num_vector_samples=250,
        flat_idx_lookup=None,
        seed=None,):
    
    # Pre-compute lookup from label idx, instance idx to flat idx
    if isinstance(agg_src, torch.Tensor):
        agg_src = agg_src.cpu().numpy()

    if flat_idx_lookup is None:
        # not pre-computed    
        flat_idx_lookup = {(label_idx, instance_idx, phoneme_idx): flat_idx
                        for flat_idx, (label_idx, instance_idx, phoneme_idx) in enumerate(agg_src)}

    def get_flat_idx(label_idx, instance_idx, phoneme_idx):
        if phoneme_idx >= 0:
            return flat_idx_lookup[label_idx, instance_idx, phoneme_idx]
        elif phoneme_idx == -1:
            # get frame just preceding word onset
            return flat_idx_lookup[label_idx, instance_idx, 0] - 1
        else:
            raise ValueError(f"Invalid phoneme index: {phoneme_idx}")

    if seed is not None:
        np.random.seed(seed)

    if "group_by" in config:
        grouper = all_cross_instances.groupby(config["group_by"])
    else:
        grouper = [("", all_cross_instances)]

    for group, rows in tqdm(grouper, leave=False):
        try:
            if "base_query" in config:
                rows_from = rows.query(config["base_query"])
            else:
                rows_from = rows

            if "inflected_query" in config:
                rows_to = rows.query(config["inflected_query"])
            else:
                rows_to = rows

            if "all_query" in config:
                rows_from = rows_from.query(config["all_query"])
                rows_to = rows_to.query(config["all_query"])

            inflection_from = rows_from.inflection.iloc[0]
            inflection_to = rows_to.inflection.iloc[0]
        except IndexError:
            continue

        if len(rows_from) == 0 or len(rows_to) == 0:
            continue

        # prepare equivalences for 'from' and 'to' groups.
        # equivalences define the set of instances over which we can average representations
        # before computing the analogy.
        if "equivalence_keys" in config:
            from_equivalence_keys = config["equivalence_keys"]
            to_equivalence_keys = config["equivalence_keys"]
        else:
            from_equivalence_keys = ["inflected_phones"]
            to_equivalence_keys = ["inflected_phones"]

        # we must group on at least the forms themselves
        assert set(["inflected_phones"]) <= set(from_equivalence_keys)
        assert set(["inflected_phones"]) <= set(to_equivalence_keys)

        from_equiv = rows_from.groupby(from_equivalence_keys)
        to_equiv = rows_to.groupby(to_equivalence_keys)
        from_equiv_labels = [k for k, count in from_equiv.size().items() if count >= 1]
        to_equiv_labels = [k for k, count in to_equiv.size().items() if count >= 1]

        if len(set(from_equiv_labels) | set(to_equiv_labels)) <= 1:
            # not enough labels to support transfer.
            L.error(f"Skipping {group} due to insufficient labels")
            continue

        # Make sure labels are tuples
        if not isinstance(from_equiv_labels[0], tuple):
            from_equiv_labels = [(label,) for label in from_equiv_labels]
        if not isinstance(to_equiv_labels[0], tuple):
            to_equiv_labels = [(label,) for label in to_equiv_labels]

        # sample pairs of base forms
        candidate_pairs = [(x, y) for x, y in itertools.product(from_equiv_labels, to_equiv_labels) if x != y]
        num_samples_i = min(num_samples, len(candidate_pairs))
        samples = np.random.choice(len(candidate_pairs), num_samples_i, replace=False)

        for idx in tqdm(samples, leave=False):
            from_equiv_label_i, to_equiv_label_i = candidate_pairs[idx]
            rows_from_i = from_equiv.get_group(tuple(from_equiv_label_i))
            rows_to_i = to_equiv.get_group(tuple(to_equiv_label_i))

            # sample pairs for comparison across the two forms
            n = min(max_num_vector_samples, max(len(rows_from_i), len(rows_to_i)))
            if len(rows_from_i) < n:
                rows_from_i = rows_from_i.sample(n, replace=True)
            elif len(rows_from_i) > n:
                rows_from_i = rows_from_i.sample(n, replace=False)

            if len(rows_to_i) < n:
                rows_to_i = rows_to_i.sample(n, replace=True)
            elif len(rows_to_i) > n:
                rows_to_i = rows_to_i.sample(n, replace=False)

            from_label = rows_from_i.inflected.iloc[0]
            from_idx = rows_from_i.inflected_idx.iloc[0]
            to_label = rows_to_i.inflected.iloc[0]
            to_idx = rows_to_i.inflected_idx.iloc[0]

            # what are the "base" and "inflected" forms?
            from_base_phones = rows_from_i.base_phones.iloc[0].split()
            from_post_divergence = rows_from_i.post_divergence.iloc[0].split()
            to_base_phones = rows_to_i.base_phones.iloc[0].split()
            to_post_divergence = rows_to_i.post_divergence.iloc[0].split()

            # compute individual representation indices
            from_inflected_flat_idx = torch.tensor(
                [get_flat_idx(row.inflected_idx, row.inflected_instance_idx, row.next_phoneme_idx)
                 for _, row in rows_from_i.iterrows()])
            # TODO design choice: do we take repr from previous phoneme or averaged over all previous
            # phonemes?
            from_base_flat_idx = torch.tensor(
                [get_flat_idx(row.inflected_idx, row.inflected_instance_idx, row.next_phoneme_idx - 1)
                 for _, row in rows_from_i.iterrows()])
            to_base_flat_idx = torch.tensor(
                [get_flat_idx(row.inflected_idx, row.inflected_instance_idx, row.next_phoneme_idx - 1)
                 for _, row in rows_to_i.iterrows()])
            
            yield {
                "group": group,

                "from_label": from_label,
                "from_idx": from_idx,
                "to_label": to_label,
                "to_idx": to_idx,

                "from_inflected_phones": rows_from_i.inflected_phones.iloc[0],
                "from_base_phones": " ".join(from_base_phones),
                "from_post_divergence": " ".join(from_post_divergence),

                "to_inflected_phones": rows_to_i.inflected_phones.iloc[0],
                "to_base_phones": " ".join(to_base_phones),
                "to_post_divergence": " ".join(to_post_divergence),

                "inflection_from": inflection_from,
                "inflection_to": inflection_to,
                "from_equiv_label_i": from_equiv_label_i,
                "to_equiv_label_i": to_equiv_label_i,
                
                "from_inflected_flat_idx": from_inflected_flat_idx,
                "from_base_flat_idx": from_base_flat_idx,
                "to_base_flat_idx": to_base_flat_idx,                
            }


def run_experiment_equiv_level(
        experiment_name, config,
        state_space_spec, all_cross_instances,
        agg, agg_src,
        cut_phonemic_forms,
        flat_idx_lookup=None,
        device: str = "cpu",
        verbose=False,
        num_samples=100, max_num_vector_samples=250,
        seed=None,
        prediction_equivalences: Optional[PredictionEquivalenceCollection] = None,
        exclude_idxs_from_predictions: Optional[dict[int, list[int]]] = None,
        include_idxs_in_predictions: Optional[dict[int, list[int]]] = None):
    """
    Args:
        prediction_equivalences: defines a collection of equivalence classes
            instantiating different prediction evaluations. Each equivalence 
            class specifies, for a given prediction instance, a set of flat
            indices (indices into `agg` which should be counted as "correct")
            for that prediction instance. The config item `prediction_equivalence_keys`
            determines which properties of a sample returned by `iter_equivalences`
            are used to map from item to prediction instance.

            If this is `None`, then prediction success is determined based on
            `include_idxs_in_predictions` (label indices), with a backup to
            simply matching the single ground truth inflected label.
    """
    print(experiment_name)

    prediction_equivalences_tensor = None
    if prediction_equivalences is not None:
        if include_idxs_in_predictions is not None:
            raise ValueError("Cannot specify both `prediction_equivalences` and `include_idxs_in_predictions`")
        if "prediction_equivalence_keys" not in config:
            raise ValueError("`prediction_equivalence_keys` must be specified in `config` if `prediction_equivalences` is provided")

        if any(not isinstance(flat_idxs, torch.Tensor) for items in prediction_equivalences.values()
               for flat_idxs in items.values()):
            prediction_equivalences_tensor = {
                key: {
                    equiv: torch.as_tensor(flat_idxs)
                    for equiv, flat_idxs in items.items()
                }
                for key, items in prediction_equivalences.items()
            }
        else:
            prediction_equivalences_tensor = prediction_equivalences

    if not isinstance(agg, torch.Tensor) or agg.device != torch.device(device):
        # move data to device
        agg = torch.as_tensor(agg).to(device)
        agg_src = torch.as_tensor(agg_src).to(device)
    
    results = []
    for sample in iter_equivalences(
            config, all_cross_instances, agg_src,
            flat_idx_lookup=flat_idx_lookup,
            num_samples=num_samples,
            max_num_vector_samples=max_num_vector_samples,
            seed=seed):

        from_inflected_flat_idx = sample["from_inflected_flat_idx"]
        from_base_flat_idx = sample["from_base_flat_idx"]
        to_base_flat_idx = sample["to_base_flat_idx"]

        # Critical analogy logic
        pair_difference = agg[from_inflected_flat_idx] - agg[from_base_flat_idx]
        pair_base = agg[to_base_flat_idx]

        pair_predicted = pair_base + pair_difference
        pair_predicted /= torch.norm(pair_predicted, dim=1, keepdim=True)

        ### Compute ranks over entire set of word tokens

        references, references_src = agg, agg_src
        with torch.no_grad():
            dists = 1 - nxn_cos_sim(pair_predicted, references)
            # mean over instances of pair
            dists = dists.mean(0)

        if exclude_idxs_from_predictions is not None:
            invalid_idxs = torch.tensor(list(exclude_idxs_from_predictions[sample["inflection_to"], sample["to_idx"]]))
            invalid_flat_idxs = torch.where(torch.isin(references_src[:, 0], invalid_idxs))[0]
            dists[invalid_flat_idxs] = torch.inf

        sorted_indices = dists.argsort()
        ranks = torch.zeros_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(sorted_indices)).to(sorted_indices)

        ### Prepare evaluations

        # Map from evaluation name to a tensor of valid flat idxs for this prediction problem
        evaluations: dict[str, torch.Tensor] = {}

        if prediction_equivalences_tensor is not None:
            prediction_equivalence_keys = config["prediction_equivalence_keys"]
            prediction_equivalence_keys = tuple(sample[key] for key in prediction_equivalence_keys)
            if prediction_equivalence_keys not in prediction_equivalences_tensor:
                continue

            for subexperiment, valid_flat_idxs in prediction_equivalences_tensor[prediction_equivalence_keys].items():
                evaluations[subexperiment] = valid_flat_idxs
        else:
            if include_idxs_in_predictions is not None:
                valid_label_idxs = torch.tensor(list(include_idxs_in_predictions[sample["to_idx"]])).to(device)
            else:
                valid_label_idxs = torch.tensor([sample["to_idx"]]).to(device)
            valid_flat_idxs = torch.where(torch.isin(references_src[:, 0], valid_label_idxs))[0]
            evaluations[""] = valid_flat_idxs

        ### Run evaluations

        evaluation_results = {}
        for evaluation, valid_flat_idxs in evaluations.items():
            if len(valid_flat_idxs) == 0:
                continue

            nearest_neighbor = references_src[sorted_indices[0]]

            # terminology
            # target: nearest valid embedding for this evaluation
            target_rank, target_subidx = torch.min(ranks[valid_flat_idxs], dim=0)
            target_rank, target_idx = target_rank.item(), valid_flat_idxs[target_subidx].item()
            target_distance = dists[target_idx].item()
            target_label_idx = references_src[target_idx, 0].item()
            target_instance_idx = references_src[target_idx, 1].item()
            target_phoneme_idx = references_src[target_idx, 2].item()
            target_label = state_space_spec.labels[target_label_idx]
            target_phones = cut_phonemic_forms.loc[target_label].loc[target_instance_idx]

            # predicted: nearest neighbor
            predicted_label_idx = nearest_neighbor[0].item()
            predicted_instance_idx = nearest_neighbor[1].item()
            predicted_label = state_space_spec.labels[predicted_label_idx]

            evaluation_results[evaluation] = {
                "target_rank": target_rank,
                "target_distance": target_distance,
                "target_label_idx": target_label_idx,
                "target_instance_idx": target_instance_idx,
                "target_phoneme_idx": target_phoneme_idx,
                "target_label": target_label,
                "target_phones": target_phones,

                "predicted_distance": dists[sorted_indices[0]].item(),
                "predicted_label_idx": predicted_label_idx,
                "predicted_instance_idx": predicted_instance_idx,
                "predicted_phoneme_idx": nearest_neighbor[2].item(),
                "predicted_label": predicted_label,
                "predicted_phones": cut_phonemic_forms.loc[predicted_label].loc[predicted_instance_idx],
            }

        if verbose:
            for flat_idx, dist, (label_idx, instance_idx, _) in zip(sorted_indices[:5], dists[sorted_indices[:5]], references_src[sorted_indices[:5]]):
                print(f"{sample['group']} {sample['from_equiv_label_i']} -> {sample['to_equiv_label_i']} {state_space_spec.labels[label_idx]} {cut_phonemic_forms.loc[state_space_spec.labels[label_idx]].loc[instance_idx.item()]} {dist.item()}")

        # Merge into a single flat dictionary
        results_i = {}
        for evaluation_name, evaluation in evaluation_results.items():
            for key, value in evaluation.items():
                output_key = f"{evaluation_name}_{key}" if evaluation_name else key
                results_i[output_key] = value
        results_i.update({
            "group": sample["group"],

            "from": sample["from_label"],
            "to": sample["to_label"],

            "from_inflected_phones": sample["from_inflected_phones"],
            "from_base_phones": sample["from_base_phones"],
            "from_post_divergence": sample["from_post_divergence"],

            "to_inflected_phones": sample["to_inflected_phones"],
            "to_base_phones": sample["to_base_phones"],
            "to_post_divergence": sample["to_post_divergence"],

            "inflection_from": sample["inflection_from"],
            "inflection_to": sample["inflection_to"],
            "from_equiv_label": sample["from_equiv_label_i"],
            "to_equiv_label": sample["to_equiv_label_i"],
        })

        results.append(results_i)

    return pd.DataFrame(results)