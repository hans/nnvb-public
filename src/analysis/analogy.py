from collections import  defaultdict
import itertools
import logging
from typing import Optional

import lemminflect
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

L = logging.getLogger(__name__)


def get_inflection(word, all_labels, target) -> set[str]:
    if target in ("VBD", "VBZ", "VBG", "NNS"):
        inflections = set(lemminflect.getInflection(word, tag=target, inflect_oov=False))
        # don't include zero-derived form
        inflections -= {word}
    elif target == "NOT-latin":
        inflections = {"in" + word}
        if word[0] == "l":
            inflections |= {"il" + word}
        elif word[0] in ["p", "b", "m"]:
            inflections |= {"im" + word}
        elif word[0] == "r":
            inflections |= {"ir" + word}

        # catch exceptional cases -- these predicted forms are attested, but don't count
        # as a negative inflection. e.g. "come" -> "income"
        if word in ("come comes deed diana dies doors fancy form formation formed forming "
                    "habit jury justice k l laid land most n part parted port press pressed "
                    "prove proved pulse pulses side sight stead sure tend tended tending tent "
                    "to trusted vent ward"):
            inflections = set()
    else:
        raise ValueError(f"Unknown target: {target}")
    
    covered_inflections = inflections & all_labels
    return covered_inflections


def is_regular(inflection, base, inflected):
    if inflection == "NNS":
        return inflected[:len(base)] == base \
                or inflected[-3:] == "ies" and base[-1] == "y" \
                or inflected[-3:] == "ves" and (base[-1] == "f" or base[-2:] == "fe")
    elif inflection == "VBZ":
        return inflected == base + "s" \
                or inflected == base + "es" \
                or (base[-1] == "y" and inflected == base[:-1] + "ies")
    elif inflection == "VBG":
        return inflected == base + "ing" \
                or (base[-1] == "e" and inflected == base[:-1] + "ing") \
                or (base[-1] in "bcdfghjklmnpqrstvwxz" and inflected == base + base[-1] + "ing") \
                or (base[-2:] == "ie" and inflected == base[:-2] + "ying")
    elif inflection == "VBD":
        return inflected == base + "ed" \
                or inflected == base + "d" \
                or inflected == base + "t" \
                or (base[-1] == "y" and inflected == base[:-1] + "ied") \
                or (base[-2:] == "ay" and inflected == base[:-1] + "id") \
                or (base[-1] in "bcdfghjklmnpqrstvwxz" and inflected == base + base[-1] + "ed")
    elif inflection == "NOT-latin":
        return inflected[:2] == "in"
    else:
        raise ValueError(f"Unknown inflection {inflection}")
    

def get_inflection_df(inflection_targets, words):
    inflection_results = {target: {} for target in inflection_targets}
    inflection_reverse = defaultdict(set)
    for target in inflection_targets:
        for word in words:
            label_inflections = get_inflection(word, words, target)
            if label_inflections:
                inflection_results[target][word] = label_inflections

                for infl in label_inflections:
                    inflection_reverse[infl].add((word, target))

    inflection_results_df = pd.DataFrame([
        {"inflected": inflection, "base": label, "inflection": target}
        for target, inflections in inflection_results.items()
        for label, label_inflections in inflections.items()
        for inflection in label_inflections
    ]).set_index("inflection")
    inflection_results_df["is_regular"] = inflection_results_df.apply(
        lambda row: is_regular(row.name, row.base, row.inflected), axis=1)

    return inflection_results_df


def has_inflection_relation(row, inflection):
    if inflection == "POS":
        return row.label.endswith("'s") and row.label[:-len("'s")] == row.inferred_base
    else:
        return row.label in lemminflect.getInflection(row.inferred_base, tag=inflection, inflect_oov=False)


def prepare_false_friends(inflection_results_df, inflection_instance_df,
                          cut_phonemic_forms, target_post_divergence,
                          avoid_inflections: list[str],
                          min_frequency=5):
    """
    Prepare a dataset of false-friend token pairs which exhibit the same
    phonological relationship between "base" and "inflected", but which do
    not share the same inflectional relationship. Does not include false-friend
    pairs which are homophonous with a real inflection in `avoid_inflections`
    (e.g. "knight" -> "night's" is a false friend pair, but homophonous with
    "knight" -> "knights").
    """

    avoid_forms = set(inflection_results_df.loc[list(set(avoid_inflections) & set(inflection_results_df.index.get_level_values("inflection")))] \
                      [["inflected", "base"]].melt().value.tolist())
    
    label_counts = cut_phonemic_forms.groupby("label").size()

    # whole form shares the post-divergence content
    step0 = cut_phonemic_forms.loc[cut_phonemic_forms.str[-len(target_post_divergence):] == target_post_divergence]
    # remove real inflections and real bases
    step1 = step0[~step0.index.get_level_values("label").isin(avoid_forms)]
    # and if you remove post-div content, it's still attested
    step2 = step1[step1.str[:-len(target_post_divergence)].str.strip().isin(cut_phonemic_forms)]
    step3 = step2.reset_index()

    def get_base_form(description) -> tuple[str, str]:
        candidates = cut_phonemic_forms[cut_phonemic_forms == description[:-len(target_post_divergence)].strip()].reset_index()
        return candidates.groupby(["label", "description"]).size().index[0]
    step3_bases = {description: get_base_form(description) for description in step3.description.unique()}
    step3["inferred_base"] = step3.description.map({desc: label for desc, (label, _) in step3_bases.items()})
    step3["inferred_base_form"] = step3.description.map({desc: form for desc, (_, form) in step3_bases.items()})

    for avoid_inflection in avoid_inflections:
        step3[f"is_{avoid_inflection}"] = step3.apply(has_inflection_relation, axis=1, inflection=avoid_inflection)
    step3["has_avoid_inflection"] = step3[[f"is_{avoid_inflection}" for avoid_inflection in avoid_inflections]].any(axis=1)
    step3 = step3[~step3.has_avoid_inflection]

    # filter for non-long-tail words
    step4 = pd.merge(step3, label_counts.rename("base_count"),
                    left_on="inferred_base", right_index=True, how="inner")
    step4 = pd.merge(step4, label_counts.rename("inflected_count"),
                    left_on="label", right_index=True, how="inner")
    step4 = step4[(step4.base_count >= min_frequency) & (step4.inflected_count >= min_frequency)]

    # should not be homophonous with avoided inflections
    avoid_phon_forms = set(inflection_instance_df[inflection_instance_df.inflection.isin(avoid_inflections)].inflected_phones.unique())
    step5 = step4[~step4.description.isin(avoid_phon_forms)]

    ret = step5.groupby(["inferred_base", "inferred_base_form"]).head(1) \
        [["inferred_base", "inferred_base_form", "label", "description"]] \
        .rename(columns={"inferred_base": "base", "inferred_base_form": "base_form",
                         "label": "inflected", "description": "inflected_form"})
    return ret


def sample_non_identical_pairs(xs, ys, k):
    xs = np.array(list(xs))
    ys = np.array(list(ys))

    mask = np.ones(k, dtype=bool)
    x_samples, y_samples = np.empty((k, *xs.shape[1:]), dtype=xs.dtype), np.empty((k, *ys.shape[1:]), dtype=ys.dtype)
    i = 0
    while mask.any():
        k_i = mask.sum()
        x_idxs = np.random.choice(len(xs), k_i, replace=True)
        y_idxs = np.random.choice(len(ys), k_i, replace=True)

        x_samples[mask] = xs[x_idxs]
        y_samples[mask] = ys[y_idxs]

        mask = x_samples == y_samples
        if mask.ndim == 2:
            mask = mask.all(1)

        i += 1
        if i > 100:
            raise RuntimeError("Infinite loop")

    return x_samples, y_samples


def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)
    

def iter_equivalences(
        config, all_cross_instances, agg_src: np.ndarray,
        num_samples=100, max_num_vector_samples=250,
        seed=None,):
    
    # Pre-compute lookup from label idx, instance idx to flat idx
    if isinstance(agg_src, torch.Tensor):
        agg_src = agg_src.cpu().numpy()
    flat_idx_lookup = {(label_idx, instance_idx): flat_idx
                       for flat_idx, (label_idx, instance_idx, _) in enumerate(agg_src)}

    if seed is not None:
        np.random.seed(seed)

    if "group_by" in config:
        grouper = all_cross_instances.groupby(config["group_by"])
    else:
        grouper = [("", all_cross_instances)]

    for group, rows in tqdm(grouper, leave=False):
        print(group)

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
            from_equivalence_keys = ["base", "inflected"]
            to_equivalence_keys = ["base", "inflected"]

        # we must group on at least the forms themselves
        assert set(["base", "inflected"]) <= set(from_equivalence_keys)
        assert set(["base", "inflected"]) <= set(to_equivalence_keys)

        from_equiv = rows_from.groupby(from_equivalence_keys)
        to_equiv = rows_to.groupby(to_equivalence_keys)
        from_equiv_labels = [k for k, count in from_equiv.size().items() if count >= 1]
        to_equiv_labels = [k for k, count in to_equiv.size().items() if count >= 1]

        if len(set(from_equiv_labels) | set(to_equiv_labels)) <= 1:
            # not enough labels to support transfer.
            L.error(f"Skipping {group} due to insufficient labels")
            continue

        # sample pairs of base forms
        candidate_pairs = [(x, y) for x, y in itertools.product(from_equiv_labels, to_equiv_labels) if x != y]
        num_samples_i = min(num_samples, len(candidate_pairs))
        samples = np.random.choice(len(candidate_pairs), num_samples_i, replace=False)

        for idx in tqdm(samples, leave=False):
            from_equiv_label_i, to_equiv_label_i = candidate_pairs[idx]
            rows_from_i = from_equiv.get_group(tuple(from_equiv_label_i))
            rows_to_i = to_equiv.get_group(tuple(to_equiv_label_i))

            base_from_label = rows_from_i.base.iloc[0]
            base_from_idx = rows_from_i.base_idx.iloc[0]
            base_to_label = rows_to_i.base.iloc[0]
            base_to_idx = rows_to_i.base_idx.iloc[0]
            inflected_from_label = rows_from_i.inflected.iloc[0]
            inflected_from_idx = rows_from_i.inflected_idx.iloc[0]
            inflected_to_label = rows_to_i.inflected.iloc[0]
            inflected_to_idx = rows_to_i.inflected_idx.iloc[0]

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

            from_inflected_flat_idx = torch.tensor(
                [flat_idx_lookup[(row.inflected_idx, row.inflected_instance_idx)]
                 for _, row in rows_from_i.iterrows()])
            from_base_flat_idx = torch.tensor(
                [flat_idx_lookup[(row.base_idx, row.base_instance_idx)]
                 for _, row in rows_from_i.iterrows()])
            to_base_flat_idx = torch.tensor(
                [flat_idx_lookup[(row.base_idx, row.base_instance_idx)]
                 for _, row in rows_to_i.iterrows()])
            
            yield {
                "group": group,

                "base_from_label": base_from_label,
                "base_from_idx": base_from_idx,
                "inflected_from_label": inflected_from_label,
                "inflected_from_idx": inflected_from_idx,
                "base_to_label": base_to_label,
                "base_to_idx": base_to_idx,
                "inflected_to_label": inflected_to_label,
                "inflected_to_idx": inflected_to_idx,

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
        device: str = "cpu",
        verbose=False,
        num_samples=100, max_num_vector_samples=250,
        seed=None,
        exclude_base_from_predictions=True,
        include_idxs_in_predictions: Optional[dict[int, list[int]]] = None):
    print(experiment_name)

    # move data to device
    agg = torch.tensor(agg).to(device)
    agg_src = torch.tensor(agg_src).to(device)
    
    results = []
    for sample in iter_equivalences(
            config, all_cross_instances, agg_src,
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

        references, references_src = agg, agg_src
        with torch.no_grad():
            dists = 1 - nxn_cos_sim(pair_predicted, references)
            # mean over instances of pair
            dists = dists.mean(0)

        if exclude_base_from_predictions:
            base_flat_idxs = torch.where(references_src[:, 0] == sample["base_to_idx"])[0]
            dists[base_flat_idxs] = torch.inf

        sorted_indices = dists.argsort()
        ranks = torch.zeros_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(sorted_indices)).to(sorted_indices)

        if include_idxs_in_predictions is not None:
            valid_label_idxs = torch.tensor(list(include_idxs_in_predictions[sample["inflected_to_idx"]])).to(device)
        else:
            valid_label_idxs = torch.tensor([sample["inflected_to_idx"]]).to(device)
        valid_flat_idxs = torch.where(torch.isin(references_src[:, 0], valid_label_idxs))[0]

        # get best rank+distance result
        gt_rank = ranks[valid_flat_idxs].min().item()
        gt_distance = dists[valid_flat_idxs].min().item()

        if verbose:
            log_k = 100
            for dist, (label_idx, instance_idx, _) in zip(dists[sorted_indices[:log_k]], references_src[sorted_indices[:log_k]]):
                print(f"{sample['group']} {sample['from_equiv_label_i']} -> {sample['to_equiv_label_i']}: {state_space_spec.labels[label_idx]} {instance_idx}")

        nearest_neighbor = references_src[sorted_indices[0]]
        results.append({
            "group": sample["group"],
            "from_equiv_label": sample["from_equiv_label_i"],
            "to_equiv_label": sample["to_equiv_label_i"],
            "base_from": sample["base_from_label"],
            "base_to": sample["base_to_label"],
            "inflection_from": sample["inflection_from"],
            "inflection_to": sample["inflection_to"],
            "predicted_label_idx": nearest_neighbor[0].item(),
            "predicted_label": state_space_spec.labels[nearest_neighbor[0]],
            "predicted_instance_idx": nearest_neighbor[1].item(),
            "gt_label": sample["inflected_to_label"],
            "gt_label_idx": sample["inflected_to_idx"],
            "gt_label_rank": gt_rank,
            "gt_distance": gt_distance,
        })

    return pd.DataFrame(results)


def get_representations_equiv_level(
        experiment_name, config,
        state_space_spec, all_cross_instances,
        agg, agg_src,
        device: str = "cpu",
        verbose=False,
        num_samples=100, max_num_vector_samples=250,
        seed=None):
    pass
    


# def run_experiment_equiv_level(
#         experiment_name, config,
#         state_space_spec, all_cross_instances,
#         agg, agg_src,
#         device: str = "cpu",
#         method="torch",
#         verbose=False,
#         num_samples=100, max_num_vector_samples=250,
#         seed=None,
#         exclude_base_from_predictions=True):
    
#     # Pre-compute lookup from label idx, instance idx to flat idx
#     flat_idx_lookup = {(label_idx, instance_idx): flat_idx
#                        for flat_idx, (label_idx, instance_idx, _) in enumerate(agg_src)}
    
#     if method == "torch":
#         # move data to device
#         agg = torch.tensor(agg).to(device)
#         agg_src = torch.tensor(agg_src).to(device)

#     if seed is not None:
#         np.random.seed(seed)

#     results = []
#     if "group_by" in config:
#         grouper = all_cross_instances.groupby(config["group_by"])
#     else:
#         grouper = [("", all_cross_instances)]
#     for group, rows in tqdm(grouper, leave=False):
#         try:
#             if "base_query" in config:
#                 rows_from = rows.query(config["base_query"])
#             else:
#                 rows_from = rows

#             if "inflected_query" in config:
#                 rows_to = rows.query(config["inflected_query"])
#             else:
#                 rows_to = rows

#             if "all_query" in config:
#                 rows_from = rows_from.query(config["all_query"])
#                 rows_to = rows_to.query(config["all_query"])

#             inflection_from = rows_from.inflection.iloc[0]
#             inflection_to = rows_to.inflection.iloc[0]
#         except IndexError:
#             continue

#         if len(rows_from) == 0 or len(rows_to) == 0:
#             continue

#         # prepare equivalences for 'from' and 'to' groups.
#         # equivalences define the set of instances over which we can average representations
#         # before computing the analogy.
#         if "equivalence_keys" in config:
#             from_equivalence_keys = config["equivalence_keys"]
#             to_equivalence_keys = config["equivalence_keys"]
#         else:
#             from_equivalence_keys = ["base", "inflected"]
#             to_equivalence_keys = ["base", "inflected"]

#         # we must group on at least the forms themselves
#         assert set(["base", "inflected"]) <= set(from_equivalence_keys)
#         assert set(["base", "inflected"]) <= set(to_equivalence_keys)

#         from_equiv = rows_from.groupby(from_equivalence_keys)
#         to_equiv = rows_to.groupby(to_equivalence_keys)
#         from_equiv_labels = [k for k, count in from_equiv.size().items() if count >= 1]
#         to_equiv_labels = [k for k, count in to_equiv.size().items() if count >= 1]

#         if len(set(from_equiv_labels) | set(to_equiv_labels)) <= 1:
#             # not enough labels to support transfer.
#             L.warning(f"Skipping {group} due to insufficient labels")
#             continue

#         # sample pairs of base forms
#         from_samples, to_samples = sample_non_identical_pairs(from_equiv_labels, to_equiv_labels, num_samples)

#         for from_equiv_label_i, to_equiv_label_i in zip(tqdm(from_samples, leave=False), to_samples):
#             rows_from_i = from_equiv.get_group(tuple(from_equiv_label_i))
#             rows_to_i = to_equiv.get_group(tuple(to_equiv_label_i))

#             base_from_label = rows_from_i.base.iloc[0]
#             base_to_label = rows_to_i.base.iloc[0]
#             base_to_idx = rows_to_i.base_idx.iloc[0]
#             inflected_to_label = rows_to_i.inflected.iloc[0]
#             inflected_to_idx = rows_to_i.inflected_idx.iloc[0]

#             # sample pairs for comparison across the two forms
#             n = min(max_num_vector_samples, max(len(rows_from_i), len(rows_to_i)))
#             if len(rows_from_i) < n:
#                 rows_from_i = rows_from_i.sample(n, replace=True)
#             elif len(rows_from_i) > n:
#                 rows_from_i = rows_from_i.sample(n, replace=False)

#             if len(rows_to_i) < n:
#                 rows_to_i = rows_to_i.sample(n, replace=True)
#             elif len(rows_to_i) > n:
#                 rows_to_i = rows_to_i.sample(n, replace=False)

#             tensor_fn = torch.tensor if method == "torch" else np.array
#             from_inflected_flat_idx = tensor_fn(
#                 [flat_idx_lookup[(row.inflected_idx, row.inflected_instance_idx)]
#                  for _, row in rows_from_i.iterrows()])
#             from_base_flat_idx = tensor_fn(
#                 [flat_idx_lookup[(row.base_idx, row.base_instance_idx)]
#                  for _, row in rows_from_i.iterrows()])
#             to_base_flat_idx = tensor_fn(
#                 [flat_idx_lookup[(row.base_idx, row.base_instance_idx)]
#                  for _, row in rows_to_i.iterrows()])

#             # Critical analogy logic
#             pair_difference = agg[from_inflected_flat_idx] - agg[from_base_flat_idx]
#             pair_base = agg[to_base_flat_idx]
            
#             pair_predicted = pair_base + pair_difference

#             norm_fn = partial(torch.norm, dim=1, keepdim=True) \
#                 if method == "torch" else partial(np.linalg.norm, axis=1, keepdims=True)
#             pair_predicted /= norm_fn(pair_predicted)

#             references, references_src = agg, agg_src
#             if method == "torch":
#                 with torch.no_grad():
#                     dists = 1 - nxn_cos_sim(pair_predicted, references)
#                     # mean over instances of pair
#                     dists = dists.mean(0)
#             else:
#                 dists = 1 - fastdist.cosine_matrix_to_matrix(pair_predicted, references)
#                 dists = dists.mean(0)
#             ranks = dists.argsort()

#             where_fn = torch.where if method == "torch" else np.where
#             if exclude_base_from_predictions:
#                 base_flat_idxs = where_fn(references_src[:, 0] == base_to_idx)[0]
#                 isin_fn = torch.isin if method == "torch" else np.isin
#                 ranks = ranks[~isin_fn(ranks, base_flat_idxs)]

#             gt_rank = where_fn(references_src[ranks, 0] == inflected_to_idx)[0][0].item()
#             gt_distance = dists[gt_rank].item()

#             if verbose:
#                 for dist, (label_idx, instance_idx, _) in zip(dists[ranks[:5]], references_src[ranks[:5]]):
#                     print(f"{group} {from_equiv_label_i} -> {to_equiv_label_i}: {state_space_spec.labels[label_idx]} {instance_idx}")

#             nearest_neighbor = references_src[ranks[0]]
#             results.append({
#                 "group": group,
#                 "base_from": base_from_label,
#                 "base_to": base_to_label,
#                 "inflection_from": inflection_from,
#                 "inflection_to": inflection_to,
#                 "predicted_label_idx": nearest_neighbor[0].item(),
#                 "predicted_label": state_space_spec.labels[nearest_neighbor[0]],
#                 "predicted_instance_idx": nearest_neighbor[1].item(),
#                 "gt_label": inflected_to_label,
#                 "gt_label_idx": inflected_to_idx,
#                 "gt_label_rank": gt_rank,
#                 "gt_distance": gt_distance,
#             })

#     return pd.DataFrame(results)


# def run_experiment_type_level(experiment_name, config, verbose=False,
#                    num_samples=100,
#                    exclude_base_from_predictions=True):
#     results = []
#     if "group_by" in config:
#         grouper = inflection_results_df.reset_index().groupby(config["group_by"])
#     else:
#         grouper = [("", inflection_results_df.reset_index())]
#     for group, rows in tqdm(grouper, leave=False):
#         if verbose:
#             print("=======")
#             print(group)

#         try:
#             if "base_query" in config:
#                 rows_from = rows.query(config["base_query"])
#             else:
#                 rows_from = rows
#             if "inflected_query" in config:
#                 rows_to = rows.query(config["inflected_query"])
#             else:
#                 rows_to = rows

#             if len(rows_from) == 0 or len(rows_to) == 0:
#                 continue

#             all_base_from = rows_from.base.unique()
#             inflection_from = rows_from.inflection.iloc[0]
#             all_base_to = rows_to.base.unique()
#             inflection_to = rows_to.inflection.iloc[0]
#         except IndexError:
#             continue

#         # sample pairs of base forms
#         all_base_pairs = list(itertools.product(all_base_from, all_base_to))
#         all_base_pairs = [all_base_pairs[idx] for idx in np.random.choice(len(all_base_pairs), num_samples, replace=True)]

#         for base_from, base_to in tqdm(all_base_pairs, leave=False):
#             pair_difference = difference_representations[inflection_from][base_from]
#             pair_base = base_representations[inflection_to][base_to]
#             pair_n = min(pair_difference.shape[0], pair_base.shape[0])

#             pair_difference = pair_difference[:pair_n]
#             pair_base = pair_base[:pair_n]

#             pair_predicted = pair_base + pair_difference
#             pair_predicted /= np.linalg.norm(pair_predicted, axis=1, keepdims=True)

#             references, references_src = trajectory_aggs_flat["mean"]
#             dists = 1 - fastdist.cosine_matrix_to_matrix(pair_predicted, references)
#             # mean over instances of pair
#             dists = dists.mean(0)
#             ranks = dists.argsort()

#             if exclude_base_from_predictions:
#                 base_to_idx = state_space_spec.labels.index(base_to)
#                 base_flat_idxs = np.nonzero(references_src[:, 0] == base_to_idx)[0]
#                 ranks = ranks[~np.isin(ranks, base_flat_idxs)]

#             gt_inflected_label = rows_to.query("base == @base_to").inflected.iloc[0]
#             gt_inflected_label_idx = state_space_spec.labels.index(gt_inflected_label)
#             gt_rank = np.where(references_src[ranks, 0] == gt_inflected_label_idx)[0][0]
#             gt_distance = dists[ranks[gt_rank]]

#             if verbose:
#                 k = 5
#                 sorted_references = references_src[ranks]
#                 for dist, (label_idx, instance_idx, _) in zip(dists[ranks[:k]], sorted_references[:k]):
#                     print(f"{group} {base_from} -> {base_to}: {state_space_spec.labels[label_idx]} {dist:.3f}")
#                 if not (sorted_references[:k] == gt_inflected_label_idx).any():
#                     # not in top k. Find the first.
#                     first_match = np.where(sorted_references[:, 0] == gt_inflected_label_idx)[0][0]
#                     print(f"\t-- First match at {first_match} {state_space_spec.labels[sorted_references[first_match, 0]]} {dists[first_match]:.3f}")

#             nearest_neighbor = references_src[ranks[0]]
#             results.append({
#                 "group": group,
#                 "base_from": base_from,
#                 "base_to": base_to,
#                 "inflection_from": inflection_from,
#                 "inflection_to": inflection_to,
#                 "predicted_label_idx": nearest_neighbor[0],
#                 "predicted_label": state_space_spec.labels[nearest_neighbor[0]],
#                 "predicted_instance_idx": nearest_neighbor[1],
#                 "gt_label": gt_inflected_label,
#                 "gt_label_idx": state_space_spec.labels.index(gt_inflected_label),
#                 "gt_label_rank": gt_rank,
#                 "gt_distance": gt_distance,
#             })

#     return pd.DataFrame(results)


# def run_experiment_equiv_level_old(experiment_name, config, verbose=False,
#                    num_samples=100,
#                    exclude_base_from_predictions=True):
#     results = []
#     if "group_by" in config:
#         grouper = inflection_instance_df.groupby(config["group_by"])
#     else:
#         grouper = [("", inflection_instance_df)]
#     for group, rows in tqdm(grouper, leave=False):
#         try:
#             if "base_query" in config:
#                 rows_from = rows.query(config["base_query"])
#             else:
#                 rows_from = rows

#             if "inflected_query" in config:
#                 rows_to = rows.query(config["inflected_query"])
#             else:
#                 rows_to = rows

#             inflection_from = rows_from.inflection.iloc[0]
#             inflection_to = rows_to.inflection.iloc[0]
#         except IndexError:
#             raise
#             continue

#         if len(rows_from) == 0 or len(rows_to) == 0:
#             continue

#         # prepare equivalences for 'from' and 'to' groups.
#         # equivalences define the set of instances over which we can average representations
#         # before computing the analogy.
#         from_equivalence_keys = ["base", "inflected", "inflected_phones"]
#         to_equivalence_keys = ["base", "inflected", "inflected_phones"]

#         # we must group on at least the forms themselves
#         assert set(["base", "inflected"]) <= set(from_equivalence_keys)
#         assert set(["base", "inflected"]) <= set(to_equivalence_keys)

#         from_equiv = rows_from.groupby(from_equivalence_keys)
#         to_equiv = rows_to.groupby(to_equivalence_keys)
#         from_equiv_labels = [k for k, v in from_equiv.groups.items() if len(v) >= 1]
#         to_equiv_labels = [k for k, v in to_equiv.groups.items() if len(v) >= 1]

#         # sample pairs of base forms
#         from_samples = np.random.choice(len(from_equiv_labels), num_samples, replace=True)
#         to_samples = np.random.choice(len(to_equiv_labels), num_samples, replace=True)

#         for from_i, to_i in zip(tqdm(from_samples, leave=False), to_samples):
#             from_equiv_label_i = from_equiv_labels[from_i]
#             to_equiv_label_i = to_equiv_labels[to_i]

#             rows_from_i = from_equiv.get_group(from_equiv_label_i)
#             rows_to_i = to_equiv.get_group(to_equiv_label_i)

#             base_from_label = rows_from_i.base.iloc[0]
#             base_to_label = rows_to_i.base.iloc[0]
#             base_to_idx = rows_to_i.base_idx.iloc[0]
#             inflected_to_label = rows_to_i.inflected.iloc[0]
#             inflected_to_idx = rows_to_i.inflected_idx.iloc[0]

#             pair_difference = difference_representations_tok[rows_from_i.index]
#             pair_base = base_representations[inflection_to][rows_to_i.base.iloc[0]]

#             n = min(pair_difference.shape[0], pair_base.shape[0])
#             pair_difference = pair_difference[:n]
#             pair_base = pair_base[:n]

#             pair_predicted = pair_base + pair_difference
#             pair_predicted /= np.linalg.norm(pair_predicted, axis=1, keepdims=True)

#             references, references_src = trajectory_aggs_flat["mean"]
#             dists = 1 - fastdist.cosine_matrix_to_matrix(pair_predicted, references)
#             # mean over instances of pair
#             dists = dists.mean(0)
#             ranks = dists.argsort()

#             if exclude_base_from_predictions:
#                 base_flat_idxs = np.nonzero(references_src[:, 0] == base_to_idx)[0]
#                 ranks = ranks[~np.isin(ranks, base_flat_idxs)]

#             gt_rank = np.where(references_src[ranks, 0] == inflected_to_idx)[0][0]
#             gt_distance = dists[gt_rank]

#             if verbose:
#                 for dist, (label_idx, instance_idx, _) in zip(dists[ranks[:5]], references_src[ranks[:5]]):
#                     print(f"{group} {from_equiv_label_i} -> {to_equiv_label_i}: {state_space_spec.labels[label_idx]} {instance_idx}")

#             nearest_neighbor = references_src[ranks[0]]
#             results.append({
#                 "group": group,
#                 "base_from": base_from_label,
#                 "base_to": base_to_label,
#                 "inflection_from": inflection_from,
#                 "inflection_to": inflection_to,
#                 "predicted_label_idx": nearest_neighbor[0],
#                 "predicted_label": state_space_spec.labels[nearest_neighbor[0]],
#                 "predicted_instance_idx": nearest_neighbor[1],
#                 "gt_label": inflected_to_label,
#                 "gt_label_idx": inflected_to_idx,
#                 "gt_label_rank": gt_rank,
#                 "gt_distance": gt_distance,
#             })

#     return pd.DataFrame(results)


# def run_experiment_token_level(experiment_name, config, verbose=False,
#                    num_samples=100,
#                    exclude_base_from_predictions=True):
#     results = []
#     if "group_by" in config:
#         grouper = inflection_instance_df.groupby(config["group_by"])
#     else:
#         grouper = [("", inflection_instance_df)]
#     for group, rows in tqdm(grouper, leave=False):
#         try:
#             if "base_query" in config:
#                 rows_from = rows.query(config["base_query"])
#             else:
#                 rows_from = rows

#             if "inflected_query" in config:
#                 rows_to = rows.query(config["inflected_query"])
#             else:
#                 rows_to = rows

#             inflection_from = rows_from.inflection.iloc[0]
#             inflection_to = rows_to.inflection.iloc[0]
#         except IndexError:
#             continue

#         if len(rows_from) == 0 or len(rows_to) == 0:
#             continue

#         # # prepare equivalences for 'from' and 'to' groups.
#         # # equivalences define the set of instances over which we can average representations
#         # # before computing the analogy.
#         # from_equivalence_keys = ["base"]
#         # to_equivalence_keys = ["base"]

#         # from_equiv = rows_from.groupby(from_equivalence_keys)
#         # to_equiv = rows_to.groupby(to_equivalence_keys)
#         # from_equiv_labels = [k for k, v in from_equiv.groups.items() if len(v) >= 1]
#         # to_equiv_labels = [k for k, v in to_equiv.groups.items() if len(v) >= 1]

#         # # sample pairs of base forms
#         # from_samples = np.random.choice(len(from_equiv_labels), num_samples, replace=True)
#         # to_samples = np.random.choice(len(to_equiv_labels), num_samples, replace=True)

#         # for from_i, to_i in zip(tqdm(from_samples, leave=False), to_samples):
#         #     rows_from_i = from_equiv.get_group(from_equiv_labels[from_i])
#         #     rows_to_i = to_equiv.get_group(to_equiv_labels[to_i])
#         #     base_to_idx = rows_to_i.base_idx.iloc[0]

#         #     pair_difference = difference_representations_tok[rows_from_i.index]
#         #     pair_base = base_representations[inflection_to][rows_to_i.base.iloc[0]].mean(0, keepdims=True)

#         rows_from_sample = np.random.choice(len(rows_from), num_samples, replace=False)
#         rows_to_sample = np.random.choice(len(rows_to), num_samples, replace=False)

#         for row_from_idx, row_to_idx in zip(tqdm(rows_from_sample, leave=False), rows_to_sample):
#             row_from = rows_from.iloc[row_from_idx]
#             row_to = rows_to.iloc[row_to_idx]
#             base_to_idx = row_to.base_idx

#             # this is 1/2 token level -- the individual inflected forms
#             # each have their own representation, and are combined with a mean
#             # representation of the new base
#             pair_difference = difference_representations_tok[row_from.name][None, :]
#             pair_base = base_representations[row_to.inflection][row_to.base].mean(0, keepdims=True)

#             # equiv version
#             # pair_difference = difference_representations_tok[rows_from_i.index]
#             # pair_base = base_representations[inflection_to][rows_to_i.base.iloc[0]]
#             # print(pair_difference.shape, pair_base.shape)
#             # print(rows_from_i)
#             # print(rows_to_i)

#             # pair_difference /= np.linalg.norm(pair_difference, axis=1, keepdims=True)
#             # pair_base /= np.linalg.norm(pair_base, axis=1, keepdims=True)
            
#             pair_predicted = (pair_base + pair_difference).mean(0, keepdims=True)
#             pair_predicted /= np.linalg.norm(pair_predicted, axis=1, keepdims=True)

#             references, references_src = trajectory_aggs_flat["mean"]
#             dists = 1 - fastdist.cosine_matrix_to_matrix(pair_predicted, references)
#             # mean over instances of pair
#             dists = dists.mean(0)
#             ranks = dists.argsort()

#             if exclude_base_from_predictions:
#                 base_flat_idxs = np.nonzero(references_src[:, 0] == base_to_idx)[0]
#                 ranks = ranks[~np.isin(ranks, base_flat_idxs)]

#             # gt_inflected_label = rows_to.inflected.iloc[0]
#             # gt_inflected_label_idx = rows_to.inflected_idx.iloc[0]
#             gt_inflected_label = row_to.inflected
#             gt_inflected_label_idx = row_to.inflected_idx
#             gt_rank = np.where(references_src[ranks, 0] == gt_inflected_label_idx)[0][0]
#             gt_distance = dists[gt_rank]

#             if verbose:
#                 for dist, (label_idx, instance_idx, _) in zip(dists[ranks[:5]], references_src[ranks[:5]]):
#                     print(f"{group} {rows_from.base.iloc[0]} -> {row_to.base}: {state_space_spec.labels[label_idx]} {instance_idx}")

#             nearest_neighbor = references_src[ranks[0]]
#             results.append({
#                 "group": group,
#                 "base_from": row_from.base,
#                 "base_to": row_to.base,
#                 "inflection_from": inflection_from,
#                 "inflection_to": inflection_to,
#                 "predicted_label_idx": nearest_neighbor[0],
#                 "predicted_label": state_space_spec.labels[nearest_neighbor[0]],
#                 "predicted_instance_idx": nearest_neighbor[1],
#                 "gt_label": gt_inflected_label,
#                 "gt_label_idx": state_space_spec.labels.index(gt_inflected_label),
#                 "gt_label_rank": gt_rank,
#                 "gt_distance": gt_distance,
#             })

#     return pd.DataFrame(results)