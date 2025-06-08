import itertools
import sys
from typing import Any
import yaml

import papermill

configfile: "config.yaml"

include: "workflows/librispeech/Snakefile"

ruleorder:
    run_no_train > run

wildcard_constraints:
    dataset = r"[a-z0-9_-]+",
    base_model_name = r"[a-z0-9_-]+",
#     feature_sets = r"[a-z0-9_]+",


# Notebooks to run for intrinsic analysis on models
ALL_MODEL_NOTEBOOKS = [
    "lexical_coherence",
    "lexical_coherence_for_aggregation",
    "line_search",
    "syllable_coherence",
    "syllable_coherence_by_position",
    "phoneme_coherence",
    "phoneme_coherence_by_position",
    "temporal_generalization_word",
    "temporal_generalization_phoneme",
    "predictions",
    "predictions_word",
    "rsa_phoneme",
    "state_space",
    "trf",
    "within_word_gradience",
    "word_discrimination",

    "word_boundary",
    "syllable_boundary",

    # "geometry/analogy",
    # "geometry/analogy_dynamic",
]


DEFAULT_PHONEME_EQUIVALENCE = "phoneme_10frames"


def select_gpu_device(wildcards, resources):
    if resources.gpu == 0:
        return None
    import GPUtil
    available_l = GPUtil.getAvailable(
        order = 'random', limit = resources.gpu,
        maxLoad = 0.01, maxMemory = 0.49, includeNan=False,
        excludeID=[], excludeUUID=[])
    available_str = ",".join([str(x) for x in available_l])

    if len(available_l) == 0 and resources.gpu > 0:
        raise Exception("select_gpu_device did not select any GPUs")
    elif len(available_l) < resources.gpu:
        sys.stderr.write("[WARN] select_gpu_device selected fewer GPU devices than requested")
    print("Assigning %d available GPU devices: %s" % (resources.gpu, available_str))
    return available_str


def hydra_param(obj):
    """
    Prepare the given object for use as a Hydra CLI / YAML override.
    """
    if isinstance(obj, snakemake.io.Namedlist):
        obj = list(obj)
    return yaml.safe_dump(obj, default_flow_style=True, width=float("inf")).strip()

def join_hydra_overrides(overrides: dict[str, Any]):
    return " ".join([f"+{k}={v}" for k, v in overrides.items()])


rule preprocess:
    input:
        timit_raw = lambda wildcards: config["datasets"].get(wildcards.dataset, {"raw_path": []})["raw_path"],
        script = "notebooks/preprocessing/{dataset}.ipynb"

    output:
        data_path = directory("outputs/preprocessed_data/{dataset}"),
        notebook_path = "outputs/preprocessing/{dataset}/{dataset}.ipynb"

    shell:
        """
        papermill --log-output {input.script} \
            {output.notebook_path} \
            -p base_dir {workflow.basedir} \
            -p dataset_path {input.timit_raw} \
            -p dataset_name {wildcards.dataset} \
            -p out_path {output.data_path}
        """


rule extract_hidden_states:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        base_model_config = "conf/base_model/{base_model_name}.yaml"

    resources:
        gpu = 1

    output:
        "outputs/hidden_states/{base_model_name}/{dataset}.h5"

    run:
        outdir = Path(output[0]).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_hidden_states.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={output} \
            dataset.processed_data_dir={input.dataset}
        """)


rule prepare_equivalence_dataset:
    input:
        timit_data = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml"

    output:
        "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        python scripts/make_equivalence_dataset.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={wildcards.equivalence_classer} \
            dataset.processed_data_dir={input.timit_data}
        """)


def _get_equivalence_dataset(dataset: str, base_model_name: str, equivalence_classer: str) -> str:
    if equivalence_classer == "random":
        # default to phoneme-level
        return f"outputs/equivalence_datasets/{dataset}/{base_model_name}/{DEFAULT_PHONEME_EQUIVALENCE}/equivalence.pkl"
    else:
        return f"outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"


def get_equivalence_dataset(wildcards):
    # if we have a target_dataset, we should be retrieving equivalences for there
    return _get_equivalence_dataset(getattr(wildcards, "target_dataset", wildcards.dataset),
                                    wildcards.base_model_name, wildcards.equivalence_classer)


rule run:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml",
        model_config = "conf/model/{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}")

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            dataset.processed_data_dir={input.dataset} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            model={wildcards.model_name} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)


rule tune_hparam:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = "conf/equivalence/{equivalence_classer}.yaml",
        model_config = "conf/model/{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = "outputs/equivalence_datasets/{dataset}/{base_model_name}/{equivalence_classer}/equivalence.pkl"

    resources:
        gpu = 1

    output:
        full_trace = directory("outputs/hparam_search/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}")

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            trainer.mode=hyperparameter_search \
            dataset.processed_data_dir={input.dataset} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            model={wildcards.model_name} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)


# Run train without actually training -- used to generate random model weights
NO_TRAIN_DEFAULT_EQUIVALENCE = DEFAULT_PHONEME_EQUIVALENCE
rule run_no_train:
    input:
        base_model_config = "conf/base_model/{base_model_name}.yaml",
        equivalence_config = f"conf/equivalence/{NO_TRAIN_DEFAULT_EQUIVALENCE}.yaml",
        model_config = "conf/model/random{model_name}.yaml",

        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5",
        equivalence_dataset = f"outputs/equivalence_datasets/{{dataset}}/{{base_model_name}}/{NO_TRAIN_DEFAULT_EQUIVALENCE}/equivalence.pkl"

    output:
        full_trace = directory("outputs/models/{dataset}/{base_model_name}/random{model_name}/random")

    shell:
        """
        export HDF5_USE_FILE_LOCKING=FALSE
        python train_decoder.py \
            hydra.run.dir={output.full_trace} \
            dataset.processed_data_dir={input.dataset} \
            model=random{wildcards.model_name} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={NO_TRAIN_DEFAULT_EQUIVALENCE} \
            +equivalence.path={input.equivalence_dataset} \
            trainer.mode=no_train \
            device=cpu
        """


MODEL_SPEC_LIST = [f"{m['dataset']}/{m['base_model']}/{m['model']}/{m['equivalence']}" for m in config["models"]]
rule run_all:
    input:
        expand("outputs/models/{model_spec}", model_spec=MODEL_SPEC_LIST)


rule extract_embeddings:
    input:
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{target_dataset}.h5",
        equivalence_dataset = get_equivalence_dataset

    resources:
        gpu = 1

    output:
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}.npy"

    run:
        outdir = Path(output.embeddings).parent
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python scripts/extract_model_embeddings.py \
            hydra.run.dir={outdir} \
            model={wildcards.model_name} \
            +model.output_dir={input.model_dir} \
            +model.embeddings_path={output.embeddings} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            equivalence={wildcards.equivalence_classer} \
            +equivalence.path={input.equivalence_dataset}
        """)



EMBEDDING_LIST = [f"{m['dataset']}/{m['base_model']}/{m['model']}/{m['equivalence']}/{m['dataset']}.npy" for m in config["models"]]
rule extract_all_embeddings:
    input:
        expand("outputs/model_embeddings/{embedding}",
                embedding=EMBEDDING_LIST)

rule compute_state_spaces:
    input:
        dataset = "outputs/preprocessed_data/{dataset}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{dataset}.h5"

    output:
        "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.h5"

    run:
        outdir = Path(output[0]).parent

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        python scripts/generate_state_space_specs.py \
            hydra.run.dir={outdir} \
            base_model={wildcards.base_model_name} \
            dataset.processed_data_dir={input.dataset} \
            +base_model.hidden_state_path={input.hidden_states} \
            +analysis.state_space_specs_path={output[0]}
        """)


rule evaluate_word_recognition:
    input:
        model_dir = "outputs/models/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}",
        hidden_states = "outputs/hidden_states/{base_model_name}/{target_dataset}.h5",
        embeddings = "outputs/model_embeddings/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}.npy",
        state_space_specs = "outputs/state_space_specs/{dataset}/{base_model_name}/state_space_specs.h5",

        model_config = "conf/recognition_model/{recognition_model}.yaml"

    resources:
        gpu = 1

    output:
        trace = directory("outputs/word_recognition/{dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{target_dataset}/{recognition_model}"),

    run:
        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export PYTHONPATH=`pwd`
        export HDF5_USE_FILE_LOCKING=FALSE
        export CUDA_VISIBLE_DEVICES={gpu_device}
        python word_recognition.py \
            hydra.run.dir={output.trace} \
            recognition_model={wildcards.recognition_model} \
            model={wildcards.model_name} \
            +model.output_dir={input.model_dir} \
            +model.embeddings_path={input.embeddings} \
            base_model={wildcards.base_model_name} \
            +base_model.hidden_state_path={input.hidden_states} \
            +analysis.state_space_specs_path={input.state_space_specs} \
        """)


rule run_all_word_recognition:
    input:
        expand("outputs/word_recognition/{model_spec}/librispeech-train-clean-100/linear",
                model_spec=MODEL_SPEC_LIST)


rule prepare_analogy_inputs:
    input:
        notebook = "notebooks/analogy/prepare_inputs.ipynb",
        state_space_specs = "outputs/state_space_specs/{dataset}/{base_model_class}_8/state_space_specs.h5",

    output:
        outdir = directory("outputs/analogy/inputs/{dataset}/{base_model_class}"),
        notebook = "outputs/analogy/inputs/{dataset}/{base_model_class}/prepare_inputs.ipynb",
        state_space_spec = "outputs/analogy/inputs/{dataset}/{base_model_class}/state_space_spec.h5",

        inflection_results = "outputs/analogy/inputs/{dataset}/{base_model_class}/inflection_results.parquet",
        inflection_instances = "outputs/analogy/inputs/{dataset}/{base_model_class}/inflection_instances.parquet",
        all_cross_instances = "outputs/analogy/inputs/{dataset}/{base_model_class}/all_cross_instances.parquet",
        most_common_allomorphs = "outputs/analogy/inputs/{dataset}/{base_model_class}/most_common_allomorphs.csv",
        false_friends = "outputs/analogy/inputs/{dataset}/{base_model_class}/false_friends.csv",

    shell:
        """
        export HDF5_USE_FILE_LOCKING=FALSE
        papermill --autosave-cell-every 30 --log-output \
            {input.notebook} {output.notebook} \
            -p output_dir {output.outdir} \
            -p state_space_specs_path {input.state_space_specs}
        """


def _compute_analogy_inputs(wildcards, pseudocausal=False, identity=True, flat=True):
    base_model_class = re.sub(r"_[0-9]+$", "", wildcards.base_model_name)
    ret = {
        "notebook": "notebooks/analogy/run_pseudocausal.ipynb" if pseudocausal \
            else "notebooks/analogy/run.ipynb",

        "hidden_states": f"outputs/hidden_states/{wildcards.base_model_name}/{wildcards.dataset}.h5",
        "state_space_specs": f"outputs/analogy/inputs/{wildcards.dataset}/{base_model_class}/state_space_spec.h5",

        "inflection_results": f"outputs/analogy/inputs/{wildcards.dataset}/{base_model_class}/inflection_results.parquet",
        "all_cross_instances": f"outputs/analogy/inputs/{wildcards.dataset}/{base_model_class}/all_cross_instances.parquet",
        "most_common_allomorphs": f"outputs/analogy/inputs/{wildcards.dataset}/{base_model_class}/most_common_allomorphs.csv",
        "false_friends": f"outputs/analogy/inputs/{wildcards.dataset}/{base_model_class}/false_friends.csv",
    }

    if not identity:
        ret["embeddings"] = f"outputs/model_embeddings/{wildcards.train_dataset}/{wildcards.base_model_name}/{wildcards.model_name}/{wildcards.equivalence_classer}/{wildcards.dataset}.npy"

    if flat:
        return list(ret.values())
    else:
        return ret


rule run_analogy_experiment:
    input:
        lambda wildcards: _compute_analogy_inputs(wildcards, identity=False)

    output:
        outdir = directory("outputs/analogy/runs/{train_dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{dataset}"),
        notebook = "outputs/analogy/runs/{train_dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{dataset}/run.ipynb",
        results = "outputs/analogy/runs/{train_dataset}/{base_model_name}/{model_name}/{equivalence_classer}/{dataset}/experiment_results.csv",

    resources:
        gpu = 1

    run:
        # HACK reconstruct inputs in a way that allows us to index them sensibly ><
        inputs = _compute_analogy_inputs(wildcards, identity=False, flat=False)

        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE
        papermill --autosave-cell-every 30 --log-output \
            {inputs[notebook]} {output.notebook} \
            -p output_dir {output.outdir} \
            -p hidden_states_path {inputs[hidden_states]} \
            -p state_space_specs_path {inputs[state_space_specs]} \
            -p embeddings_path {inputs[embeddings]} \
            -p inflection_results_path {inputs[inflection_results]} \
            -p all_cross_instances_path {inputs[all_cross_instances]} \
            -p most_common_allomorphs_path {inputs[most_common_allomorphs]} \
            -p false_friends_path {inputs[false_friends]}
        """)


rule run_analogy_experiment_identity:
    input:
        lambda wildcards: _compute_analogy_inputs(wildcards, identity=True)

    output:
        outdir = directory("outputs/analogy/runs_id/{dataset}/{base_model_name}/"),
        notebook = "outputs/analogy/runs_id/{dataset}/{base_model_name}/run.ipynb",
        results = "outputs/analogy/runs_id/{dataset}/{base_model_name}/experiment_results.csv",

    resources:
        gpu = 1

    run:
        # HACK reconstruct inputs in a way that allows us to index them sensibly ><
        inputs = _compute_analogy_inputs(wildcards, identity=True, flat=False)

        gpu_device = select_gpu_device(wildcards, resources)

        shell("""
        export CUDA_VISIBLE_DEVICES={gpu_device}
        export HDF5_USE_FILE_LOCKING=FALSE

        papermill --autosave-cell-every 30 --log-output \
            {inputs[notebook]} {output.notebook} \
            -p output_dir {output.outdir} \
            -p hidden_states_path {inputs[hidden_states]} \
            -p state_space_specs_path {inputs[state_space_specs]} \
            -p embeddings_path ID \
            -p inflection_results_path {inputs[inflection_results]} \
            -p all_cross_instances_path {inputs[all_cross_instances]} \
            -p most_common_allomorphs_path {inputs[most_common_allomorphs]} \
            -p false_friends_path {inputs[false_friends]}
        """)


base_models = [f"w2v2_{i}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
rule run_all_analogy_experiments:
    input:
        # w2v2 hidden state
        expand("outputs/analogy/runs_id/librispeech-train-clean-100/{base_model}",
                base_model=base_models),
        # contrastive ff
        expand("outputs/analogy/runs/librispeech-train-clean-100/{base_model}/ffff_32-pc-mAP1/word_broad_10frames_fixedlen25/librispeech-train-clean-100",
               base_model=base_models),