from argparse import ArgumentParser
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch

from src.datasets.speech_equivalence import SpeechEquivalenceDataset, SpeechHiddenStateDataset
from src.models import get_best_checkpoint
from src.models.integrator import compute_embeddings, ContrastiveEmbeddingModel


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    checkpoint_dir = get_best_checkpoint(config.model.output_dir)
    model: ContrastiveEmbeddingModel = ContrastiveEmbeddingModel.from_pretrained(checkpoint_dir)  # type: ignore
    model.eval()
    model = model.to(config.device)

    hidden_states = SpeechHiddenStateDataset.from_hdf5(config.base_model.hidden_state_path)

    equiv_dataset_path = config.equivalence.path
    with open(equiv_dataset_path, "rb") as f:
        equiv_dataset: SpeechEquivalenceDataset = torch.load(f)

    embeddings = compute_embeddings(model, equiv_dataset, hidden_states, batch_size=96)
    embeddings = embeddings.cpu().numpy()
    
    np.save(config.model.embeddings_path, embeddings)


if __name__ == "__main__":
    main()