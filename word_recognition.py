import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from src.analysis.state_space import StateSpaceAnalysisSpec
from src.datasets.speech_equivalence import SpeechHiddenStateDataset
from src.word_recognition import train


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()