import hydra
from omegaconf import DictConfig

from src.train import train


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()