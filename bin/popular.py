import sys
from omegaconf import DictConfig
import hydra

sys.path.append(".")
from pipeline.models.PopularRecommender import PopularRecommender
from pipeline.models.Stub import Stub


def get_model(kind="Stub", **kwargs):
    if kind == "PopularRecommenderAll":
        return PopularRecommender
    else:
        return Stub


@hydra.main(config_path="../configs")
def app(cfg: DictConfig):
    model = get_model(cfg.model.kind)
    print(model)


if __name__ == "__main__":
    app()
