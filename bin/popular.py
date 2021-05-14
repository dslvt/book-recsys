import sys
from omegaconf import DictConfig
import hydra

sys.path.append(".")
from pipeline.models.PopularRecommender import PopularRecommender
from pipeline.models.RecommenderByGender import PopularByGender
from pipeline.models.RecommenderByAge import PopularByAge
from pipeline.models.RecommenderWithAgeDistribution import (
    PopularWithAgeDistribution,
)
from pipeline.models.Stub import Stub
from pipeline.data.poprec_sample import get_data


def get_model(kind="Stub", **kwargs):
    if kind == "PopularRecommenderAll":
        return PopularRecommender(**kwargs)
    elif kind == "RecommenderByGender":
        return PopularByGender(**kwargs)
    elif kind == "RecommenderByAge":
        return PopularByAge(**kwargs)
    elif kind == "PopularWithAgeDistribution":
        return PopularWithAgeDistribution(**kwargs)
    else:
        return Stub


@hydra.main(config_path="../configs")
def app(cfg: DictConfig):
    data = get_data()

    model = get_model(**cfg.model)
    model.fit(data["interactions"], users=data["users"])

    pred_pop = data["submission"]
    pred_pop["item_id"] = model.predict(pred_pop, N=10)
    pred_pop["item_id"] = pred_pop["item_id"].apply(
        lambda x: " ".join([str(it) for it in x])
    )
    pred_pop.columns = ["Id", "Predicted"]
    pred_pop.to_csv("submission_pop.csv", index=False)


if __name__ == "__main__":
    app()
