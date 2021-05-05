import os
import pandas as pd


def get_data():
    users = pd.read_csv("../../../large_files/users.csv")
    items = pd.read_csv("../../../large_files/items.csv")

    interactions = pd.read_csv("../../../large_files/interactions.csv")
    interactions["start_date"] = pd.to_datetime(interactions["start_date"])

    submission = pd.read_csv("../../../large_files/sample_submission.csv")
    pred_pop = pd.DataFrame({"user_id": submission["Id"].unique()})

    return {
        "items": items,
        "users": users,
        "interactions": interactions,
        "submission": pred_pop,
    }
