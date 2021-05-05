from itertools import islice, cycle
import pandas as pd


class PopularRecommender:
    def __init__(self, max_K=100, days=30, **kwargs):
        self.max_K = max_K
        self.days = days
        self.recommendations = []

    def fit(self, df):
        min_date = df["start_date"].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.recommendations = (
            df.loc[df["start_date"] > min_date, "item_id"]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def predict(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))
