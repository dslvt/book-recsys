import pandas as pd


class PopularByGender:
    def __init__(self, max_K=100, days=30, **kwargs):
        self.max_K = max_K
        self.days = days
        self.users = None
        self.recommendations_m = []
        self.recommendations_f = []

    def fit(self, df, users):
        min_date = df["start_date"].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.users = users
        merged = df.merge(users, on="user_id")
        self.recommendations_m = (
            merged[merged["sex"] == 1.0]
            .loc[merged["start_date"] > min_date, "item_id"]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

        self.recommendations_f = (
            merged[merged["sex"] == 0.0]
            .loc[merged["start_date"] > min_date, "item_id"]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def predict(self, users=None, N=10):
        recsm = self.recommendations_m[:N]
        recsf = self.recommendations_f[:N]

        if users is None:
            return recsm

        full_users = users.merge(self.users, how="left", on="user_id").fillna(
            1.0
        )
        ans = []
        for idx, user in full_users.iterrows():
            if user["sex"] == 1.0:
                ans.append(recsm)
            else:
                ans.append(recsf)

        return ans
