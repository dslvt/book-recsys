import pandas as pd


class PopularByAge:
    def __init__(self, max_K=100, days=30, **kwargs):
        self.max_K = max_K
        self.days = days
        self.users = None
        self.recommendations = {}

    def fit(self, df, users):
        min_date = df["start_date"].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.users = users
        merged = df.merge(users, on="user_id")
        merged = (
            merged.loc[merged["start_date"] > min_date, ["item_id", "age"]]
            .fillna("25_34")
            .groupby("age")
        )
        for name, group in merged:
            self.recommendations[name] = (
                group["item_id"].value_counts().head(self.max_K).index.values
            )

    def predict(self, users=None, N=10):
        rec = self.recommendations
        for key, val in rec.items():
            rec[key] = val[:N]

        if users is None:
            return rec["25_34"]

        full_users = users.merge(self.users, how="left", on="user_id").fillna(
            "25_34"
        )
        ans = []
        for idx, user in full_users.iterrows():
            ans.append(rec[user["age"]])

        return ans
