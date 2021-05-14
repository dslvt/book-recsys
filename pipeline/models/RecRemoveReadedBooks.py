import pandas as pd


class PopularWithoutReaded:
    def __init__(self, max_K=100, days=30, **kwargs):
        self.max_K = max_K
        self.days = days
        self.users = None
        self.recommendations = {}
        self.merged_users = None

    def fit(self, df, users):
        min_date = df["start_date"].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.users = users
        merged = df.merge(users, on="user_id")
        merged["sex"] = merged["sex"].fillna(0.0)
        merged["age"] = merged["age"].fillna("25_34")
        self.merged_users = merged
        merged = merged.loc[
            merged["start_date"] > min_date, ["item_id", "age", "sex"]
        ].groupby(["age", "sex"])
        for name, group in merged:
            self.recommendations[name] = (
                group["item_id"].value_counts().head(self.max_K).index.values
            )

    def predict(self, users=None, N=10):
        rec = self.recommendations
        merged = self.merged_users

        if users is None:
            return rec[("25_34", 0.0)]

        full_users = users.merge(self.users, how="left", on="user_id")
        full_users["sex"] = full_users["sex"].fillna(0.0)
        full_users["age"] = full_users["age"].fillna("25_34")
        ans = []
        for idx, user in full_users.iterrows():
            usr_id = user["user_id"]
            full_readed = merged[
                (merged["user_id"] == usr_id) & (merged["progress"] == 100)
            ]["item_id"].values
            first_pred = rec[(user["age"], user["sex"])]
            filtered_pred = []
            book_counter = 0
            while len(filtered_pred) != N:
                if not first_pred[book_counter] in full_readed:
                    filtered_pred.append(first_pred[book_counter])
                book_counter += 1

            ans.append(filtered_pred)

        return ans
