import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def prepare_data(df):
    X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
    y = df["Exited"]

    X["AgeGroup"] = pd.cut(
        X["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=False
    )

    X["ProductsRisk"] = (X["NumOfProducts"] >= 3).astype(int)

    X["Inactive"] = (X["IsActiveMember"] == 0).astype(int)

    return X, y


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Geography", "Gender"]
            )
        ],
        remainder="passthrough"
    )