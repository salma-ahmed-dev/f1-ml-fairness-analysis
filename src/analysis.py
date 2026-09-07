from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "f1_pitstops_2018_2024.csv"
TARGET = "Lap Time Variation"
TOP_TEAMS = {"Mercedes", "Ferrari", "Red Bull"}


def team_tier(team: str) -> str:
    return "Top_Tier" if team in TOP_TEAMS else "Other_Tier"


def load_and_prepare_data(path: Path):
    data = pd.read_csv(path)

    columns = [
        "Stint Length",
        "Air_Temp_C",
        "Tire Compound",
        "Constructor",
        TARGET,
    ]

    data = data[columns].dropna().copy()
    data["Team_Category"] = data["Constructor"].map(team_tier)
    data = data.drop(columns=["Constructor"])

    encoded = pd.get_dummies(
        data,
        columns=["Tire Compound", "Team_Category"],
        drop_first=False,
    )

    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]

    return train_test_split(X, y, test_size=0.30, random_state=10)


def mae_by_group(X_test, y_test, predictions):
    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = predictions
    results["Team_tier"] = np.where(
        results["Team_Category_Top_Tier"] == 1,
        "Top_Tier",
        "Other_Tier",
    )

    scores = {}
    for group in ["Top_Tier", "Other_Tier"]:
        group_rows = results[results["Team_tier"] == group]
        scores[group] = mean_absolute_error(
            group_rows["Actual"], group_rows["Predicted"]
        )

    return scores


def train_unified_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=6,
        min_samples_leaf=20,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mae_by_group(X_test, y_test, predictions)


def train_specialised_models(X_train, X_test, y_train, y_test):
    train = X_train.copy()
    train[TARGET] = y_train

    test = X_test.copy()
    test[TARGET] = y_test

    scores = {}

    for group_name, group_value in [("Top_Tier", 1), ("Other_Tier", 0)]:
        train_group = train[train["Team_Category_Top_Tier"] == group_value]
        test_group = test[test["Team_Category_Top_Tier"] == group_value]

        X_group_train = train_group.drop(columns=[TARGET])
        y_group_train = train_group[TARGET]
        X_group_test = test_group.drop(columns=[TARGET])
        y_group_test = test_group[TARGET]

        model = DecisionTreeRegressor(
            random_state=42,
            max_depth=6,
            min_samples_leaf=20,
        )
        model.fit(X_group_train, y_group_train)
        predictions = model.predict(X_group_test)
        scores[group_name] = mean_absolute_error(y_group_test, predictions)

    return scores


def plot_comparison(unified_scores, specialised_scores):
    groups = ["Top_Tier", "Other_Tier"]
    unified = [unified_scores[group] for group in groups]
    specialised = [specialised_scores[group] for group in groups]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, unified, width, label="Unified")
    plt.bar(x + width / 2, specialised, width, label="Specialised")
    plt.xticks(x, groups)
    plt.xlabel("Team group")
    plt.ylabel("Mean Absolute Error")
    plt.title("Unified and specialised model MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. See data/README.md for the expected file."
        )

    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)

    unified = train_unified_model(X_train, X_test, y_train, y_test)
    specialised = train_specialised_models(X_train, X_test, y_train, y_test)

    unified_gap = abs(unified["Top_Tier"] - unified["Other_Tier"])
    specialised_gap = abs(
        specialised["Top_Tier"] - specialised["Other_Tier"]
    )

    print("Unified model")
    print(f"Top tier MAE:   {unified['Top_Tier']:.4f}")
    print(f"Other tier MAE: {unified['Other_Tier']:.4f}")
    print(f"Group MAE gap:  {unified_gap:.4f}\n")

    print("Specialised models")
    print(f"Top tier MAE:   {specialised['Top_Tier']:.4f}")
    print(f"Other tier MAE: {specialised['Other_Tier']:.4f}")
    print(f"Group MAE gap:  {specialised_gap:.4f}")

    plot_comparison(unified, specialised)


if __name__ == "__main__":
    main()
