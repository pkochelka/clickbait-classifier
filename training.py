import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from utils import balance_dataset, run_experiment
from utils_transformer import run_experiment_transformer

EXPERIMENTS = [
    (Ridge(random_state=13), {"regressor__alpha": [0.1, 1.0]}),
    (Lasso(random_state=13), {"regressor__alpha": [0.01, 0.1]}),
    (SVR(), {"regressor__C": [0.1, 1.0], "regressor__kernel": ["linear", "rbf"]}),
    (RandomForestRegressor(random_state=13), {"regressor__n_estimators": [100, 200], "regressor__max_depth": [10, None]}),
    (GradientBoostingRegressor(random_state=13), {"regressor__n_estimators": [100, 200], "regressor__max_depth": [3, None], "regressor__learning_rate": [0.1, 0.001]}),
    (MLPRegressor(random_state=13, max_iter=500), {"regressor__hidden_layer_sizes": [(64,), (128,)], "regressor__alpha": [0.001, 0.01]}),
    (XGBRegressor(random_state=13, n_jobs=-1), {"regressor__n_estimators": [100, 200], "regressor__max_depth": [3, 6], "regressor__learning_rate": [0.1, 0.01]}),
]

TRANSFORMER_EXPERIMENTS = ["distilbert-base-uncased", "roberta-base"]


def load_data(train_path: str, test_path: str) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    df, df_test = pd.read_json(train_path), pd.read_json(test_path)
    return df["postText"], df[["truthMedian", "truthClass"]], df_test["postText"], df_test[["truthMedian", "truthClass"]]


def split(X: pd.Series, y: pd.DataFrame, test_size: float = 0.1) -> tuple:
    return train_test_split(X, y, test_size=test_size, random_state=13, stratify=y["truthClass"])


def run_experiments(X_train: pd.Series, y_train: pd.DataFrame,
                    X_dev: pd.Series,   y_dev: pd.DataFrame,
                    X_test: pd.Series,  y_test: pd.DataFrame,
                    output_path: str) -> pd.DataFrame:
    results = [run_experiment_transformer(transformer_name, X_train, y_train, X_dev, y_dev, X_test, y_test) for transformer_name in TRANSFORMER_EXPERIMENTS] 
    results += [run_experiment(reg, params, X_train, y_train, X_dev, y_dev, X_test, y_test) for reg, params in EXPERIMENTS]
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(df.sort_values(by="f1_macro", ascending=False))
    return df


X, y, X_test, y_test = load_data("data/preprocessed.json", "data/test_preprocessed.json")
X_train, X_dev, y_train, y_dev = split(X, y)
run_experiments(X_train, y_train, X_dev, y_dev, X_test, y_test, "model_comparison_results.csv")

print("\n>>> Starting balanced dataset")
X_bal, y_bal = (df_bal := balance_dataset(pd.concat([X, y], axis=1)))["postText"], df_bal[["truthMedian", "truthClass"]]
X_train_bal, X_dev_bal, y_train_bal, y_dev_bal = split(X_bal, y_bal)
run_experiments(X_train_bal, y_train_bal, X_dev_bal, y_dev_bal, X_test, y_test, "model_comparison_results_balanced.csv")