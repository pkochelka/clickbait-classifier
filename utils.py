import optuna
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, root_mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def balance_dataset(df: pd.DataFrame, label_col: str = 'truthClass') -> pd.DataFrame:
    df_clickbait = df[df[label_col] == 'clickbait']
    df_no_clickbait = df[df[label_col] == 'no-clickbait']
    min_size = min(len(df_clickbait), len(df_no_clickbait))
    df_cb_down = df_clickbait.sample(n=min_size, random_state=13)
    df_ncb_down = df_no_clickbait.sample(n=min_size, random_state=13)

    df_balanced = pd.concat([df_cb_down, df_ncb_down]).sample(frac=1, random_state=13)
    return df_balanced


def train_and_tune(regressor: BaseEstimator, param_grid,
                   X_train: pd.Series, y_train: pd.DataFrame,
                   X_dev: pd.Series, y_dev: pd.DataFrame,
                   n_trials: int = 5) -> tuple[Pipeline, dict]:
    tfidf_param_grid = {
        'tfidf__max_features': [10000, 12000, 14000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
    }
    all_params = param_grid | tfidf_param_grid

    def objective(trial: optuna.Trial) -> float:
        suggested = {key: trial.suggest_categorical(key, vals) for key, vals in all_params.items()}

        tfidf_params = {k.removeprefix('tfidf__'): v for k, v in suggested.items() if k.startswith('tfidf__')}
        reg_params = {k.removeprefix('regressor__'): v for k, v in suggested.items() if k.startswith('regressor__')}

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('regressor', clone(regressor).set_params(**reg_params)),
        ])
        pipeline.fit(X_train, y_train['truthMedian'])
        predictions = pipeline.predict(X_dev).clip(0, 1)
        return root_mean_squared_error(y_dev['truthMedian'], predictions)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    tfidf_params = {k.removeprefix('tfidf__'): v for k, v in best_params.items() if k.startswith('tfidf__')}
    reg_params = {k.removeprefix('regressor__'): v for k, v in best_params.items() if k.startswith('regressor__')}

    best_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('regressor', clone(regressor).set_params(**reg_params)),
    ])
    best_pipeline.fit(X_train, y_train['truthMedian'])

    predictions = best_pipeline.predict(X_dev).clip(0, 1)
    rmse = root_mean_squared_error(y_dev['truthMedian'], predictions)
    print(f"Best Params: {best_params}, regression RMSE: {rmse:.2f}")
    return best_pipeline, best_params


def run_experiment(regressor: BaseEstimator, params,
                   X_train: pd.Series, y_train: pd.DataFrame,
                   X_dev: pd.Series, y_dev: pd.DataFrame,
                   X_test: pd.Series, y_test: pd.DataFrame) -> dict:
    class_mapping = {'no-clickbait': 0, 'clickbait': 1}
    y_test_binary = y_test['truthClass'].map(class_mapping)

    best_pipeline, best_params = train_and_tune(regressor, params, X_train, y_train, X_dev, y_dev)

    predictions = best_pipeline.predict(X_test).clip(0, 1)
    binary_predictions = (predictions >= 0.5).astype(int)

    rmse = root_mean_squared_error(y_test['truthMedian'], predictions)
    accuracy = accuracy_score(y_test_binary, binary_predictions)
    f1_macro = f1_score(y_test_binary, binary_predictions, average='macro')
    f1_per_class = f1_score(y_test_binary, binary_predictions, average=None)

    return {
        'model_name': regressor.__class__.__name__,
        'rmse': round(rmse, 2),
        'accuracy': round(accuracy, 2),
        'f1_macro': round(f1_macro, 2),
        'f1_no_clickbait': round(f1_per_class[0], 2),
        'f1_clickbait': round(f1_per_class[1], 2),
        'best_params': str(best_params),
    }
