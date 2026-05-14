from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error
import pandas as pd


class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_experiment_transformer(
    model_name: str,
    X_train: pd.Series, y_train: pd.DataFrame,
    X_dev: pd.Series, y_dev: pd.DataFrame,
    X_test: pd.Series, y_test: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    output_dir: str = "./transformer_output",
) -> dict:
    class_mapping = {'no-clickbait': 0, 'clickbait': 1}

    y_train_binary = y_train['truthClass'].map(class_mapping).tolist()
    y_dev_binary = y_dev['truthClass'].map(class_mapping).tolist()
    y_test_binary = y_test['truthClass'].map(class_mapping).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = ClickbaitDataset(X_train, y_train_binary, tokenizer, max_length)
    dev_dataset = ClickbaitDataset(X_dev, y_dev_binary, tokenizer, max_length)
    test_dataset = ClickbaitDataset(X_test, y_test_binary, tokenizer, max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    binary_predictions = np.argmax(predictions.predictions, axis=-1)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    rmse = root_mean_squared_error(y_test['truthMedian'], probs)

    accuracy = accuracy_score(y_test_binary, binary_predictions)
    f1_macro = f1_score(y_test_binary, binary_predictions, average='macro')
    f1_per_class = f1_score(y_test_binary, binary_predictions, average=None)

    return {
        'model_name': model_name,
        'rmse': round(rmse, 2),
        'accuracy': round(accuracy, 2),
        'f1_macro': round(f1_macro, 2),
        'f1_no_clickbait': round(f1_per_class[0], 2),
        'f1_clickbait': round(f1_per_class[1], 2),
        'best_params': str({'num_epochs': num_epochs, 'batch_size': batch_size, 'lr': learning_rate}),
    }