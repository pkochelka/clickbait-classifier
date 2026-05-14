import pandas as pd

df_train_instances = pd.read_json("data/train/instances.jsonl", lines=True)
df_train_instances.set_index("id" ,inplace=True)

df_train_labels = pd.read_json("data/train/truth.jsonl", lines=True)
df_train_labels.set_index("id" ,inplace=True)

df_test_instances = pd.read_json("data/test/instances.jsonl", lines=True)
df_test_instances.set_index("id" ,inplace=True)
df_test_labels = pd.read_json("data/test/truth.jsonl", lines=True)
df_test_labels.set_index("id" ,inplace=True)

df_combined = df_train_instances.join(df_train_labels)

df_test = df_test_instances.join(df_test_labels)[["postText", "truthMedian", "truthClass"]].copy()
df_test['postText'] = df_test['postText'].apply(lambda x: "\n".join(x) if isinstance(x, list) else x)

# Keep only the columns we'll use
df = df_combined[["postText", "truthMedian", "truthClass"]].copy()
df['postText'] = df['postText'].apply(lambda x: "\n".join(x) if isinstance(x, list) else x)
df = df[df['postText'].str.len() > 5].copy()

df.to_json("data/preprocessed.json")
df_test.to_json("data/test_preprocessed.json")