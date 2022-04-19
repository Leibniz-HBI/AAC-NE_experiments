# %%
import pandas as pd

# %%
# split a dev set from the training data
train_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_train.csv", header=None, names=['text', 'labels'])
test_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_test.csv", header=None, names=['text', 'labels'])
dev_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_dev.csv", header=None, names=['text', 'labels'])

# %%
all_train_labels = train_df.labels.tolist()
unique_labels = list(set(all_train_labels))
label2int = {label:i for i, label in enumerate(unique_labels)}
int2label = {i:label for i, label in enumerate(unique_labels)}


# %%
train_df.labels.value_counts()

# %%
# split into 0-shot, few-shot datasets
n_trials = 5
n_shots = [0, 1, 2, 4, 8, 10, 50, 100, 209]

# %%
# 1) pretain TARS on n-1 labels complete data
# 2) ft TARS on n-th label few data points
fs_training_data = {}
for label in unique_labels:
    fs_training_data[label] = {"pre":train_df[train_df.labels!=label], "ft":[]}
    # perform pretraining here
    for n_shot in n_shots:
        for trial in range(1,n_trials+1):
            if trial > 1 and n_shot == 0:
                continue
            seed = trial*n_shot
            idx = train_df.labels==label
            sample_ft =  train_df[idx].sample(n=min(n_shot, sum(idx)), random_state=seed)
            fs_training_data[label]["ft"].append({"n_shot":n_shot, "trial":trial, "data":sample_ft})

# %%
import pickle
with open("data_samples.pkl", "wb") as f:
    pickle.dump(fs_training_data, f)

# %%
from flair.datasets import SentenceDataset
from flair.data import Corpus, Sentence

def get_flair_dataset_from_dataframe(data, text_col, label_col):
    sentences = list(data.apply(lambda row: Sentence(row[text_col]).add_label('class', row[label_col]), axis=1))
    return SentenceDataset(sentences)

dev_dataset = get_flair_dataset_from_dataframe(dev_df, "text", "labels")
test_dataset = get_flair_dataset_from_dataframe(test_df, "text", "labels")

# %%
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

for label in unique_labels:
    print("Pretraining excluding ", label)
    model_name = label+"_pretraining"
    train_dataset = get_flair_dataset_from_dataframe(fs_training_data[label]["pre"], "text", "labels")
    corpus = Corpus(train=train_dataset, dev=dev_dataset, test=test_dataset, name=model_name, sample_missing_splits=False)
    label_dict = corpus.make_label_dictionary(label_type='class')
    tars = TARSClassifier(num_negative_labels_to_sample=4, embeddings='distilbert-base-german-cased') # roberta-large
    tars.add_and_switch_to_new_task(task_name=model_name,
        label_dictionary=label_dict,
        label_type='class',
    )
    trainer = ModelTrainer(tars, corpus)
    trainer.train(
        base_path='models/' + model_name,  # path to store the model artifacts
        learning_rate=0.01,                # use very small learning rate
        mini_batch_size=8,
        max_epochs=20
    )

# %%



