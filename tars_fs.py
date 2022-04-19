# %%
import pandas as pd

import pickle
with open("data_samples.pkl", "rb") as f:
    fs_training_data = pickle.load(f)

# %%
from flair.datasets import SentenceDataset
from flair.data import Corpus, Sentence

def get_flair_dataset_from_dataframe(data, text_col, label_col):
    sentences = list(data.apply(lambda row: Sentence(row[text_col]).add_label('fs_class', row[label_col]), axis=1))
    return SentenceDataset(sentences)

# CAUTION: switched dev and test for more robust evaluation (original test is only 10%, smallest category had only 2 examples)
test_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_dev.csv", header=None, names=['text', 'labels'])
dev_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_test.csv", header=None, names=['text', 'labels'])

# %%
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from flair.optim import LinearSchedulerWithWarmup
from flair.data import Dictionary

restart = False
for label in fs_training_data.keys():
    print("FS experiment ", label)
    model_name = label+"_pretraining"
    for experiment in fs_training_data[label]["ft"]:
        if label == "waste" and experiment["n_shot"] == 8 and experiment["trial"] == 5:
            restart = True
        if not restart:
            continue
        if not len(experiment["data"]):
            continue
        # import pdb; pdb.set_trace()
        train_dataset = get_flair_dataset_from_dataframe(experiment["data"], "text", "labels")
        test_dataset = get_flair_dataset_from_dataframe(test_df, "text", "labels")
        dev_dataset = get_flair_dataset_from_dataframe(dev_df, "text", "labels")
        corpus = Corpus(train=train_dataset, dev=dev_dataset, test=test_dataset, name=label, sample_missing_splits=False)
        # import pdb; pdb.set_trace()
        # label_dict = corpus.make_label_dictionary(label_type='fs_class')
        label_dict = Dictionary(add_unk=False)
        label_dict.add_item(label)
        tars = TARSClassifier(num_negative_labels_to_sample=None).load("models/"+model_name+"/best-model.pt") # roberta-large
        tars.add_and_switch_to_new_task(task_name=label,
            label_dictionary=label_dict,
            label_type='fs_class',
        )
        trainer = ModelTrainer(tars, corpus)
        trainer.train(
            base_path='models_fs/' + "_".join([label, str(experiment["n_shot"]), str(experiment["trial"])]),   # path to store the model artifacts
            learning_rate=0.005,                   # use very small learning rate
            mini_batch_size=8,
            max_epochs=4,
            use_final_model_for_eval=True,
            #scheduler=LinearSchedulerWithWarmup
        )

# %%




