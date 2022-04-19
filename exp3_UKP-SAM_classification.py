from collections import defaultdict
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from nltk.tokenize import sent_tokenize
import re
import csv
import pickle
import numpy as np


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def get_train_args(model_name):
    train_args = {
        "output_dir": "../models/FAME_UKP-SAM/" + model_name,
        "cache_dir": "../cache/",
        "best_model_dir": "../FAME_UKP-SAM/" + model_name + "/best_model/",

        "fp16": False,
        "fp16_opt_level": "O1",
        
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.2,
        "adam_epsilon": 1e-9,
        "warmup_ratio": 0.1,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        # "scheduler": "cosine_with_hard_restarts_schedule_with_warmup",
        
        "n_gpu": 1,
        "learning_rate": 5e-6,
        "num_train_epochs": 20,
        "max_seq_length": 512,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "do_lower_case": False,
        "strip_accents": True,

        "logging_steps": 50,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 0,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": False,
        "save_eval_checkpoints": False,
        "save_steps": 0,
        "no_cache": True,
        "save_model_every_epoch": True,
        "tensorboard_dir": None,

        "overwrite_output_dir": True,
        "reprocess_input_data": True,

        "silent": False,
        "use_multiprocessing": True,

        "wandb_project": None,
        "wandb_kwargs": {},

        "use_early_stopping": False,
        "early_stopping_patience": 4,
        "early_stopping_delta": 0,
        "early_stopping_metric": "f1",
        "early_stopping_metric_minimize": False,

        "manual_seed": 9721,
        "encoding": None,
        "config": {},
    }
    return train_args

def precision_macro(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred, average='macro')
def recall_macro(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, average='macro')
def f1_macro(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')

relevance_labelled_data = pd.read_csv("../data/all_topics.tsv", sep="\t", encoding="UTF-8", quoting=csv.QUOTE_NONE)
# relevance_labelled_data = relevance_labelled_data[1:600]

label_dict = {'NoArgument':0, 'Argument_against':1, 'Argument_for':2}

st_df = relevance_labelled_data[['topic', 'sentence', 'annotation', 'set']]
st_df.columns = ['text_a', 'text_b', 'labels', 'set']
st_df['labels'].replace(label_dict, inplace=True)

# split a dev set from the training data
train_df = st_df[st_df['set'] == 'train'].drop(columns=['set'])
test_df = st_df[st_df['set'] == 'test'].drop(columns=['set'])
dev_df = st_df[st_df['set'] == 'val'].drop(columns=['set'])

# weights
all_train_labels = train_df.labels.tolist()
unique_labels = [0, 1, 2]
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = unique_labels, y = all_train_labels)
print(class_weights)

# start
model_to_test = ("roberta", "roberta-large")
use_cuda = True
train_args = get_train_args("ukp_sam")

# Create a ClassificationModel
model = ClassificationModel(
    model_to_test[0], 
    model_to_test[1], 
    num_labels=3, 
    # weight=class_weights,
    use_cuda=use_cuda, 
    cuda_device=-1, 
    args=train_args)

# Train the model
model.train_model(train_df, eval_df=dev_df, precision=precision_macro, recall=recall_macro, f1=f1_macro)

# Eval best model
model = ClassificationModel(model_to_test[0], train_args['best_model_dir'], num_labels=3, use_cuda=use_cuda, cuda_device=0, args=train_args)
result, model_outputs, wrong_predictions = model.eval_model(
    test_df, acc=sklearn.metrics.accuracy_score, precision=precision_macro, recall=recall_macro, f1=f1_macro
)

# get labels
pred_labels = model_outputs.argmax(axis=-1).tolist()
true_labels = test_df["labels"].tolist()

# save
with open("tmp_results.pkl", "wb") as f:
    pickle.dump(pred_labels, f)
    pickle.dump(true_labels, f)
    pickle.dump(result, f)
    pickle.dump(model_outputs, f)

# report
print(sklearn.metrics.classification_report(true_labels, pred_labels, digits=3))
print("Kappa", sklearn.metrics.cohen_kappa_score(true_labels, pred_labels))

# pred output
test_df['predicted'] = pred_labels
test_df.to_csv("ukp_sam.csv")
