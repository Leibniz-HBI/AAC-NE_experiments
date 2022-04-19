from collections import defaultdict
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

from simpletransformers.classification import ClassificationModel
import argparse
import pandas as pd
import logging
import sklearn
from sklearn.model_selection import train_test_split

from nltk.tokenize import sent_tokenize
import re
import pickle
import numpy as np
import scipy


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# extract sentences around keyterms and mask named entities
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield i, i+n, lst[i:i + n]

if __name__=='__main__':
    
    # raw data for prediction
    raw_df = pd.read_csv("../data/tg_sentence_splits.csv", encoding="utf8")  # .iloc[870:900]
    raw_df["pred_label"] = False
    raw_df["pred_score"] = .0
    predict_df = raw_df[["topic", "text"]].copy()  # .iloc[:20]
    predict_df.columns = ['text_a', 'text_b']
    predict_df.text_b.fillna("The", inplace=True)
    # print(predict_df)
    predict_df = predict_df.values.tolist()
    # print(predict_df.values.tolist())

    # start prediction
    model_type = "roberta"
    model_dir = "../FAME_UKP-SAM/ukp_sam/best_model/"

    model = ClassificationModel(model_type, model_dir)
    
    for begin, end, chunk in chunks(predict_df, 1000):
        print(begin)
        # import pdb; pdb.set_trace()
        predictions, raw_outputs = model.predict(chunk)

        class_probs = scipy.special.softmax(raw_outputs, axis = -1)

        raw_df["pred_label"].iloc[begin:end] = predictions
        raw_df["pred_score"].iloc[begin:end] = class_probs[:,1]

    raw_df.to_csv("../data/tg_sentence_splits_pred.csv")


