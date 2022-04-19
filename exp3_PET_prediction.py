import os
import re
import logging

import numpy as np
import pandas as pd
import scipy

import pet
from pet.wrapper import WrapperConfig, TransformerModelWrapper
from pet.utils import InputExample
from pet.tasks import PROCESSORS

os.chdir('~/argument-mining/classification/PET/')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# extract sentences around keyterms and mask named entities
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield i, i+n, lst[i:i + n]

n_shots = ['1', '2', '4', '8', '10', '50', '100', 'full']
reps = list(range(0,5))

# raw data for prediction
raw_df = pd.read_csv("~/argument-mining/application/data/tg_sentence_splits_pred.csv", encoding="utf8")  # .iloc[870:900]
raw_df = raw_df[raw_df.topic == "nuclear energy"][raw_df.pred_label > 0]
raw_df.head(2)

processor = PROCESSORS['nuclear-aspect']()
class_idx_to_class_name = {idx: name for idx, name in enumerate(processor.get_labels())}
predict_df = raw_df[['idx', 'text']].copy()  # .iloc[:20]
predict_df.text.fillna("The", inplace=True)
predict_df = [InputExample(row['idx'], row['text']) for index, row in predict_df.iterrows()]

for n_shot in n_shots:
    for rep in reps:
        rep = str(rep)
        exp_name = n_shot + '_' + rep + '_'
        print(exp_name)
        print("*****************")
        
        # load model
        wrapper = TransformerModelWrapper.from_pretrained("/data/pet_" + n_shot + "_pattern-123/p1-i" + rep)
        wrapper.model.to("cuda")

        # convert data
        raw_df[exp_name + "pred_label"] = False
        raw_df[exp_name + "pred_score"] = .0
        
        # run prediction
        for begin, end, chunk in chunks(predict_df, 1000):
            print(begin)
            # import pdb; pdb.set_trace()
            eval_result = wrapper.eval(chunk, device="cuda")
            predictions = np.argmax(eval_result['logits'], axis=1)
            predictions = [class_idx_to_class_name[prediction] for prediction in predictions]
            
            class_probs = scipy.special.softmax(eval_result['logits'], axis = -1)

            raw_df[exp_name + "pred_label"].iloc[begin:end] = predictions
            raw_df[exp_name + "pred_score"].iloc[begin:end] = np.max(class_probs, axis=1)

raw_df.to_csv("~/argument-mining/application/data/tg_sentence_splits_pred_exp3.csv", index=False)
