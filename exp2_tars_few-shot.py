import pandas as pd
import glob
import os

from flair.datasets import SentenceDataset
from flair.data import Corpus, Sentence, Dictionary
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from flair.optim import LinearSchedulerWithWarmup


def get_flair_dataset_from_dataframe(data, text_col, label_col):
    sentences = list(data.apply(lambda row: Sentence(row[text_col]).add_label('aspect', row[label_col]), axis=1))
    return SentenceDataset(sentences)

VERBALIZER = {
    "reactorsecurity": 'reactor safety',
    "weapons": 'nuclear weapons proliferation',
    "waste": 'nuclear waste',
    "enviroment/health": 'health and environment issues',
    "reliability": 'energy supply reliability',
    "costs": 'nuclear energy costs',
    "alternatives": 'alternative energies, renewables, and coal',
    "improvement": 'technological innovation and progress',
    "other": 'other aspects'
}

# CAUTION: switched dev and test for more robust evaluation (original test is only 10%, smallest category had only 2 examples)
dev_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_dev.csv", header=None, names=['text', 'labels'])
test_df = pd.read_csv("../classification/Train-splits/testnway/full_shot/NuclearEnergy_test.csv", header=None, names=['text', 'labels'])
dev_df.labels = dev_df.labels.replace(VERBALIZER)
test_df.labels = test_df.labels.replace(VERBALIZER)
dev_dataset = get_flair_dataset_from_dataframe(dev_df, "text", "labels")
test_dataset = get_flair_dataset_from_dataframe(test_df, "text", "labels")

splits =glob.glob('Train-splits/testnway/*')
# import pdb; pdb.set_trace()
# splits = ['Train-splits/testnway/full_shot']
for s in splits:
    print('Train Split: %s' % s)
    s = os.path.basename(s)
    print(s)
    data_folder = 'Train-splits/testnway/%s/'%s
    for i in range(1,6):
        print('Model pass %d' % i)

        # load training data
        train_df = pd.read_csv(data_folder + "NuclearEnergy_train.csv", header=None, names=['text', 'labels'])
        train_df.labels = train_df.labels.replace(VERBALIZER)
        train_dataset = get_flair_dataset_from_dataframe(train_df, "text", "labels")

        # create corpus
        corpus = Corpus(train=train_dataset, dev=dev_dataset, test=test_dataset, name="nuclear aspect " + s, sample_missing_splits=False)

        # 4. make a label dictionary
        label_dict = corpus.make_label_dictionary(label_type='aspect')

        # 5. start from our existing TARS base model for English
        tars = TARSClassifier(num_negative_labels_to_sample=4).load('Tars_large/model_ukp-topic_cmp-code/best-model.pt')

        # 6. add task
        tars.add_and_switch_to_new_task(task_name="aspect classification",
                                label_dictionary=label_dict,
                                label_type='aspect',
                                multi_label=True
                                )

        # 7. initialize the text classifier trainer
        trainer = ModelTrainer(tars, corpus)

        # 8. start the training
        trainer.train(
                base_path='Tars_large/tars_ft_ml-false/model_%s/rep_%d'%(s,i),  # path to store the model artifacts
                learning_rate=0.01,                                    # use very small learning rate
                mini_batch_size=4,
                # mini_batch_chunk_size=4,
                max_epochs=20,
                # train_with_dev= "True"
        )




