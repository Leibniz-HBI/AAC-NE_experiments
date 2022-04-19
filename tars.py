import flair
import pandas as pd
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
import glob
import os

# define columns
column_name_map = {0: 'text', 1: 'label'}

# this is the folder in which train, test and dev files reside
data_folder = 'Train-splits/testway/'

splits =glob.glob('Train-splits/testnway/*')

for s in splits:
    print('Train Split: %s'%s)
    s = os.path.basename(s)
    print(s)
    data_folder = 'Train-splits/testnway/%s/'%s
    for i in range(1,6):
        print('Model pass %d'%i)
    # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         train_file='NuclearEnergy_train.csv',
                                         test_file='NuclearEnergy_test.csv',
                                         dev_file='NuclearEnergy_dev.csv',
                                         skip_header=True,
                                         delimiter=',',
                                         label_type='topic')



    # 4. make a label dictionary
        label_dict = corpus.make_label_dictionary(label_type='topic')

    # 5. start from our existing TARS base model for English
        tars = TARSClassifier(num_negative_labels_to_sample=8).load("tars-base")


        tars.add_and_switch_to_new_task(task_name="aspect classification",
                                label_dictionary=label_dict,
                                label_type='topic',
                                )

    # 7. initialize the text classifier trainer
        trainer = ModelTrainer(tars, corpus)

    # 8. start the training
        trainer.train(base_path='Tars/model_%s/rep_%d'%(s,i),  # path to store the model artifacts
                learning_rate=0.01,  # use very small learning rate
                mini_batch_size=32,
                mini_batch_chunk_size=4,
                max_epochs=9,
                train_with_dev= "True"  
                 )

