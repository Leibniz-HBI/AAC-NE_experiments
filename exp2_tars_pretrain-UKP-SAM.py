import flair
import pandas as pd
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
import glob
import csv
import os



ukpsam_data = pd.read_csv("../application/data/UKP-SAM_all_topics.csv", sep="\t", encoding="UTF-8", quoting=csv.QUOTE_NONE)
df_data = {'train':[], 'val':[], 'test':[]}
for i, row in ukpsam_data.iterrows():
    # print(row)
    try: 
        s = row['sentence'].replace(row['topic'], 'topic')
        df_data[row['set']].append((s, row['topic']))
    except:
        print(row)
for s in df_data.keys():
    df = pd.DataFrame(df_data[s])
    df.columns = ['sentence', 'label']
    df.to_csv("../application/data/UKP-SAM_tars_pretraining_" + str(s) + ".csv", index=False)

# exit(0)

# define columns
# column_name_map = {0: 'text', 1: 'label'}
column_name_map = {1: 'text', 2: 'label'}

# this is the folder in which train, test and dev files reside
#data_folder = '../application/data/'
data_folder = 'Train-splits'

#splits = ['../application/data/']
splits = ['Train_splits']

for s in splits:
    print('Train Split: %s'%s)
    s = os.path.basename(s)
    print(s)
    for i in range(1,2):
        print('Model pass %d'%i)
    # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         #train_file='UKP-SAM_tars_pretraining_train.csv',
                                         #test_file='UKP-SAM_tars_pretraining_test.csv',
                                         #dev_file='UKP-SAM_tars_pretraining_val.csv',
                                         train_file='cmp_train.csv',
                                         test_file='cmp_test.csv',
                                         dev_file='cmp_dev.csv',
                                         skip_header=True,
                                         delimiter=',',
                                         label_type='cmp_code')



    # 4. make a label dictionary
        label_dict = corpus.make_label_dictionary(label_type='cmp_code')

    # 5. start from our existing TARS base model for English
        # tars = TARSClassifier(num_negative_labels_to_sample=4, embeddings='roberta-large')
        tars = TARSClassifier(num_negative_labels_to_sample=4).load('Tars_large/model_model_ukp-topic/best-model.pt')

        tars.add_and_switch_to_new_task(task_name="cmp code classification",
                                label_dictionary=label_dict,
                                label_type='cmp_code',
                                )

    # 7. initialize the text classifier trainer
        trainer = ModelTrainer(tars, corpus)

    # 8. start the training
        trainer.train(
                base_path='Tars_large/model_ukp-topic_cmp-code',  # path to store the model artifacts
                learning_rate=0.01,                               # use very small learning rate
                mini_batch_size=4,
                # mini_batch_chunk_size=4,
                max_epochs=20,
                # train_with_dev= "True"  
        )

