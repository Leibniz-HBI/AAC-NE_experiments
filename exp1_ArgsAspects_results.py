import pandas as pd

base_dir_pattern = "models/exp1_trial"
model_names = ["bert-base-cased", "bert-large-cased", "roberta-large", "xlm-roberta-large", "albert-base-v2", "google_electra-large-discriminator"]
metrics = ["precision", "recall", "f1", "acc"]

def read_results(model, trial):
    res = {"model": model, "trial": trial}
    with open(base_dir_pattern + str(trial) + "/" + model + "/eval_results.txt", "r") as f:
       for line in f:
           line = line.strip()
           if line:
               items = line.split(" = ")
               print(items)
               if items[0] in metrics:
                   res[items[0]] = round(float(items[1]) * 100, 1)
    return res

df = pd.DataFrame()
for model_name in  model_names:
    for i in range(1,6):
        row = read_results(model_name, i)
        df = df.append(row, ignore_index=True)
# df = df[metrics]
print(df)

df_agg = df.groupby("model").agg({
    'precision':['mean', 'std'],
    'recall':['mean', 'std'],
    'f1':['mean', 'std'],
    'acc':['mean', 'std'],
}).round(1)
print(df_agg)

df_agg.to_csv("exp1_results.csv")
