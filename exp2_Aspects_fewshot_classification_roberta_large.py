import logging
import os
import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel
from sklearn.utils import class_weight

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_train_args(model_dir, exp_i=1):
    train_args = {
        "output_dir": model_dir,
        "cache_dir": "cache/",
        "best_model_dir": model_dir + "/best_model/",

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
        "max_seq_length": 256,
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
        "save_model_every_epoch": False,
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

        "manual_seed": 9721 * exp_i,
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


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # split a dev set from the training data
    train_df = pd.read_csv("Train-splits/testnway/full_shot/NuclearEnergy_train.csv", header=None,
                           names=['text', 'labels'])
    test_df = pd.read_csv("Train-splits/testnway/full_shot/NuclearEnergy_test.csv", header=None,
                          names=['text', 'labels'])
    dev_df = pd.read_csv("Train-splits/testnway/full_shot/NuclearEnergy_dev.csv", header=None,
                         names=['text', 'labels'])

    # weights
    all_train_labels = train_df.labels.tolist()
    unique_labels = list(set(all_train_labels))
    unique_labels.sort()
    label2int = {label: i for i, label in enumerate(unique_labels)}
    int2label = {i: label for i, label in enumerate(unique_labels)}
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=unique_labels, y=all_train_labels)
    cw_dict = {label2int[l]: class_weights[i] for i, l in enumerate(unique_labels)}
    print(cw_dict)
    print(int2label)
    #exit(0)

    train_df.labels = [label2int[label] for label in train_df.labels]
    test_df.labels = [label2int[label] for label in test_df.labels]
    dev_df.labels = [label2int[label] for label in dev_df.labels]

    # start
    models_to_test = [
        ("roberta", "roberta-large")
    ]
    model_repititions: int = 5
    k_train_splits = [100]

    # few-shots with k random samples per class
    for k in k_train_splits:
        print(f"{k} Few-Shots")
        if k ==0:
            train_split_df = pd.DataFrame(columns=['text','labels'])
        else:
            train_split_df = pd.read_csv(f'Train-splits/testnway/{k}_shot/NuclearEnergy_train.csv',
                                        header=None,names=['text', 'labels'])

        train_split_df.labels = [label2int[label] for label in train_split_df.labels]


        # r repititions
        for r in range(1, 6):
            use_cuda = True
            model_base = f"models_seed/exp2_{k}shot_trial{r}/"

            for model_i, model_to_test in enumerate(models_to_test):
                print(model_to_test, "\n", "*********************************************")
                model_name = model_to_test[1].replace("/", "_")
                model_dir = model_base + model_name
                train_args = get_train_args(model_dir, r)

                if model_i >= 0:
                    # Create a ClassificationModel
                    model = ClassificationModel(
                        model_to_test[0],
                        model_to_test[1],
                        num_labels=len(unique_labels),
                        # weight=cw_dict,
                        use_cuda=use_cuda,
                        cuda_device=-1,
                        args=train_args
                    )
                    if k > 0:
                        model.train_model(train_split_df, train_args['best_model_dir'],eval_df=dev_df, precision=precision_macro, recall=recall_macro,
                                          f1=f1_macro)


                if k > 0:
                    model = ClassificationModel(model_to_test[0], train_args['best_model_dir'], use_cuda=use_cuda,
                                                cuda_device=0, args=train_args)

                result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score,
                                                                            precision=precision_macro,
                                                                            recall=recall_macro, f1=f1_macro)

                # get labels
                pred_labels = model_outputs.argmax(axis=-1).tolist()
                true_labels = test_df["labels"].tolist()

                # report
                report = sklearn.metrics.classification_report(true_labels, pred_labels, digits=3)
                kappa = sklearn.metrics.cohen_kappa_score(true_labels, pred_labels)
                print(report)
                print(kappa)

                with open(model_dir + "/results.txt", "w") as f:
                    f.write(report)
                    f.write("\nKappa: " + str(kappa))

                # pred output
                test_df['predicted'] = pred_labels
                test_df.to_csv(model_dir + "/fame_argaspects.csv")
