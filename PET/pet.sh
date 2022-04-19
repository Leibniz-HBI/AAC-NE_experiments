python3 cli.py \
--method pet \
--pattern_ids 0 1 2 \
--data_dir ../Train-splits/testnway/full_shot/ \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name nuclear-aspect \
--output_dir /data/pet_full_pattern-123/ \
--overwrite_output_dir \
--do_train \
--do_eval  \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 4 \
--pet_max_seq_length 256 \
--pet_num_train_epochs 4 \
--pet_repetitions 5 \
--eval_set 'test'
