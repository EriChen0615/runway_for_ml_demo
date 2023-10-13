source scripts/activate_env.sh

# prepare data
python src/main.py \
    --config "configs/imdb_classification/exp_config.jsonnet" \
    --mode 'prepare_data' \
    --opts \
    "meta.EXPERIMENT_FOLDER"="experiments/imdb"

EXPERIMENT_NAME="distilbert-imdb_classification"
# training
# python src/main.py \
#     --config "configs/imdb_classification/exp_config.jsonnet" \
#     --experiment_name $EXPERIMENT_NAME \
#     --mode 'train' \
#     --opts \
#     meta.seed=615926 \
#     meta.EXPERIMENT_FOLDER="experiments/imdb" \
#     executor.init_kwargs.use_data_node="output:MakeImdbDataloaders" \
#     train.batch_size=32 \
#     train.optimizer_config.scheduler_params.num_warmup_steps=0 \
#     train.trainer_paras.num_sanity_val_steps=0 \
#     train.trainer_paras.accelerator="gpu" \
#     train.trainer_paras.devices=1 \
    # train.trainer_paras.strategy="ddp" \

# testing
# python src/main.py \
#     --config "configs/imdb_classification/exp_config.jsonnet" \
#     --experiment_name $EXPERIMENT_NAME \
#     --mode 'test' \
#     --opts \
#     meta.seed=615926 \
#     meta.EXPERIMENT_FOLDER="experiments/imdb" \
#     executor.init_kwargs.use_data_node="output:MakeImdbDataloaders" \
#     test.batch_size=128 \
#     test_suffix="distilbert-bs=32-ep=2" \
#     train.trainer_paras.accelerator="gpu" \
#     train.trainer_paras.devices=1 \


