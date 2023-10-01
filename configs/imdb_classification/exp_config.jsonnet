// [Runway experiment config]: experiment settings
// This is the configuration file for specific experiment settings

local data = import 'data_config.libsonnet';
local eval = import 'eval_config.libsonnet';
local meta = import 'meta_config.libsonnet';

{
  experiment_name: 'YOUR EXPERIMENT NAME HERE',
  test_suffix: 'TEST SUFFIX HERE',
  meta: meta.default_meta,
  data_pipeline: data.imdb_preprocess_pipeline,
  tokenizer_config: data.distilbert_tokenizer_config,
  model_config: {
    from_pretrained_kwargs: {
        'pretrained_model_name_or_path': 'distilbert-base-uncased',
        'num_labels': 2,
    },
  },
  executor: {
    ExecutorClass: 'TextClassificationExecutor',
    init_kwargs: {
        use_data_node: 'output:MakeImdbDataloaders',
        use_fp16: true,
    },
  }, 
  train: {
    batch_size: 16,
    trainer_paras: {
      max_epochs: 2,
      log_every_n_steps: 50,
      check_val_every_n_epoch: 1,
    },
    model_checkpoint_callback_paras: {
      save_top_k: 0,
      filename: 'model_step_{step}',
      verbose: true,
      save_last: true,
      auto_insert_metric_name: false,
      save_on_train_epoch_end: false,
    },
    optimizer_config: {
      optimizer_name: 'AdamW',
      optimizer_params: {
        lr: 1e-5, 
      },
    },
    use_lora: false,
    lora_config: {
      'unet': {
        r: 8,
        lora_alpha: 32,
        target_modules: [],
        suffices_to_lora: ['to_q', 'to_v'],
        lora_dropout: 0.1,
        bias: "lora_only",
      },
    },
  },
  test: {
    batch_size: 32,
  },
  eval: {
    pipeline_config: eval.eval_pipeline,
    valid_eval_pipeline_config: eval.eval_pipeline,
  },
}
