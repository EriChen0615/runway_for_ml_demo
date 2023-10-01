// [Runway experiment config]: (pre-processing) data pipeline settings

local DistilBertTokenizerConfig = {
  'version_name': 'distilbert-base-uncased',
  'class_name': 'AutoTokenizer',
  'tokenize_kwargs': {
    'padding': 'max_length',
    'truncation': true
  },
};

local imdb_preprocess_pipeline= {
    DataPipelineLib: 'data_modules',
    DataPipelineClass: 'DataPipeline',
    name: 'IMDBDatapipeline',
    regenerate: false,
    do_inspect: true,
    inspector_config: { 
        log_dir: 'tests/'
    },
    transforms: {
        'input:LoadImdbDataset': {
            transform_name: 'LoadHFDataset',
            setup_kwargs: {
                dataset_name: 'imdb'
            },
            cache: true,
            regenerate: false,
        },
        'process:TokenizeImdbDataset': {
          input_node: 'input:LoadImdbDataset',
          transform_name: 'HFDatasetTokenizeTransform',
          setup_kwargs: {
            tokenizer_config: DistilBertTokenizerConfig,
            tokenize_fields_list: ["text"],
            splits_to_process: ["train", "valid", "test"],
            rename_col_dict: {'text_input_ids': 'input_ids', 'text_attention_mask': 'attention_mask'},
          },
          cache: true,
          regenerate: false,
        },
        'process:SplitValidationFromTrain': {
          input_node: 'process:TokenizeImdbDataset',
          transform_name: 'SplitHFDataset',
          setup_kwargs: {
            'src_split': 'train',
            'tgt_split': 'valid',
            'split_size': 1000,
          },
          cache: true,
          regenerate: false,
        },
        'output:MakeImdbDataloaders': {
            input_node: 'process:SplitValidationFromTrain',
            transform_name: 'MakeImdbDataloaders',
            setup_kwargs: {

            },
            cache: false,
            regenerate: true,
        },
        'output:InspectImdbDataloaders': {
            input_node: 'process:SplitValidationFromTrain',
            transform_name: 'InsertBreakpoint',
            setup_kwargs: {

            },
            cache: false,
            regenerate: true,
        },
    },
};

{
    imdb_preprocess_pipeline: imdb_preprocess_pipeline,
    distilbert_tokenizer_config: DistilBertTokenizerConfig,
}