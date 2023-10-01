// [Runway experiment config]: evaluation pipeline settings
// This is the configuration file that defines the evaluation pipeline

local eval_pipeline= {
    DataPipelineLib: 'data_modules',
    DataPipelineClass: 'DataPipeline',
    name: 'ArchitectDatapipeline',
    regenerate: true,
    do_inspect: true,
    out_ops: ['output:MergeAllEvalRecorderAndSave'],
    inspector_config: { 
        log_dir: 'tests/'
    },
    transforms: {
        'input:GetEvaluationRecorder': {
            transform_name: 'GetEvaluationRecorder',
            setup_kwargs: {},
            cache: false,
        },
        'process:EvaluateBinaryClassification': {
            input_node: 'input:GetEvaluationRecorder',
            transform_name: 'EvaluateBinaryClassification',
            setup_kwargs: {
                pred_field: 'predicted_class_id',
                ref_field: 'ground_truth_class_id',
            },
            cache: false
        },
        'output:MergeAllEvalRecorderAndSave': {
            input_node: ['process:EvaluateBinaryClassification'],
            // input_node: ['process:ComputeNIMAScore'],
            transform_name: 'MergeAllEvalRecorderAndSave',
            cache: false
        },
        'output:UploadEvaluationResultToWandb': {
            input_node: 'output:MergeAllEvalRecorderAndSave',
            transform_name: 'UploadEvalRecorderToWandb',
            setup_kwargs: {
                log_stats_dict: true,
                prefix_to_log: ['generated_image-'],
                // columns_to_log: ['edge_recall'],
                wandb_tab_name: 'validation',
            },
            cache: false,
        },
    },
};

{
    eval_pipeline: eval_pipeline
}