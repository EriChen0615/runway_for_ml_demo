# runway_for_ml Demo Projects

This repository contains the following demo projects of the [runway_for_ml](https://github.com/EriChen0615/runway_for_ml) framework. These projects can be useful for understanding the framework or bootstrapping your own project.

# runway_for_ml in 30 minutes - IMDb Classification

## Introduction

Let's see how `runway_for_ml` can be used to tackle the IMDb text classification task to demonstrate the key features offered by the framework.

IMDb is a binary text classifcation task. Below is an example in the `train` split. Label 0 stands for negative review and 1 stands for positive.

```json
{
    "label": 0,
    "text": "Oh brother... this is a terrible film."
}
```

Our task is to train a transformer model that predicts whether the review is positive or negative. In this demo project and most ML project in general, the research pipeline usually consists of the following stages:

1. **Data Processing**: prepare the data for training/test/validation.
2. **Training**: train the model while tracking losses and validation performance.
3. **Testing**: run inference on the test set and computes evaluation metrics.

We will start with data processing.

## Data Processing 

In `runway_for_ml`, the data processing pipeline is a Directed Acyclic Graph (DAG, meaning directionally-connected nodes that form no loops). Each node in the DAG takes the input data from preceding nodes, does some processing work, and produces the output data to succeeding nodes. This allows for maximal code and data reusuability as we will see shortly.

### DataOps - the data processing unit

A Data Operation (DataOp) is the basic data processing unit in `runway_for_ml`. To define a DataOp, You first write the code (a functor) that does the data processing, and then configure it in the configuration file to specify how it works for particular nodes. 

A functor is simply a class whose object can be called as a function. Below is a functor called `SplitHFDataset` that splits a split in the dataset to two splits with a specified proportion. 

```python
from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor

@register_transform_functor
class SplitHFDataset(HFDatasetTransform):
    def setup(self, src_split, tgt_split, split_size, split_kwargs={}, **kwargs):
        self.src_split = src_split
        self.tgt_split = tgt_split
        self.split_size = split_size
        self.split_kwargs = split_kwargs
        super().setup(**kwargs)
    
    def _call(self, data, *args, **kwargs):
        src_ds = data[self.src_split]
        splited_dataset_dict = src_ds.train_test_split(self.split_size, **self.split_kwargs)
        src_ds, tgt_ds = splited_dataset_dict['train'], splited_dataset_dict['test']

        data[self.src_split] = src_ds
        data[self.tgt_split] = tgt_ds
        
        print("Split into train/test/validation:", data)
        return data
```

To define a functor, you need to:

1. **Decorate** it with `@register_transform_functor`, so that `runway_for_ml` recognizes it.
2. **Inherit** from `HFDatasetTransform` or `BaseTransform`, depending on the type of the input data you expect. We recommend you make the input/output data objects of huggingface `Dataset` or `DatasetDict` and inherit from `HFDatasetTransform`, as this provides much more efficient caching functionality.
3. **Override** the `setup()` function, which gets called for every node in the data processing DAG to configure the DataOp. You can add as many arguments to it as you like. But do remember to call `super().setup(**kwargs)` EXCEPT when you inherit from `BaseTransform`.
4. **Override** the `_call()` function, which gets called for every node in the data processing DAG. `data` will be the output from the input node (or a list of data if there are multiple input nodes). It must return the data processed.

In this example, `SplitHFDataset` accepts `DatasetDict` and outputs a `DatasetDict`. It takes a proporation of (specified by `split_size`) the split named `src_split` and make a new split with name `tgt_split`. 

In the next section, we will look at how to configure these DataOps and connect them in the DAG via configuration files.


### Define the DAG with jsonnet

The following figure shows the data processing DAG of the demo IMDb project. The node names are self-explanatory. The nodes and connections of the DAG are defined with jsonnet syntax as in `configs/imdb_classification/data_config.libsonnet`. Each entry in the `transforms` field defines a node and its predecessors. The important fields are:

- **node name** (key): the unique name of the node. We use `input:/process:/output:` to indicate the role of the node.
- **input_node**: the name of the node from which this node takes data from. If left blank, the model takes no data.
- **transform_name**: name of the data processing functor class in your code.
- **setup_kwargs**: key-value arguments to be passed into the `.setup()` function when the functor is initialized.
- **cache**: whether the output of the node should be saved to disk, so that it can be loaded without re-computation in the next run. Note that some objects are not serializable and hence cannot be cached to disk (e.g., torch's DataLoader). In these cases, `cache` must be set to `false`.
- **regenerate**: if true, the node will re-compute the data using its input even if a cache file on disk is available.

Here is an example of how our `SplitHFDataset` functor can be configured as a node in the DAG. The key-value pairs in `setup_kwargs` will be used as arguments to the `setup()` function of the DataOp `SplitHFDataset`.


```json
{
    "process:SplitValidationFromTrain": {
          "input_node": "process:TokenizeImdbDataset",
          "transform_name": "SplitHFDataset",
          "setup_kwargs": {
            "src_split": "train",
            "tgt_split": "valid",
            "split_size": 1000,
          },
          "cache": true,
          "regenerate": false,
        },
}
```

> jsonnet (pronounced "jay sonnet") and json: Configuration files in `runway_for_ml` uses the jsonnet syntax. In essence, jsonnet is a superset of json with support of variables, exporting, etc. The use of jsonnet enalbes us to build modular configuration files. Detailed documentation can be found on https://jsonnet.org/. 

Have a look at `configs/imdb_classification/data_config.libsonnet` to understand how the data pipeline is constructed. 

`runway_for_ml` provides a list of ready-to-use DataOps to handle common data processing. The full list can be found here (WIP). The functional design of DataOps makes it easy to share code between programmers. You may use runway plugin (WIP) or simply copy over the functor code and  configuration files to share your work with others!


### Run the data processing pipeline

How do I tell runway which data pipeline to run? It is specified in the configuration file `configs/imdb_classification/exp_config.jsonnet`. We refer to it as the **experiment config**, which dictates the core settings of an experiment. Let's have a break-down look of the experiment config.

#### Jsonnet Pre-requisite: export and import

Before we jump into the details, it's important to understand the variable and import syntax in jsonnet as it is heavily used to modularize our configs. For example:

```python
# In file `data_config.libsonnet`
# ...
# Export variables
{
    imdb_preprocess_pipeline: imdb_preprocess_pipeline,
    distilbert_tokenizer_config: DistilBertTokenizerConfig,
}

# In file `exp_config.jsonnet`
local data = import 'data_config.libsonnet' # imports all the exported variable in `data_config.libsonnet` file.

{
    #...
    data_pipeline: data.imdb_preprocess_pipeline, # access imported variables with dot operator.
}
```

As shown in the comment, the `import` statement allows us to access variables defined in another config file with dot operator. `runway_for_ml` conventionally breaks the configs into data, evaluation, meta, and experiment config to maximize reuse. However, you do not need to adhere to this modularization strictly as long as you have a legal experiment config. 

#### Breakdown explanations of experiment config

The experiment config must have some important fields so that `runway_for_ml` can run the experiments. These include:

- `experiment_name`: name of the experiment. Determines the name of the directory for saving artifacts
- `meta`: meta information of the experiment. For example, cache location, wandb logger information, etc. See `meta_config.libsonnet` for details.
- `data_pipeline`: specify which pipeline to use
- `model_config`: model configurations to be used in an executor (explained later).
- `exector`: specify which executor class to use for training/validation/testing. More on this later.
- `train`: training configurations (later).
- `test`: testing configurations (later).
- `eval`: evaluation pipeline (later).

In this example, we specify the `data_pipeline` as `data.imdb_preprocess_pipeline`, so that's is what runway will run!

#### Run and inspect pre-processed data

Now, let's run the pipeline to process our data! Run the following command or the corresponding portion in `run_experiment.sh`. This will set the pipeline to work. 

```bash
python src/main.py \
    --config "configs/imdb_classification/exp_config.jsonnet" \
    --mode 'prepare_data' \
    --opts \
    "meta.EXPERIMENT_FOLDER"="experiments/imdb"
```

By setting the mode to `prepare_data`, runway will try to run every node whose name starts with `output:` and recursively runs their predecessors if needed. If your code runs successfully, you should enter a **breakpoint** at the end of the pipeline. This is specified by the following config:

```python
'output:InspectImdbDataloaders': {
    input_node: 'process:SplitValidationFromTrain',
    transform_name: 'InsertBreakpoint',
    setup_kwargs: {},
    cache: false,
    regenerate: true,
},
```

The `InsertBreakpoint` is a bulit-in DataOp. It transparently pass the data through the node, and invoke the `breakpoint()` function when it's called. When the program breaks here, you have access to all the local variables. You will see the following output: 

```python
Data under inspection: DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 24000
    })
    test: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
    valid: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 1000
    })
})
```

Now you can play around with the `data` variable. 

```python
data['train'][0]['text']
"This movie appears to have been made by someone ......"
(Pdb) data['train'][0]['label']
0
```

During training/testing, you would take the output from one of the node. We will see how to specify it in the next section.

## Training



















