// [Runway experiment config]: meta information
// This is the project-wide meta configuration file
// It defines project directories, logging options, and other settings shared across different experiments.
// Its value can be override directly in lower-level config files

// Uncomment and set value as appropriate. 

// Default values for training control
local seed=2022;

// data cache configuration
local wandb_cache_dir = "cache/wandb_cache/"; 
local default_cache_dir = "cache/"; 

local default_meta = {
  "DATA_FOLDER": "", 
  "EXPERIMENT_FOLDER": "./experiments/",
  "TENSORBOARD_FOLDER": "./tensorboards/", 
  "WANDB": { # these key-value pairs will be used as arguments to `wand.init()` function
      "CACHE_DIR":  wandb_cache_dir,
      "entity": "erichen0615",
      "project": "runway_demo",
      "tags": ['runway_demo'],
  },
  "logger_enable": ["tensorboard"], # "wandb" for wandb logger, "csv" for csv logger
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "seed": seed,
  "default_cache_dir": default_cache_dir,
  "cuda": 0,
  "gpu_device":0,
};

{
  default_meta: default_meta
}