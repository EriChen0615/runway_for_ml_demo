from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)

@register_executor
class TextClassificationExecutor(BaseExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        *args, **kwargs
        ):
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        super().__init__(
            data_pipeline_config, 
            model_config, 
            mode, 
            train_config=train_config, 
            test_config=test_config, 
            log_file_path=log_file_path, 
            use_data_node=use_data_node,
            global_config=global_config,
            *args, **kwargs
        )
    
    def _init_model(self, model_config): 
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            **model_config.from_pretrained_kwargs, id2label=self.id2label, label2id=self.label2id
        )
    
    def prepare_data(self):
        super().prepare_data()
    
    def train_dataloader(self):
        return self.train_dataloaders[0]
    
    def test_dataloader(self):
        return self.test_dataloaders[0]
    
    def val_dataloader(self):
        return self.valid_dataloaders[0]
    
    def setup(self, stage):
        super().setup(stage)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)

        self.train_dataloaders = self.prepared_data.get('train_dataloaders', None)
        self.valid_dataloaders = self.prepared_data.get('val_dataloaders', None)
        self.test_dataloaders = self.prepared_data.get('test_dataloaders', None)
    
    def training_step(self, batch, batch_idx):
        """Defines training step for each batch

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss
    
    def _model_predict(self, input_ids, mask, labels=None):
        with torch.no_grad():
            output = self.model(input_ids=input_ids, labels=labels, attention_mask=mask)
            loss = output.loss.detach().item() if labels else -1 
            logits = output.logits
            predicted_class_id = logits.argmax(axis=1).tolist()
            predicted_class_labels = [self.model.config.id2label[_id] for _id in predicted_class_id]
        return {
            'predicted_class_id': predicted_class_id, 
            'predicted_class_labels': predicted_class_labels,
        }, loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']

        pred_res, loss = self._model_predict(input_ids=x, mask=mask, labels=y)
        
        y = y.squeeze().tolist()
        gt_class_labels = [self.model.config.id2label[_id] for _id in y]

        self.log("val_loss", loss)

        dict_to_log = {
            'ground_truth_class_id': y,
            'gruond_truth_class_label': gt_class_labels
        }
        dict_to_log.update(pred_res)

        self.valid_eval_recorder.log_sample_dict_batch(dict_to_log)
    
    def on_validation_end(self) -> None:
        return super().on_validation_end()
    
    def test_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']

        pred_res, _ = self._model_predict(input_ids=x, mask=mask)
        
        y = y.squeeze().tolist()
        gt_class_labels = [self.model.config.id2label[_id] for _id in y]

        dict_to_log = {
            'ground_truth_class_id': y,
            'gruond_truth_class_label': gt_class_labels
        }
        dict_to_log.update(pred_res)

        self.test_eval_recorder.log_sample_dict_batch(dict_to_log)
    
    def on_test_end(self) -> None:
        return super().on_test_end()



