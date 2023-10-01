
from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor
from datasets import DatasetDict
from torch.utils.data import DataLoader
import evaluate
import torch
    

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
    
@register_transform_functor
class MakeImdbDataloaders(BaseTransform):
    def setup(self, train_kwargs={}, test_kwargs={}, val_kwargs={}, use_columns=None, output_format='torch'):
        self.use_columns = use_columns
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
        self.val_kwargs = val_kwargs

    def _call(self, data, *args, **kwargs):
        train_ds, test_ds, valid_ds = data['train'], data['test'], data['valid']
        train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        valid_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        def train_collate_fn(examples):
            input_ids = torch.vstack([example['input_ids'] for example in examples])
            labels = torch.vstack([example['label'] for example in examples])
            attention_mask = torch.vstack([example['attention_mask'] for example in examples])
            return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        
        def test_collate_fn(examples):
            input_ids = torch.vstack([example['input_ids'] for example in examples])
            labels = torch.vstack([example['label'] for example in examples])
            attention_mask = torch.vstack([example['attention_mask'] for example in examples])
            return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

        train_dataloader = DataLoader(
            train_ds,
            shuffle=True,
            collate_fn=train_collate_fn,
            batch_size=self.global_config.train.get('batch_size', 1),
            num_workers=self.global_config.train.get('dataloader_workers', 0),
            **self.train_kwargs
        )
        test_dataloader = DataLoader(
            test_ds,
            shuffle=False,
            collate_fn=test_collate_fn,
            batch_size=self.global_config.test.get('batch_size', 1),
            num_workers=self.global_config.test.get('dataloader_workers', 0),
            **self.test_kwargs
        )
        val_dataloader = DataLoader(
            valid_ds,
            shuffle=False,
            collate_fn=test_collate_fn,
            batch_size=self.global_config.test.get('batch_size', 1),
            num_workers=self.global_config.test.get('dataloader_workers', 0),
            **self.val_kwargs
        )

        return {
            'train_dataloaders': [train_dataloader],
            'test_dataloaders': [test_dataloader],
            'val_dataloaders': [val_dataloader],
        }

@register_transform_functor
class EvaluateBinaryClassification(BaseTransform):
    def setup(self, pred_field, ref_field):
        self.pred_field = pred_field
        self.ref_field = ref_field
    
    def _call(self, eval_recorder):
        preds = eval_recorder.get_sample_logs_column(self.pred_field)
        refs = eval_recorder.get_sample_logs_column(self.ref_field)

        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")

        acc = accuracy.compute(predictions=preds, references=refs)
        prec = precision.compute(predictions=preds, references=refs)

        eval_recorder.log_stats_dict({'accuracy': acc['accuracy'], 'precision': prec['precision']})
        return eval_recorder

