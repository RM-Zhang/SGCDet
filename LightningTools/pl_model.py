import os
import torch
import numpy as np
import pytorch_lightning as pl
from mmdet3d.models import build_model
from mmcv.parallel import DataContainer as DC


def tensor_to_device(data, device):
    if isinstance(data, DC):
        return tensor_to_device(data.data[0], device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [tensor_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: tensor_to_device(value, device) for key, value in data.items()}
    else:
        return data
    

class pl_model(pl.LightningModule):
    def __init__(self, config, dara_dm):
        super().__init__()
        self.config = config
        
        self.model = build_model(config['model'], config['train_cfg'], config['test_cfg'])
        self.model.init_weights()
        if config['load_from'] is not None:
            state_dict = torch.load(config['load_from'], map_location=torch.device('cpu'))['state_dict']
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if "model" in key:
                    new_key = key.replace("model.", "")
                    filtered_state_dict[new_key] = value
            self.model.load_state_dict(filtered_state_dict, strict=False)
        
        self.data_dm = dara_dm
        self.val_results = []
        self.test_results = []
    
    def forward(self, batch, mode='train'):
        batch = tensor_to_device(batch, self.device)
        if mode == 'train':
            output = self.model.forward_train(batch)
        else:
            output = self.model.forward_test(batch)
        return output
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch, mode='train')
        loss = 0
        for key, value in loss_dict.items():
            self.log("train/"+key, value.detach(), on_epoch=True, sync_dist=True, batch_size=self.config.train_dataloader_config['batch_size'])
            loss += value
        self.log("train/loss", loss.detach(), on_epoch=True, sync_dist=True, batch_size=self.config.train_dataloader_config['batch_size'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        if torch.distributed.get_rank() != 0:
            return
        output = self.forward(batch, mode='test')
        self.val_results.append(output[0])
        torch.cuda.empty_cache()
    
    def validation_epoch_end(self, outputs):
        if torch.distributed.get_rank() != 0:
            return
        ret_dict = self.data_dm.val_dataset.evaluate(self.val_results)
        self.log("val/mAP_0.25", torch.tensor(ret_dict['mAP_0.25'], dtype=torch.float32), sync_dist=False)
        self.log("val/mAR_0.25", torch.tensor(ret_dict['mAR_0.25'], dtype=torch.float32), sync_dist=False)
        self.log("val/mAP_0.50", torch.tensor(ret_dict['mAP_0.50'], dtype=torch.float32), sync_dist=False)
        self.log("val/mAR_0.50", torch.tensor(ret_dict['mAR_0.50'], dtype=torch.float32), sync_dist=False)
        self.val_results = []
        
    def test_step(self, batch, batch_idx):
        output = self.forward(batch, mode='test')
        self.test_results.append(output[0])
        torch.cuda.empty_cache()
    
    def test_epoch_end(self, outputs):
        ret_dict = self.data_dm.test_dataset.evaluate(self.test_results)
        self.log("test/mAP_0.25", torch.tensor(ret_dict['mAP_0.25'], dtype=torch.float32), sync_dist=True)
        self.log("test/mAR_0.25", torch.tensor(ret_dict['mAR_0.25'], dtype=torch.float32), sync_dist=True)
        self.log("test/mAP_0.50", torch.tensor(ret_dict['mAP_0.50'], dtype=torch.float32), sync_dist=True)
        self.log("test/mAR_0.50", torch.tensor(ret_dict['mAR_0.50'], dtype=torch.float32), sync_dist=True)
        if self.config.mode == 'show':
            out_dir = os.path.join('logs', self.config['log_folder'], 'show')
            self.data_dm.test_dataset.show(self.test_results, out_dir=out_dir)
        self.test_results = []
    
    def configure_optimizers(self):
        if self.config['optimizer']['type'] == 'AdamW':
            # params_to_optimize = [param for param in self.model.parameters() if param.requires_grad]
            # optimizer = torch.optim.AdamW(
            #     params_to_optimize,
            #     lr=self.config['optimizer']['lr'],
            #     weight_decay=self.config['optimizer']['weight_decay']
            # )
            params_to_optimize = [
                {"params": [param for name, param in self.model.named_parameters() if param.requires_grad and "backbone" in name],
                "lr": self.config['optimizer']['lr'] * 0.1,
                "weight_decay": self.config['optimizer']['weight_decay'] * 1.0,
                "name": "backbone"},
                {"params": [param for name, param in self.model.named_parameters() if param.requires_grad and "backbone" not in name],
                "lr": self.config['optimizer']['lr'],
                "weight_decay": self.config['optimizer']['weight_decay'],
                "name": "others"}
            ]
            optimizer = torch.optim.AdamW(
                params_to_optimize
            )
        else:
            raise NotImplementedError
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            max_lr = [
                self.config['lr_scheduler']['max_lr'] * 0.1,
                self.config['lr_scheduler']['max_lr']
            ]
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=self.config['lr_scheduler']['total_steps'],
                pct_start=self.config['lr_scheduler']['pct_start'],
                cycle_momentum=self.config['lr_scheduler']['cycle_momentum'],
                anneal_strategy=self.config['lr_scheduler']['anneal_strategy'],
                final_div_factor=self.config['lr_scheduler']['final_div_factor'],
            )
            interval=self.config['lr_scheduler']['interval']
            frequency=self.config['lr_scheduler']['frequency']
        else:
            raise NotImplementedError
        
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': interval,
            'frequency': frequency
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }