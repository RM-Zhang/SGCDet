import pytorch_lightning as pl
from mmdet.datasets import build_dataset
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from functools import partial
from mmcv.parallel import collate


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config      
    ):
        super().__init__()
        self.trainset_config = config.data.train
        self.testset_config = config.data.test
        self.valset_config = config.data.val

        self.train_dataloader_config = config.train_dataloader_config
        self.test_dataloader_config = config.test_dataloader_config
        self.val_dataloader_config = config.test_dataloader_config
        self.config = config
    
    def setup(self, stage=None):
        self.train_dataset = build_dataset(self.trainset_config)
        self.test_dataset = build_dataset(self.testset_config)
        self.val_dataset = build_dataset(self.valset_config)
    
    def train_dataloader(self):
        if self.trainer.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            drop_last=True,
            num_workers=self.train_dataloader_config.num_workers,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=False,
            collate_fn=partial(collate, samples_per_gpu=self.train_dataloader_config.batch_size))
    
    def val_dataloader(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            dummy = torch.utils.data.TensorDataset(torch.zeros(1, 3, 224, 224))
            return DataLoader(dummy, batch_size=1)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.val_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=partial(collate, samples_per_gpu=self.train_dataloader_config.batch_size))
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.test_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=partial(collate, samples_per_gpu=self.train_dataloader_config.batch_size))