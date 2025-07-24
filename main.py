import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='configs/config.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=1234, help='random seed point')
    parser.add_argument('--log_folder', default='debug')
    parser.add_argument('--mode', default='train', help='train or eval or show')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--load_from', type=str, default=None)
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg


if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    data_dm = DataModule(config)
    
    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config, data_dm)
    
    checkpoint_callback = ModelCheckpoint(
        filename='epoch-{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        verbose=True,
        monitor=None)
    
    if config.mode == 'train':
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=True
            ),
            replace_sampler_ddp=False,
            gradient_clip_val=35,
            gradient_clip_algorithm='norm',
            max_steps=config.training_steps,
            resume_from_checkpoint=None,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step')
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch'],
            num_sanity_val_steps=0 # validation -1; skip 0
        )
        trainer.fit(model=model, datamodule=data_dm)
    elif config.mode == 'eval' or config.mode == 'show':
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=True
            ),
            logger=tb_logger,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])