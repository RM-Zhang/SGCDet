### Dataset  
Please refer to [ImGeoNet](https://github.com/ttaoREtw/ImGeoNet) for instructions on preparing the datasets (ScanNet, ScanNet200, ARKitScenes).

### Training
We present two variants of our network: **SGCDet**, which computes a 3D volume with a spatial resolution of 40×40×16, and **SGCDet-L**, which uses a higher spatial resolution of 80×80×36.

Note that detection performance may vary depending on the number of GPUs used. Experimentally, we find that training with 2 GPUs yields better results.

#### ScanNet
```shell
# SGCDet
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path=configs/SGCDet_ScanNet.py --log_folder=SGCDet_ScanNet
```

#### ScanNet200
```shell
# SGCDet-L
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path=configs/SGCDet_large_ScanNet200.py --log_folder=SGCDet_large_ScanNet200
```

#### ARKitScenes
```shell
# SGCDet
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path=configs/SGCDet_ARKit.py --log_folder=SGCDet_ARKit
# SGCDet-L
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path=configs/SGCDet_large_ARKit.py --log_folder=SGCDet_large_ARKit
```

### Test
Please set the `--ckpt_path` argument to the path where the corresponding checkpoint is stored.

#### ScanNet
```shell
# SGCDet
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --config_path=configs/SGCDet_ScanNet.py --ckpt_path=path/to/ckpt/xxx.ckpt --log_folder=eval_SGCDet_ScanNet
```

#### ScanNet200
```shell
# SGCDet-L
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --config_path=configs/SGCDet_large_ScanNet200.py --ckpt_path=path/to/ckpt/xxx.ckpt --log_folder=eval_SGCDet_large_ScanNet200
```

#### ARKitScenes
```shell
# SGCDet
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --config_path=configs/SGCDet_ARKit.py --ckpt_path=path/to/ckpt/xxx.ckpt --log_folder=eval_SGCDet_ARKit
# SGCDet-L
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --config_path=configs/SGCDet_large_ARKit.py --ckpt_path=path/to/ckpt/xxx.ckpt --log_folder=eval_SGCDet_large_ARKit
```
