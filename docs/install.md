### install mmdetection3d
```shell
conda create --name SGCDet python=3.8.5
conda activate SGCDet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
cd packages/mmdetection3d
pip install -v -e .
cd ../../
```

### install pytorch_lightning
```shell
python -m pip install pip==19.2
pip install torchmetrics==0.11.4 pytorch_lightning==1.7.0 setuptools==59.5.0
pip install numpy==1.23.5 yapf==0.40.1
```

### install DFA3D
```shell
cd packages/3D-deformable-attention/DFA3D/
bash setup.sh 0
cd ../
python unittest_DFA3D.py
cd ../../
```
