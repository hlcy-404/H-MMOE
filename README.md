# Challenging Dataset and Filter-Group Mixture-of-Experts Model for Remote Sensing Forged Target Perception Understanding

This is the initial version of the Real-RSCM dataset and FG-MoE Framework. 

### Installation

```
conda create -n MMoE python=3.11
conda activate MMoE
```

##### pytorch

[**install pytorch**](https://pytorch.org/)

```
# e.g. CUDA 11.8
# with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# with pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

##### Install Packages

```
pip install -r requirements.txt
```

### Download Datasets

- [**Datasets V1.0 is released at Baidu Cloud**](https://pan.baidu.com/s/1itum7p1b5_4vKFCaskPgyQ?pwd=real) and [**Google Drive**](https://drive.google.com/drive/folders/1uSCa8U0jGs2QHPB34zJ6jfslCvE-sCLq?usp=drive_link) (2024.12.25)
- Dataset Directory: ` datasets/`
- Dataset Subdirectory: `datasets/JsonFiles/`,  `datasets/image/`, `datasets/source/`, `datasets/target/`, `datasets/background/`, `datasets/segmentation/`


### Download pre-trained weights

[**Download clip-b-32 weights from Hugging Face**](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)

- Clip Directory: `models/clipModels/openai_clip_b_32/`

[**Download U-Net weights from Github**](https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth) 

- U-Net Directory: `models/imageModels/milesial_UNet/`

### Start Training

```
python main.py
```

- Modify the experiment settings and hyperparameters in `src/config.py`

### Data Examples

![](https://i.meee.com.tw/gxerqE2.png)

### Citation

### License

All images and their associated annotations in Global-TQA can be used for academic purposes only, **but any commercial use is prohibited.**
