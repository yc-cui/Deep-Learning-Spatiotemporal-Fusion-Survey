# A Decade of Deep Learning for Remote Sensing Spatiotemporal Fusion: Advances, Challenges, and Opportunities

This repository provides a comprehensive implementation of deep learning models for remote sensing spatiotemporal fusion, featuring state-of-the-art architectures and evaluation frameworks.

## 📋 Table of Contents

- [Environment Setup](#🚀-environment-setup)
- [Dataset Configuration](#📊-dataset-configuration)
- [Model Architecture](#🏗️-model-architecture)
- [Training and Testing](#🎯-training-and-testing)
- [Evaluation Metrics](#📈-evaluation-metrics)
- [Logging and Monitoring](#📊-logging-and-monitoring)
- [Citation](#📚-citation)

## 🚀 Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yc-cui/Deep-Learning-Spatiotemporal-Fusion-Survey
cd code
```

2. **Create and activate conda environment:**
```bash
# Create environment from YAML file
conda env create -f stf-fusion.yaml

# Activate environment
conda activate stf-fusion
```


### Environment Details
The `stf-fusion.yaml` file contains all necessary dependencies including:
- PyTorch with CUDA support
- PyTorch Lightning for training
- TensorBoard and WandB for logging
- Image processing libraries (tifffile, PIL)
- Scientific computing (numpy, scipy)

## 📊 Dataset Configuration

### Supported Datasets

| Dataset | Bands | Image Size | Normalized Value |
|---------|-------|------------|-----------|
| CIA | 6 | 1792×1280 | 10000 |
| LGC | 6 | 2560×3072 | 10000 |
| AHB | 6 | 2480×2800 | 255 |
| DX | 6 | 1640×1640 | 255 |
| TJ | 6 | 2100×1970 | 255 |
| WH | 4 | 1000×1000 | 10000 |
| IC | 4 | 1500×1500 | 1.0 |
| BC | 4 | 1500×1500 | 1.0 |

### Dataset Structure
```
data/
├── CIA/
│   ├── Landsat/     # High-resolution images
│   │   ├── L71093084_08420020401_HRF_modtran_surf_ref_agd66.tif
│   │   ├── L71093084_08420020410_HRF_modtran_surf_ref_agd66.tif
│   │   └── ...
│   └── MODIS/       # Low-resolution images
│       ├── MOD09GA_A2001290.sur_refl.tif
│       ├── MOD09GA_A2002044.sur_refl.tif
│       └── ...
├── LGC/
│   ├── Landsat/
│   │   ├── 20050403_TM.tif
│   │   ├── 20041228_TM.tif
│   │   └── ...
│   └── MODIS/
│       ├── MOD09GA_A2005013.sur_refl.tif
│       ├── MOD09GA_A2004347.sur_refl.tif
│       └── ...
├── Datasets/
│   ├── AHB/
│   │   ├── Landsat/
│   │   │   ├── L_2015-6-21.tif
│   │   │   ├── L_2017-6-10.tif
│   │   │   └── ...
│   │   └── MODIS/
│   │       ├── M_2016-4-20.tif
│   │       ├── M_2017-7-28.tif
│   │       └── ...
│   ├── Daxing/
│   └── Tianjin/
├── Wuhan/
│   ├── Landsat/
│   │   ├── G_20180109.tif
│   │   ├── G_20210118.tif
│   │   └── ...
│   └── MODIS/
│       ├── L_20210118.tif
│       ├── L_20211220.tif
│       └── ...
├── IC/
│   ├── Landsat/
│   │   ├── S2_20220405.tif
│   │   ├── S2_20220505.tif
│   │   └── ...
│   └── MODIS/
│       ├── S3_20220105_real.tif
│       ├── S3_20220604_real.tif
│       └── ...
└── BC/
    ├── Landsat/
    │   ├── S2_20220713.tif
    │   ├── S2_20220822.tif
    │   └── ...
    └── MODIS/
        ├── S3_20221001_real.tif
        ├── S3_20220713_real.tif
        └── ...
```


## 🏗️ Model Architecture

### Available Models

| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **ECPW-STFN** | Parallel Wavelet + Refinement | Wavelet decomposition, parallel processing |
| **EDC-STFN** | Encoder-Decoder + Cross-Scale | Multi-scale feature fusion |
| **GAN-STFM** | Generative Adversarial Network | Adversarial training, realistic generation |
| **MLFF-GAN** | Multi-Level Feature Fusion GAN | Hierarchical feature learning |
| **SRSF_GAN** | Spatial Resolution Enhancement GAN | Spatial detail enhancement |
| **STFDiff** | Diffusion Model | Denoising diffusion process |
| **SwinSTFM** | Swin Transformer | Self-attention, hierarchical structure |

### Model Location
All models are located in `LibSTFv1/model/` with the following structure:
```
LibSTFv1/model/
├── ECPW_STFN/
├── EDCSTFN/
├── GAN_STFM/
├── MLFF_GAN/
├── SRSF_GAN/
├── STFDiff/
├── SwinSTFM/
```

### 🚀 Adding Your Model
By following our standardized format, you can quickly integrate your model into the framework and compare results with other models.



## 🎯 Training and Testing

### Quick Start

1. **Training and testing a model:**
```bash
python examples/ECPW_STFN.py \
    --dataset AHB \
    --epochs 500 \
    --batch_size 6 \
    --device 0 \
    --seed 42
```


**Note**: When testing only, you need to modify the example script by:
- Commenting out: `trainer.fit(model, dataset)`
- Uncommenting: `trainer.test(model, ckpt_path="path/to/checkpoint.ckpt", datamodule=dataset)`

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | AHB | Dataset name |
| `--epochs` | 500 | Number of training epochs |
| `--batch_size` | 6 | Batch size |
| `--device` | 0 | GPU device ID |
| `--seed` | 42 | Random seed |
| `--test_freq` | 10 | Validation frequency |
| `--num_workers` | 8 | Data loader workers |
| `--wandb` | False | Enable WandB logging |

### Example Training Commands

```bash
# Train ECPW_STFN on AHB dataset
python examples/ECPW_STFN.py --dataset AHB --epochs 500 --device 0

# Train SwinSTFM on CIA dataset with WandB
python examples/SwinSTFM.py --dataset CIA --epochs 300 --wandb --device 1

# Train GAN_STFM on LGC dataset
python examples/GAN_STFM.py --dataset LGC --epochs 400 --batch_size 4
```

## 📈 Evaluation Metrics

### Metrics Supported
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **SAM**: Spectral Angle Mapper
- **ERGAS**: Erreur Relative Globale Adimensionnelle de Synthèse
- **RASE**: Relative Average Spectral Error
- **UQI**: Universal Quality Index
- **SCC**: Spatial Correlation Coefficient
- **CC**: Cross Correlation
- **R²**: Coefficient of Determination



## 📊 Logging and Monitoring

### Logging Options

Logs are automatically saved to `log/{dataset}/{model}/` directory

### Log Structure
```
log/
├── AHB/
│   ├── ECPW_STFN/
│   │   ├── log_m=STF.ECPW_STFNModel_sd=42_d=AHB/
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   └── metrics.csv
│   └── SwinSTFM/
└── CIA/
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{sun2025decadedeeplearningremote,
      title={A Decade of Deep Learning for Remote Sensing Spatiotemporal Fusion: Advances, Challenges, and Opportunities}, 
      author={Enzhe Sun and Yongchuan Cui and Peng Liu and Jining Yan},
      year={2025},
      eprint={2504.00901},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.00901}, 
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and researchers in the spatiotemporal fusion community
- Special thanks to the authors of the original datasets and baseline methods

## 📞 Contact

For questions and issues:
- Email: yongchuancui@gmail.com
