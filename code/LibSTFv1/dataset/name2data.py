#LibSTFv1/dataset/name2data.py
from datetime import datetime

from torch.utils.data import Dataset
import os
import torch
import numpy as np


def get_pairs(data_path, num_train, key_Landsat, key_MODIS):
    get_sorted_files = lambda x,k: sorted(os.listdir(x), key=k)
    LR_sorted_files = get_sorted_files(os.path.join(data_path, "MODIS"), key_MODIS)
    HR_sorted_files = get_sorted_files(os.path.join(data_path, "Landsat"), key_Landsat)
    LRs = []
    HRs = []
    for LR_name, HR_name in zip(LR_sorted_files, HR_sorted_files):
        LRs.append(os.path.abspath(os.path.join(data_path, "MODIS", LR_name)))
        HRs.append(os.path.abspath(os.path.join(data_path, "Landsat", HR_name)))

    train_val_dict =  {
        "train": {
            "LRs": LRs[:num_train],
            "HRs": HRs[:num_train],
        },
        "val":{
            "LRs": LRs[num_train:],
            "HRs": HRs[num_train:],
        }
    }

    # print(train_val_dict)

    return train_val_dict



CIA_data = get_pairs(data_path="data/CIA", 
                    num_train=11, 
                    key_Landsat=lambda x: x[13:21],
                    key_MODIS=lambda x: x[9:16],
                    )

LGC_data = get_pairs(data_path="data/LGC", 
                    num_train=9, 
                    key_Landsat=lambda x: x[:8],
                    key_MODIS=lambda x: x[9:16],
                    )

AHB_data = get_pairs(data_path="data/Datasets/AHB", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

Daxing_data = get_pairs(data_path="data/Datasets/Daxing", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

Tianjin_data = get_pairs(data_path="data/Datasets/Tianjin", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

Wuhan_data = get_pairs(data_path="data/Wuhan",
                    num_train=5,
                    key_Landsat=lambda x: x[2:],
                    key_MODIS=lambda x: x[2:],
                    )

IC_data = get_pairs(data_path="data/IC",
                    num_train=3, 
                    key_Landsat=lambda x: x[3:11],
                    key_MODIS=lambda x: x[3:11],
                    )

BC_data = get_pairs(data_path="data/BC",
                    num_train=3,
                    key_Landsat=lambda x: x[3:11],
                    key_MODIS=lambda x: x[3:11],
                    )

name2data = {
    "CIA": {
        "train": CIA_data["train"],
        "val": CIA_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1792, 1280),
        "slide_size": 512,
        "max_value": 10000.,
        "key_Landsat": lambda x: x[13:21],
    },
    "LGC": {
        "train": LGC_data["train"],
        "val": LGC_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2560, 3072),
        "slide_size": 512,
        "max_value": 10000.,
        "key_Landsat": lambda x: x[:8],
    },
    "AHB": {
        "train": AHB_data["train"],
        "val": AHB_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2480, 2800),
        "slide_size": 512,
        "max_value": 255.,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
    },
    "DX": {
        "train": Daxing_data["train"],
        "val": Daxing_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1640, 1640),
        "slide_size": 512,
        "max_value": 255.,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
    },
    "TJ": {
        "train": Tianjin_data["train"],
        "val": Tianjin_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2100, 1970),
        "slide_size": 512,
        "max_value": 255.,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
    },
    "WH": {
        "train": Wuhan_data["train"],
        "val": Wuhan_data["val"], 
        "band": 4,  
        "rgb_c": [2, 1, 0],  
        "img_size": (1000, 1000),
        "slide_size": 512,
        "max_value": 10000.,
        "key_Landsat": lambda x: x[2:],
    },
    "IC": {
        "train": IC_data["train"],
        "val": IC_data["val"],
        "band": 4,
        "rgb_c": [2, 1, 0],  
        "img_size": (1500, 1500),
        "slide_size": 512,
        "max_value": 1.0,
        "key_Landsat": lambda x: x[3:11],
    },
    "BC": {
        "train": BC_data["train"],
        "val": BC_data["val"],
        "band": 4,
        "rgb_c": [2, 1, 0],
        "img_size": (1500, 1500),
        "slide_size": 512,
        "max_value": 1.0,
        "key_Landsat": lambda x: x[3:11],
    }
}
