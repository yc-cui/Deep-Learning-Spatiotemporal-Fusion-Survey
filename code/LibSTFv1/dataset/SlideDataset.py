from torch.utils.data import Dataset
import os
import tifffile
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

def _slide(x, k):
    B, C, H, W = x.shape
    patches = x.unfold(2, k, k).unfold(3, k, k).permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, k, k)
    return patches


class SlideDataset(Dataset):
    def __init__(self, pairs, img_size, slide_size, max_val):
        self.pairs = pairs
        self.slide_size = slide_size
        self.img_size = img_size
        
        LRs_arr = []
        HRs_arr = []
        for LR, HR in zip(pairs['LRs'], pairs['HRs']):
            LR_data = np.array(tifffile.imread(LR))
            HR_data = np.array(tifffile.imread(HR))
            
            if len(LR_data.shape) == 3 and LR_data.shape[-1] <= 6:  # (H,W,C)
                LR_data = np.transpose(LR_data, (2, 0, 1))
                HR_data = np.transpose(HR_data, (2, 0, 1))
            elif len(LR_data.shape) == 3 and LR_data.shape[0] <= 6: 
                pass
            else:
                raise ValueError(f"不支持的数据格式: {LR_data.shape}")
                
            LR_data[LR_data < 0] = 0
            HR_data[HR_data < 0] = 0
            
            LRs_arr.append(_slide(torch.from_numpy(LR_data).unsqueeze(0), k=slide_size).numpy() / max_val)
            HRs_arr.append(_slide(torch.from_numpy(HR_data).unsqueeze(0), k=slide_size).numpy() / max_val)
        
        self.LRs_arr = np.array(LRs_arr).astype(np.float32)
        self.HRs_arr = np.array(HRs_arr).astype(np.float32)
        
        print(self.LRs_arr.shape)
        print(self.HRs_arr.shape)
    
    def __len__(self):
        return (self.LRs_arr.shape[0] - 1) * self.LRs_arr.shape[1]
    
    def __getitem__(self, index):
        
        len_patch = self.LRs_arr.shape[1]
        
        LR_t1 = torch.from_numpy(self.LRs_arr[index//len_patch][index%len_patch])
        HR_t1 = torch.from_numpy(self.HRs_arr[index//len_patch][index%len_patch])

        LR_t2 = torch.from_numpy(self.LRs_arr[index//len_patch + 1][index%len_patch])
        HR_t2 = torch.from_numpy(self.HRs_arr[index//len_patch + 1][index%len_patch])

        return {
            "LR_t1": LR_t1,
            "HR_t1": HR_t1,
            "LR_t2": LR_t2,
            "HR_t2": HR_t2,
        }




class plNBUDataset(pl.LightningDataModule):
    def __init__(self, dataset_dict, batch_size, num_workers=4, pin_memory=True, ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        train_paths, val_paths = dataset_dict["train"], dataset_dict["val"]
        img_size, slide_size = dataset_dict["img_size"], dataset_dict["slide_size"]
        max_val = dataset_dict["max_value"]
        self.dataset_train = SlideDataset(train_paths, img_size, slide_size, max_val)
        self.dataset_val = SlideDataset(val_paths, img_size, slide_size, max_val)
        self.dataset_test = SlideDataset(val_paths, img_size, slide_size, max_val)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
