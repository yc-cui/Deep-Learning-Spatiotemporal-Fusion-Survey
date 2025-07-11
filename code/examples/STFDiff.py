
import argparse
import pytorch_lightning as pl
import os
import os, sys


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)


from LibSTFv1.dataset.name2data import name2data
from LibSTFv1.model.STFDiff.STFDiff import STFDiffModel

from pytorch_lightning import seed_everything
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from LibSTFv1.dataset.SlideDataset import plNBUDataset

from LibSTFv1.util.misc import check_and_make



def get_args_parser():
    parser = argparse.ArgumentParser('LibSTFv1 training', add_help=False)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--dataset', default="WH", choices=["CIA", "LGC", "AHB", "DX", "TJ", "WH", "IC", "BC"], type=str)
    parser.add_argument('--test_freq', default=10, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=44, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = "STF." + STFDiffModel.__name__

    #output_dir = f"log/log_m={model_name}_sd={args.seed}_d={args.dataset}"
    
    model_short_name = {
    "STF.ECPW_STFNModel": "ECPW_STFN",
    "STF.EDCSTFNModel": "EDC_STFN", 
    "STF.GANSTFMModel": "GAN_STFM",
    "STF.MLFFGANModel": "MLFFGAN",
    "STF.SRSFGANModel": "SRSFGAN",
    "STF.STFDiffModel": "STFDiff",
    "STF.SwinSTFMModel": "SwinSTFM"
    }

    short_name = model_short_name.get(model_name, model_name.split('.')[-1])

    output_dir = f"log/{args.dataset}/{short_name}/log_m={model_name}_sd={args.seed}_d={args.dataset}"

    check_and_make(output_dir)
    seed_everything(args.seed)

    dataset = plNBUDataset(name2data[args.dataset],
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem
                           )
    model = STFDiffModel(
                    epochs=args.epochs,
                    bands=name2data[args.dataset]["band"],
                    rgb_c=name2data[args.dataset]["rgb_c"],
                    dataname=args.dataset
                    )

    if args.wandb:
        wandb_logger = WandbLogger(project=model_name, name=output_dir, save_dir=output_dir)
    else:
        wandb_logger = [CSVLogger(name=output_dir, save_dir=output_dir)]
        wandb_logger.append(TensorBoardLogger(name=output_dir, save_dir=output_dir))
                
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/PSNR_mean',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_PSNR={val/PSNR_mean:.4f}',
                                       save_last=True,
                                       every_n_epochs=args.test_freq
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[args.device],
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                         )

    trainer.fit(model, dataset)
    trainer.test(ckpt_path="best", datamodule=dataset)
    #trainer.test(model, ckpt_path="log/WH/STFDiff/log_m=STF.STFDiffModel_sd=44_d=WH/ep=29_PSNR=28.6469.ckpt", datamodule=dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
