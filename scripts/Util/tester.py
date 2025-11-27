# C:\Workspace\Robosort\RoboSort\scripts\Util\tester.py

import pytorch_lightning as pl
import torch
from nanodet.util import cfg, load_config, NanoDetLightningLogger
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.data.collate import naive_collate

CFG_PATH = r"C:\Workspace\Robosort\RoboSort\nanodet_waybill_finetune.yml"
CKPT_PATH = r"C:\Workspace\Robosort\RoboSort\nanodet\workspace\waybill_finetune\model_last.ckpt"


def main():
    # 1) load config
    load_config(cfg, CFG_PATH)

    # 2) evaluator & task
    val_dataset = build_dataset(cfg.data.val, 'val')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    task = TrainingTask(cfg, evaluator=evaluator)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    # logger 세팅
    logger = NanoDetLightningLogger(cfg.save_dir)


    # 3) Trainer – validate mode only
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    # 4) 평가만 수행
    trainer.validate(task, val_dataloader, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    main()
