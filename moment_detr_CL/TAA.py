import os
import copy
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from moment_detr_CL.start_end_dataset_all import \
    StartEndDataset, start_end_collate, prepare_batch_inputs


from moment_detr_CL.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from moment_detr_CL.continual_trainer import ContinualTrainer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)




###### Temporal Aware Augmentation (TAA)
class TAA(ContinualTrainer):
    def __init__(self):
        super().__init__()
        self.taa_loss_coef1 = self.opt.taa_loss_coef1
        self.taa_loss_coef2 = self.opt.taa_loss_coef2

    # Override compute_loss func
    def compute_loss(self, model, criterion, model_inputs, targets):
        outputs, (enc_feat, _) = model(**model_inputs, get_mid_outputs=True)
        loss_dict = criterion(outputs, targets)

        if self.cur_task > 0:
            # forward pass of TAA
            aug_outputs, (enc_feat_aug, _) = model.forward_taa(**model_inputs, get_mid_outputs=True)
            # token-level loss
            loss_taa_token = torch.norm(enc_feat - enc_feat_aug, dim=-1).mean()
            # prediction-level loss
            loss_dict_pred = criterion(aug_outputs, targets)
        else:
            loss_taa_token = None
            loss_dict_pred = None

        return loss_dict, (loss_taa_token, loss_dict_pred)
    

    # Override train_epoch func
    def train_epoch(self, model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
        logger.info(f"[Epoch {epoch_i+1}]")
        model.train()
        criterion.train()

        # init meters
        time_meters = defaultdict(AverageMeter)
        loss_meters = defaultdict(AverageMeter)

        num_training_examples = len(train_loader)
        timer_dataloading = time.time()
        for batch_idx, batch in tqdm(enumerate(train_loader),
                                    desc="Training Iteration",
                                    total=num_training_examples):
            time_meters["dataloading_time"].update(time.time() - timer_dataloading)

            timer_start = time.time()
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
            time_meters["prepare_inputs_time"].update(time.time() - timer_start)

            timer_start = time.time()
            loss_dict, (loss_taa_token, loss_dict_pred) = self.compute_loss(model, criterion, model_inputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if loss_taa_token is not None:
                # calculate loss_taa_pred
                loss_taa_pred = sum(loss_dict_pred[k] * weight_dict[k] for k in loss_dict_pred.keys() if k in weight_dict)
                losses += self.taa_loss_coef1 * loss_taa_token + self.taa_loss_coef2 * loss_taa_pred

            time_meters["model_forward_time"].update(time.time() - timer_start)

            timer_start = time.time()
            optimizer.zero_grad()
            losses.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            time_meters["model_backward_time"].update(time.time() - timer_start)

            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))
            if loss_taa_token is not None:
                loss_meters['loss_taa_token'].update(loss_taa_token)
                loss_meters['loss_taa_pred'].update(loss_taa_pred)

            timer_dataloading = time.time()
            if opt.debug and batch_idx == 3:
                break

        # print/add logs
        tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i+1,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)

        logger.info("Epoch time stats:")
        for name, meter in time_meters.items():
            d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
            logger.info(f"{name} ==> {d}")

        


if __name__ == '__main__':
    trainer = TAA()
    best_ckpt_path, eval_split_name, eval_path, debug = trainer.start_training()

    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()
