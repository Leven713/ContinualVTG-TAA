import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from moment_detr_CL.start_end_dataset_all import \
    StartEndDataset, start_end_collate, prepare_batch_inputs

from moment_detr_CL.start_end_dataset_manager import DataManager

from moment_detr_CL.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from moment_detr_CL.continual_trainer import ContinualTrainer
from moment_detr_CL.ewc_regularizer import EWC_Regularizer


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


###### Elastic Weights Consolidation ######
class EWC(ContinualTrainer):
    def __init__(self):
        super().__init__()
        self.ewc_regularizer = EWC_Regularizer(self.opt.online)
        self.sample_size = self.opt.sample_size  
        self.ewc_loss_coef = self.opt.ewc_loss_coef 

    # Override compute_loss func
    def compute_loss(self, model, criterion, model_inputs, targets):
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        
        # ewc loss
        if self.ewc_regularizer.tasks_encountered == []:
            loss_ewc = None
        else:
            loss_ewc = self.ewc_regularizer.regularize(model.named_parameters())

        return loss_dict, loss_ewc

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
            loss_dict, loss_ewc = self.compute_loss(model, criterion, model_inputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            if loss_ewc is not None:
                # print(loss_ewc)
                # print(losses)
                losses += self.ewc_loss_coef * loss_ewc
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
            if loss_ewc is not None:
                loss_meters['loss_ewc'].update(loss_ewc)
                # print('loss_replay:', loss_meters['loss_replay'].avg)

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


    def train(self, model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
        if opt.device.type == "cuda":
            logger.info("CUDA enabled.")
            model.to(opt.device)

        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
        opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
        opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )

        prev_best_score = 0.
        es_cnt = 0
        # start_epoch = 0
        if opt.start_epoch is None:
            start_epoch = -1 if opt.eval_untrained else 0
        else:
            start_epoch = opt.start_epoch
        save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
        for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
            if epoch_i > -1:
                self.train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
                lr_scheduler.step()
            eval_epoch_interval = 5
            if opt.eval_path is not None and ((epoch_i + 1) % eval_epoch_interval == 0 or (epoch_i + 1) == opt.n_epoch):
                with torch.no_grad():
                    metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

                # log
                to_write = opt.eval_log_txt_formatter.format(
                    time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                    epoch=epoch_i,
                    loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                    eval_metrics_str=json.dumps(metrics_no_nms))

                with open(opt.eval_log_filepath, "a") as f:
                    f.write(to_write)
                logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
                if metrics_nms is not None:
                    logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

                metrics = metrics_no_nms
                for k, v in metrics["brief"].items():
                    tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

                if opt.dset_name in ['tvsum', 'youtube_uni']:
                    stop_score = metrics["brief"]["mAP"] 
                else:
                    stop_score = metrics["brief"]["MR-full-mAP"]
                    
                if stop_score > prev_best_score:
                    es_cnt = 0
                    prev_best_score = stop_score

                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch_i,
                        "opt": opt
                    }
                    torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                    best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                    for src, tgt in zip(latest_file_paths, best_file_paths):
                        os.renames(src, tgt)
                    logger.info("The checkpoint file has been updated.")
                else:
                    es_cnt += 1
                    if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                        with open(opt.train_log_filepath, "a") as f:
                            f.write(f"Early Stop at epoch {epoch_i}")
                        logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                        break

                # save ckpt
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

            if opt.debug:
                break

        tb_writer.close()


        # EWC Fisher matrix estimation
        if self.cur_task < self.opt.task_num:
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            self.ewc_regularizer.update_fisher_optpar(model, criterion,
                                                      self.cur_task, train_loader, 
                                                      sample_size=self.sample_size,
                                                      opt = opt)
            print(' Done!')

        


if __name__ == '__main__':
    trainer = EWC()
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
