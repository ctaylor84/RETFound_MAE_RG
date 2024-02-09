# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycm import *
import numpy as np




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, norm_params,
                    max_norm: float = 0, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.float, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples).squeeze()
            loss = criterion(outputs, (targets - norm_params["mean"]) / norm_params["std"])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, criterion, norm_params, device, task, epoch, mode):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_list = list()
    target_list = list()
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images).squeeze()
            loss = criterion(output, (target - norm_params["mean"]) / norm_params["std"])
            output_np = ((output * norm_params["std"]) + norm_params["mean"]).cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            prediction_list.append(output_np)
            target_list.append(target_np)

        batch_rmse = mean_squared_error(target_np, output_np, squared=False)
        batch_r2 = r2_score(target_np, output_np)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['rmse'].update(batch_rmse, n=batch_size)
        metric_logger.meters['r2'].update(batch_r2, n=batch_size)
    
    prediction_list = np.concatenate(prediction_list)
    target_list = np.concatenate(target_list)
    
    rmse = mean_squared_error(target_list, prediction_list, squared=False)
    mse = mean_squared_error(target_list, prediction_list)
    mae = mean_absolute_error(target_list, prediction_list)
    r2 = r2_score(target_list, prediction_list)

    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - RMSE: {:.4f} MSE: {:.4f} MAE: {:.4f} r2: {:.4f}'.format(rmse, mse, mae, r2)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[rmse, mse, mae, r2, metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mse

