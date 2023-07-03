import argparse
import importlib
import os
from logger import Logger
import torch
import torch.nn as nn
from utils.distributed import init_dist
from utils.config import FLAGS
from utils.distributed import AllReduceDistributedDataParallel
import numpy as np
from torchvision import datasets, transforms
from .data_loader import load_partition_data_cifar10
from .client import ClientManager
# 1.  Init Server global model
# 2.  Init Client Class
# 3.  Assign data (IID or non-IID)
# 3.  Start training
#     3.1  Server quantizes full precision model to different bitwidth models
#     3.2  Send to corresponding clients
#     3.3  Client local training, push model
#     3.4  Server Aggregates the quantized model

def add_args(parser):
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_clients', default='avg', type=str)
    parser.add_argument('--iterations', default=None, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_dir', default=None, type=str)
    args = parser.parse_args()
    return args


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist()
        if getattr(FLAGS, 'distributed_all_reduce', False):
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id)
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


parser = argparse.ArgumentParser()
args = add_args(parser)
# ================================
# Init server model
model, model_wrapper = get_model()

# ================================
# Generate IID/non-IID data
partition_method = "homo" #"hetero"
partition_alpha = 0.5
dataset = "cifar10"
train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(dataset, args.data_dir, partition_method,
                                partition_alpha, args.num_clients, args.batch_size)
# ================================
# Init Client class
logger_path = os.path.join('output', 'runs', 'train_{}'.format(f'{dataset}_{partition_method}'))
logger = Logger(logger_path)

clients = {}
for i in range(args.num_clients):
    clients[i] = ClientManager(logger_path,args)

# Training
for i in range(args.epochs):
    # Server quantize full-precision model to different bit-width models
    for bits_idx, bits in enumerate(FLAGS.bits_list):
        q_model = model.apply(lambda m: setattr(m, 'bits', bits))
    # Server samples clients

    # Client update local model parameters and data
    # Client training
    