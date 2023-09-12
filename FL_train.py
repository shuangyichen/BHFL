import argparse
import importlib
import os
import logging
import sys
import os
import shutil
# from logger import Logger
import torch
import torch.nn as nn
from utils.distributed import init_dist
from utils.config import FLAGS
from utils.distributed import AllReduceDistributedDataParallel
import numpy as np
from torchvision import datasets, transforms
from data_loader import load_partition_data_cifar10
from client import ClientManager
from utils.model_profiling import model_profiling
import time
# 1.  Init Server global model
# 2.  Init Client Class
# 3.  Assign data (IID or non-IID)
# 3.  Start training
#     3.1  Server quantizes full precision model to different bitwidth models
#     3.2  Send to corresponding clients
#     3.3  Client local training, push model
#     3.4  Server Aggregates the quantized model
class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results

# def server_test(loader, model, criterion):
def server_test(loader, model):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    # end = time.time()
    metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

    model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            # data_time.update(time.time() - end)

            # if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            pred = model(inputs)
            loss = criterion(pred, targets)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(targets).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * targets.size(0)
            metrics['test_total'] += targets.size(0)


            # if (batch_index + 1) %  == 0:
            #     logging.getLogger().info(
            #         '({batch:3d}/{size:3d}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            #             batch=batch_index + 1,
            #             size=len(loader),
            #             data=data_time.avg,
            #             bt=batch_time.avg,
            #             loss=losses.avg,
            #             top1=top1.avg,
            #             top5=top5.avg,
            #         ))
        return metrics['test_loss']/float(metrics['test_total']), float(metrics['test_correct'])/float(metrics['test_total'])


def transform_list_to_tensor(model_params_list):
        for k in model_params_list.keys():
            model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
        return model_params_list

def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].cpu().detach().numpy().tolist()
    return model_params


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'global_model_best.pth.tar'))

def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    print("client_indexes = ", str(client_indexes))
    return client_indexes

def get_model():
    """get model"""
    torch.manual_seed(FLAGS.random_seed)
    torch.cuda.manual_seed(FLAGS.random_seed)
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    # if getattr(FLAGS, 'distributed', False):
    #     gpu_id = init_dist()
    #     if getattr(FLAGS, 'distributed_all_reduce', False):
    #         model_wrapper = AllReduceDistributedDataParallel(model.cuda())
    #     else:
    #         model_wrapper = torch.nn.parallel.DistributedDataParallel(
    #             model.cuda(), [gpu_id], gpu_id)
    # else:
    #     print("get model")
    #     model_wrapper = torch.nn.DataParallel(model).cuda()
    return model#, model_wrapper


# ================================
# Init server model

checkpoint_path = "checkpoint_cifar10_resnet_fed"
make_directory(checkpoint_path)
logging.basicConfig(level=logging.INFO, filename="log.txt", filemode='a', format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(f"Start of target model training with {FLAGS.num_clients} clients")
print("Model preparation ...")
model = get_model()

# ================================
# Generate IID/non-IID data
print("Data Partitioning ...")
partition_method = "homo" #"hetero"
partition_alpha = 0.5
dataset = "cifar10"
train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(dataset, FLAGS.data_dir, partition_method,
                                partition_alpha, FLAGS.num_clients, FLAGS.batch_size)



# print(train_data_num)
# ================================
# Init Client class
# logger_path = os.path.join('output', 'runs', 'train_{}'.format(f'{dataset}_{partition_method}'))
# logger = Logger(logger_path)
print("Init workers ...")
clients = {}
for i in range(FLAGS.num_clients):
    clients[i] = ClientManager(train_data_local_dict,train_data_local_num_dict,test_data_local_dict,FLAGS.device,logger,checkpoint_path)

# # Training
assert len(FLAGS.bits_list)==FLAGS.num_workers, "Bits length and num of workers not equal"
# print(torch.cuda.get_arch_list())

best_acc = 0
for iter in range(FLAGS.num_iters):
    logger.info(f"Iteration: [%d | %d]" % (iter + 1, FLAGS.num_iters))
    # Server samples clients
    q_model_set = []
    model_aggregate = dict()
    model_list = []
    client_indexes = client_sampling(i, FLAGS.num_clients,FLAGS.num_workers)

    model_params = transform_tensor_to_list(model.state_dict())
    for i,index in enumerate(client_indexes):
        logger.info('{}-bit model training'.format(FLAGS.bits_list[i]))
        inx = FLAGS.bits_index[i]
        if inx-1>=0:
            clients[i].update(index, model_params,FLAGS.bits_list[i],FLAGS.bits[inx-1])
        else:
            clients[i].update(index, model_params,FLAGS.bits_list[i],32)
        model_aggregate[i] = transform_list_to_tensor(clients[i].step(iter))

        model_list.append((index, model_aggregate[i]))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            beta = 1.0 / float(FLAGS.num_workers)
            if i == 0:
                averaged_params[k] = local_model_params[k] * beta
            else:
                averaged_params[k] += local_model_params[k] * beta

    model.load_state_dict(averaged_params)
    loss,test_acc = server_test(test_data_global,model)
    logger.info('Server Test. \t Loss:  {:.6f}  \tAccuracy: {:.6f}'.format(loss,test_acc))
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': iter + 1,
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': is_best,
    }, is_best, filename='global_iter_%d_' % (iter), checkpoint=checkpoint_path)
    # print("Iterarion done")
    # Extract model parameters



    # Client update local model parameters and data
    # Client training
    