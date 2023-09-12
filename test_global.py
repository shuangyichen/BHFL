import torch
import importlib
import torch.nn as nn
from data_loader import load_partition_data_cifar10

from utils.config import FLAGS
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
    
def get_model():
    """get model"""
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
    res = [0,0,0,0,0]
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            # data_time.update(time.time() - end)

            # if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            pred = model(inputs)
            loss = criterion(pred, targets)
            _, predicted = pred.topk(5, 1, True, True)
            predicted = predicted.t()
            # _, predicted = torch.max(pred, -1)
            # correct = predicted.eq(targets).sum()
            correct = (predicted == targets.unsqueeze(dim=0)).expand_as(predicted)

            # metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * targets.size(0)
            metrics['test_total'] += targets.size(0)
            # res = []
            for i,k in enumerate([1,2,3,4,5]):
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res[i]+=correct_k
                # res.append(correct_k.mul_(1.0 / batch_size))


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
    
        return metrics['test_loss']/float(metrics['test_total']), res[0]/float(metrics['test_total']),res[1]/float(metrics['test_total']),res[2]/float(metrics['test_total']),res[3]/float(metrics['test_total']),res[4]/float(metrics['test_total'])


model = get_model()

pretrained_file = "checkpoint_cifar10_resnet_fed/global_iter_1385_"
checkpoint = torch.load(
    pretrained_file, map_location=lambda storage, loc: storage)
# update keys from external models
if type(checkpoint) == dict and 'state_dict' in checkpoint:
    print(1)
    checkpoint = checkpoint['state_dict']
# if getattr(FLAGS, 'pretrained_model_remap_keys', False):
    new_checkpoint = {}
    new_keys = list(model.state_dict().keys())
    old_keys = list(checkpoint.keys())
    for key_new, key_old in zip(new_keys, old_keys):
        new_checkpoint[key_new] = checkpoint[key_old]
        # mprint('remap {} to {}'.format(key_new, key_old))
    checkpoint = new_checkpoint
model.load_state_dict(checkpoint)
# mprint('Loaded model {}.'.format(pretrained_file))
partition_method = "homo" #"hetero"
partition_alpha = 0.5
dataset = "cifar10"
# model.load_state_dict(averaged_params)

train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(dataset, FLAGS.data_dir, partition_method,
                                partition_alpha, 1, FLAGS.batch_size)
loss,test_acc1,test_acc2,test_acc3,test_acc4,test_acc5 = server_test(test_data_global,model)
print(loss)
print(test_acc1.item())
print(test_acc2.item())
print(test_acc3.item())
print(test_acc4.item())
print(test_acc5.item())
# logger.info('Server Test. \t Loss:  {:.6f}  \tAccuracy: {:.6f}'.format(loss,test_acc))