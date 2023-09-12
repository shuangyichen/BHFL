import datetime
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
# from data import SplitDataset, make_data_loader
# from logger import Logger
# from metrics import Metric
# from models import resnet
# from utils import make_optimizer, collate, to_device
from utils.config import FLAGS


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)

    return model#, model_wrapper

def save_checkpoint(state, client_id, bit, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        model_name = 'client_'+str(client_id)+ '_bit_'+str(bit) +'_model_best.pth.tar'
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, model_name))

# @ray.remote(num_gpus=0.15)
class ClientManager:
    def __init__(self,train_data_local_dict,train_data_local_num_dict,test_data_local_dict,device,logger,checkpoint_path):
        # with open('config.yml', 'r') as f:
        #     cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.local_parameters = None
        self.start_time = None
        self.num_active_users = None
        self.optimizer = None
        self.model = get_model()
        self.model2 = get_model()
        self.lr = None
        self.data_loader = None
        self.client_id = None
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.device = device
        self.optimizer = self.get_optimizer()
        self.logger = logger
        self.best_acc = 0
        self.checkpoint_path = checkpoint_path
        self.bitwidth = None
        self.bitwidth2 = None
        # cfg = cfg[0] #ray.get(cfg[0])
        # self.args = args

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def update(self, client_id,local_parameters, bit, bit2):

        # self.local_parameters = local_parameters
        self.client_id = client_id
        self.update_dataset(self.client_id)
        # self.lr = lr
        model_params = self.transform_list_to_tensor(local_parameters)
        self.set_model_params(model_params)
        self.bitwidth = bit
        self.bitwidth2 = bit2
        print("Teacher bitwidth ", self.bitwidth2)

    def transform_list_to_tensor(self,model_params_list):
        for k in model_params_list.keys():
            model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
        return model_params_list
        # self.metric = Metric()

    def train(self, train_data, device,iter):
        model = self.model
        model.cuda()
        model.train()

        criterion = nn.CrossEntropyLoss().cuda()#to(device)    

        if FLAGS.lr_scheduler == 'linear_decaying':
                linear_decaying_per_step = (
                    FLAGS.lr/FLAGS.num_iters/len(train_data.dataset)*FLAGS.batch_size)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] -= linear_decaying_per_step

        if FLAGS.knowledge_distillation:
            # if iter>100:
            if self.bitwidth2 == 32:
                model2 = self.model2

                model2.eval()
                model2.cuda()
                epoch_loss = []
                # model.apply(lambda m: setattr(m, 'bits', self.bitswidth))
                for epoch in range(FLAGS.num_epochs):
                    batch_loss = []
                    for batch_idx, (x, labels) in enumerate(train_data):
                        # logging.info(images.shape)
                        # x, labels = x.to(device), labels.to(device)
                        x, labels = x.cuda(), labels.cuda()
                        self.optimizer.zero_grad()

                        # loss = self.forward_loss(model, criterion, input, labels, meter)
                        # loss.backward()
                        # model2.apply(lambda m: setattr(m, 'bits', self.bitwidth2))
                        model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
                        log_probs = model(x)
                        log_probs2 = model2(x)
                        loss = criterion(log_probs, labels) + F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, -1),reduction="none").mean()
                        # print("KL divergence ",F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, -1),reduction="none").mean().item())
                        loss.backward()
                        self.optimizer.step()
                        batch_loss.append(loss.item())
                    if len(batch_loss) > 0:
                        epoch_loss.append(sum(batch_loss) / len(batch_loss))
                        self.logger.info('Trainer_ID {}.Bitwidth {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_id,self.bitwidth,
                                                                                                    epoch,
                                                                                                    sum(epoch_loss) / len(
                                                                                                        epoch_loss)))
            else:     
                model2 = self.model2

                model2.eval()
                model2.cuda()
                # cos = torch.nn.CosineSimilarity(dim=0).cuda()

                epoch_loss = []
                # model.apply(lambda m: setattr(m, 'bits', self.bitswidth))
                for epoch in range(FLAGS.num_epochs):
                    batch_loss = []
                    for batch_idx, (x, labels) in enumerate(train_data):
                        print(batch_idx)
                        # logging.info(images.shape)
                        # x, labels = x.to(device), labels.to(device)
                        x, labels = x.cuda(), labels.cuda()
                        self.optimizer.zero_grad()

                        # loss = self.forward_loss(model, criterion, input, labels, meter)
                        # loss.backward()
                        # model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
                        # log_probs = model(x)
                        model2.apply(lambda m: setattr(m, 'bits', self.bitwidth2))
                        log_probs2 = model2(x)
                        model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
                        log_probs = model(x)
                        # cos_loss = cos(F.softmax(log_probs2),F.softmax(log_probs))
                        # loss = criterion(log_probs2, labels) - 0.1* cos_loss.mean()#+ F.kl_div(F.log_softmax(log_probs2,-1), F.softmax(log_probs,-1),reduction="none").mean()
                        # print("Cosine Similarity loss",cos_loss.mean().item())
                        kl_loss = F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, -1),reduction="none").mean()
                        print("KL loss",kl_loss.item())
                        loss = criterion(log_probs, labels) + kl_loss
                        
                        # loss = F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, -1),reduction="none").mean()
                        # print("Regular loss",criterion(log_probs, labels).item())
                        # print("KL loss",kl_loss.item())
                        loss.backward()
                        self.optimizer.step()
                        batch_loss.append(loss.item())
                        print(batch_loss)
                    if len(batch_loss) > 0:
                        epoch_loss.append(sum(batch_loss) / len(batch_loss))
                        self.logger.info('Trainer_ID {}.Bitwidth {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_id,self.bitwidth,
                                                                                                    epoch,
                                                                                                        sum(epoch_loss) / len(
                                                                                                            epoch_loss)))
            # else:
            #     epoch_loss = []
            #         # model.apply(lambda m: setattr(m, 'bits', self.bitswidth))
            #     for epoch in range(FLAGS.num_epochs):
            #         batch_loss = []
            #         for batch_idx, (x, labels) in enumerate(train_data):
            #             # logging.info(images.shape)
            #             # x, labels = x.to(device), labels.to(device)
            #             x, labels = x.cuda(), labels.cuda()
            #             self.optimizer.zero_grad()

            #             # loss = self.forward_loss(model, criterion, input, labels, meter)
            #             # loss.backward()
            #             # model2.apply(lambda m: setattr(m, 'bits', self.bitwidth2))
            #             model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
            #             log_probs = model(x)
            #             # log_probs2 = model2(x)
            #             loss = criterion(log_probs, labels) #+ F.kl_div(F.log_softmax(log_probs2), F.softmax(log_probs, -1),reduction="none").mean()
            #             # print("Regular loss",criterion(log_probs, labels).item())
            #             # print("KL loss",F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, 0),reduction="none").mean())
            #             loss.backward()
            #             self.optimizer.step()
            #             batch_loss.append(loss.item())
            #         if len(batch_loss) > 0:
            #             epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #             self.logger.info('Trainer_ID {}.Bitwidth {} without teacher. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_id,self.bitwidth,
            #                                                                                         epoch,
            #                                                                                         sum(epoch_loss) / len(
                                                                                                        # epoch_loss)))
        else:
            epoch_loss = []
                    # model.apply(lambda m: setattr(m, 'bits', self.bitswidth))
            for epoch in range(FLAGS.num_epochs):
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(train_data):
                    # logging.info(images.shape)
                    # x, labels = x.to(device), labels.to(device)
                    x, labels = x.cuda(), labels.cuda()
                    self.optimizer.zero_grad()

                    # loss = self.forward_loss(model, criterion, input, labels, meter)
                    # loss.backward()
                    # model2.apply(lambda m: setattr(m, 'bits', self.bitwidth2))
                    model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
                    log_probs = model(x)
                    # log_probs2 = model2(x)
                    loss = criterion(log_probs, labels) #+ F.kl_div(F.log_softmax(log_probs2), F.softmax(log_probs, -1),reduction="none").mean()
                    # print("Regular loss",criterion(log_probs, labels).item())
                    # print("KL loss",F.kl_div(F.log_softmax(log_probs), F.softmax(log_probs2, 0),reduction="none").mean())
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    self.logger.info('Trainer_ID {}.Bitwidth {} without teacher. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_id,self.bitwidth,
                                                                                                epoch,
                                                                                                sum(epoch_loss) / len(
                                                                                                    epoch_loss)))

        

        weights = transform_tensor_to_list(self.model.cpu().state_dict())
        return weights
        # else:


 

    def step(self,iter):
        # model_param = 
        self.train(self.train_local,self.device,iter)
        metrics = self.test(self.test_local)
        test_acc = float(metrics['test_correct']) / float(metrics['test_total'])
        loss_avg = float(metrics['test_loss'])/float(metrics['test_total'])
        self.logger.info('Trainer_ID {}. Local Test: \tLoss: {:.6f}  \tAccuracy: {:.6f}'.format(self.client_id, loss_avg,test_acc))
        is_best = test_acc > self.best_acc
        self.best_acc = max(test_acc, self.best_acc)
        save_checkpoint({
            'epoch': iter + 1,
            'state_dict': self.model.state_dict(),
            'acc': test_acc,
            'best_acc': is_best,
        }, self.client_id,self.bitwidth,is_best, filename='client_%d_bit_%d_iter_%d_main' % (self.client_id,self.bitwidth,iter), checkpoint=self.checkpoint_path)
        model_param = transform_tensor_to_list(self.model.cpu().state_dict())
        return model_param

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        self.model2.load_state_dict(model_parameters)


    def get_optimizer(self):
    # """get optimizer"""
        if FLAGS.optimizer == 'sgd':
            # all depthwise convolution (N, 1, x, x) has no weight decay
            # weight decay only on normal conv and fc
            model_params = []
            for params in self.model.parameters():
                ps = list(params.size())
                if len(ps) == 4 and ps[1] != 1:
                    weight_decay = FLAGS.weight_decay
                elif len(ps) == 2:
                    weight_decay = FLAGS.weight_decay
                else:
                    weight_decay = 0
                item = {'params': params, 'weight_decay': weight_decay,
                        'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                        'nesterov': FLAGS.nesterov}
                model_params.append(item)
            optimizer = torch.optim.SGD(model_params)
        elif FLAGS.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=FLAGS.lr, alpha=FLAGS.optim_decay, eps=FLAGS.optim_eps, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
        else:
            try:
                optimizer_lib = importlib.import_module(FLAGS.optimizer)
                return optimizer_lib.get_optimizer(self.model)
            except ImportError:
                raise NotImplementedError(
                    'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
        return optimizer

    def test(self, test_data):
        model = self.model

        model.eval()
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()


        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.cuda()
                target = target.cuda()
                model.apply(lambda m: setattr(m, 'bits', self.bitwidth))
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics




    # def pull(self):
    #     model_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
    #     return model_state



    # def pull(self):
    #     model_state = {k: v.detach().clone().cpu() for k, v in self.model.to(self.cfg['device']).state_dict().items()}
    #     return model_state

    # def log(self, epoch, cfg):
    #     if self.m % int((self.num_active_users * cfg['log_interval']) + 1) == 0:
    #         local_time = (time.time() - self.start_time) / (self.m + 1)
    #         epoch_finished_time = datetime.timedelta(seconds=local_time * (self.num_active_users - self.m - 1))
    #         exp_finished_time = epoch_finished_time + datetime.timedelta(
    #             seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * self.num_active_users))
    #         info = {'info': ['Model: {}'.format(cfg['model_tag']),
    #                          'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * self.m / self.num_active_users),
    #                          'ID: {}({}/{})'.format(self.client_id, self.m + 1, self.num_active_users),
    #                          'Learning rate: {}'.format(self.lr),
    #                          'Rate: {}'.format(self.model_rate),
    #                          'Epoch Finished Time: {}'.format(epoch_finished_time),
    #                          'Experiment Finished Time: {}'.format(exp_finished_time)]}
    #         self.logger.append(info, 'train', mean=False)
    #         self.logger.write('train', cfg['metric_name']['train']['Local'])

    # def test_model_for_user(self, m, ids):
    #     cfg = self.cfg
    #     metric = Metric()
    #     [dataset, data_split, model, label_split] = ids#ray.get(ids)
    #     model = model.to('cuda')
    #     data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
    #     results = []
    #     for _, data_input in enumerate(data_loader):
    #         data_input = collate(data_input)
    #         input_size = data_input['img'].size(0)
    #         data_input['label_split'] = torch.tensor(label_split[m])
    #         data_input = to_device(data_input, 'cuda')
    #         output = model(data_input)
    #         output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
    #         evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], data_input, output)
    #         results.append((evaluation, input_size))
    #     return results
