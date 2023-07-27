import re
import torch
import torch.nn as nn
from types import MethodType
from tensorboardX import SummaryWriter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils.metric import accumulate_acc_ssl, accumulate_seen_acc, accumulate_acc, AverageMeter, Timer 
from utils.utils import count_parameter, factory
import time
import torch.nn.init as init
import torch.nn as nn
import math
import torch.nn.functional as f
import numpy as np



class Agent(nn.Module):
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super().__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
                *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.log(agent_config)
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        self.task_count = 0
        self.model = self.create_model()
        
        self.criterion_fn = nn.CrossEntropyLoss()
        
        self.init_model_optimizer()

        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active # Set a interger here for the incremental class scenario

        self.regularization_terms = {}
        self.clf_param_num = count_parameter(self.model)
        

        self.reg_step = 0
        self.summarywritter = SummaryWriter('./log/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if self.config['gpu']:
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
            
        self.log('#param of model:{}'.format(self.clf_param_num))

        self.reset_model_optimizer = agent_config['reset_model_opt']
        self.with_head = agent_config['with_head']
        if self.with_head:
            self.reg_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self.log('Constraining head...')
        else:
            self.reg_params = {n: p for n, p in self.model.named_parameters() if not bool(re.match('last', n))}
            self.log('Not constraining head...')
            if self.reset_model_optimizer is False:
                import warnings
                warnings.warn('the head for previous task may be '
                              'not constrained but updated if the '
                              'optimizer is momentum-based')

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = factory('models', cfg['model_type'], cfg['model_name'])()
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features  # input_dim

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, 4*out_dim, bias=False)# SSL

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model


    def init_model_optimizer(self):
        model_optimizer_arg = {'params': self.model.parameters(),
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config is 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(torch.optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.5)
                                            
    def init_hypermodel_optimizer(self):
        hypermodel_optimizer_arg = {'params': self.hypermodel.parameters(),
                               'lr': self.config['hypermodel_lr'],
                               'weight_decay': self.config['hypermodel_weight_decay']}
        if self.config['hypermodel_optimizer'] in ['SGD', 'RMSprop']:
            hypermodel_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['hypermodel_optimizer'] in ['Rprop']:
            hypermodel_optimizer_arg.pop('weight_decay')
        elif self.config['hypermodel_optimizer'] in ['amsgrad', 'Adam']:
            if self.config is 'amsgrad':
                hypermodel_optimizer_arg['amsgrad'] = True
            self.config['hypermodel_optimizer'] = 'Adam'

        self.hypermodel_optimizer = getattr(torch.optim, self.config['hypermodel_optimizer'])(**hypermodel_optimizer_arg)
        self.hypermodel_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.hypermodel_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.5)                

    def train_task(self, train_loader, val_loader=None):
        raise NotImplementedError

    def train_model(self, train_loader, val_loader=None):
        if self.reset_model_optimizer:  # Reset model optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_model_optimizer()
            if self.task_count > 0:
                self.init_hypermodel_optimizer()
                self.com_prototype()
                self.hypermodel.reset_model()

        count_cls_step = 0

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()

            for param_group in self.model_optimizer.param_groups:
                self.log('LR:', param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (inputs, target, task) in enumerate(train_loader):
                
                count_cls_step += 1
                data_time.update(data_timer.toc())  # measure data loading time

                if self.config['gpu']:
                    inputs = inputs.cuda()
                    target = target.cuda()
                # SSL
                inputs = torch.stack([torch.rot90(inputs, k, (2,3)) for k in range(4)], 1)
                inputs = inputs.view([-1] + list(inputs.size()[-3:]))
                target = torch.stack([target*4 + k for k in range(4)], 1).view(-1)
                task = np.stack([task]*4,1).reshape(-1).tolist()
                task = tuple(task)
                _, output = self.model(inputs)
                loss = self.criterion(output, target, task)
                self.model_optimizer.zero_grad()
                if self.task_count > 0:
                    self.hypermodel_optimizer.zero_grad()
                loss.backward()
                
                if self.task_count > 0:
                    self.hypermodel_optimizer.step()
                    self.hypermodel_scheduler.step(epoch)
                    
                self.model_optimizer.step()
                self.model_scheduler.step(epoch)

                self.summarywritter.add_scalar(
                    'training_classifier_loss/task_%d' % (len(self.regularization_terms) + 1), loss, count_cls_step)
                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, inputs.size(0))
                self.summarywritter.add_scalar(
                    'training_classifier_accuracy/task_%d' % (len(self.regularization_terms) + 1),
                    acc.avg, count_cls_step)

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                        train_loader):
                    self.log('[{0}/{1}]\t'
                             '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                             '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                             '{loss.val:.3f} ({loss.avg:.3f})\t'
                             '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
            if val_loader is not None:
                with torch.no_grad():
                    self.validation(val_loader)

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def validation(self,dataloader, val_name=None):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        val_acc = AverageMeter()
        batch_timer.tic()

        self.model.eval()

        for i, (inputs, target, task) in enumerate(dataloader):

            if self.config['gpu']:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()

            _, output = self.model(inputs)
            for t in output.keys():
                output[t] = output[t].detach()
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            if self.multihead:
                val_acc = accumulate_acc_ssl(output, target, task, val_acc)
            else:
                val_acc = accumulate_seen_acc(output['All'][:, :self.valid_out_dim][:, ::4], target, val_acc)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                 .format(acc=val_acc, time=batch_timer.toc()))
        return val_acc.avg, val_acc.count

    def criterion(self, preds, targets, tasks, regularization=True):
        loss = self.cross_entropy(preds, targets, tasks)
        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = self.reg_loss()
            loss += self.config['reg_coef'] * reg_loss
        return loss

    def cross_entropy(self, preds, targets, tasks):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim,
                          int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss




    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_start = self.valid_out_dim = 0  # Initialize it with zero
        self.valid_start = self.valid_out_dim 
        self.valid_out_dim += dim * 4 # SSL
        self.log('Incremental class: New Valid output dimension:', self.valid_start, self.valid_out_dim)
        return self.valid_out_dim



