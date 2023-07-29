import torch
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler

from types import MethodType
import re

from utils.utils import factory
from utils.metric import accumulate_acc, AverageMeter, Timer

from .agent import *
from collections import defaultdict
import numpy as np
from utils.utils import tensor_diag



class VDFD(Agent):
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super().__init__(agent_config)
        if self.multihead:
            n_feat = self.model.last[str(self.task_count+1)].in_features
        else:
            n_feat = self.model.last['All'].in_features
        self.hypermodel = factory('models', self.config['hypermodel_type'], self.config['hypermodel_name'])(in_dim=n_feat, hid_dim=n_feat // 16, out_dim=n_feat // 8, p=0)
        self.hyper_param_num = count_parameter(self.hypermodel)
        if self.config['gpu']:
            self.hypermodel = self.hypermodel.cuda()
        self.log('#param of hypermodel:{}'.format(self.hyper_param_num))



    def train_task(self, train_loader, val_loader=None):
        self.train_model(train_loader, val_loader)
        if len(self.regularization_terms) == 0:
            self.regularization_terms = {'imp':{'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list)}, 'GWstar': []}
        importance = self.calculate_importance(train_loader, str(self.task_count+1))
        num = 0
        GWstar = torch.zeros(1, self.config['cls_num'])
        if self.config['gpu']:
            GWstar = GWstar.cuda()
        for n, p in self.reg_params.items():
            left_eigen_vec, eigen_val, right_eigen_vec = self.svd(importance[n])
            num += np.prod(left_eigen_vec.shape) + np.prod(eigen_val.shape) + np.prod(right_eigen_vec.shape)
            self.regularization_terms['imp']['left_eigen_vec'][n].append(left_eigen_vec.unsqueeze(0))
            self.regularization_terms['imp']['eigen_val'][n].append(eigen_val.unsqueeze(0))
            self.regularization_terms['imp']['right_eigen_vec'][n].append(right_eigen_vec.unsqueeze(0))
            wstar = p.clone().detach().unsqueeze(-1).expand([1]+list(p.shape)+[self.config['cls_num']])
            GWstar += self.comp_GW(left_eigen_vec.unsqueeze(0), eigen_val.unsqueeze(0), right_eigen_vec.unsqueeze(0), wstar)
        self.regularization_terms['GWstar'].append(GWstar)
        self.log('storage: {}'.format(num))
        self.log('Singular value: q={}'.format(eigen_val.shape))
        self.task_count += 1

    def load_task(self, train_loader, val_loader=None):
        if len(self.regularization_terms) == 0:
            self.regularization_terms = {'imp':{'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list)}, 'GWstar': []}
        importance = self.calculate_importance(train_loader, str(self.task_count+1))
        num = 0
        GWstar = torch.zeros(1, self.config['cls_num'])
        if self.config['gpu']:
            GWstar = GWstar.cuda()
        for n, p in self.reg_params.items():
            left_eigen_vec, eigen_val, right_eigen_vec = self.svd(importance[n])
            num += np.prod(left_eigen_vec.shape) + np.prod(eigen_val.shape) + np.prod(right_eigen_vec.shape)
            self.regularization_terms['imp']['left_eigen_vec'][n].append(left_eigen_vec.unsqueeze(0))
            self.regularization_terms['imp']['eigen_val'][n].append(eigen_val.unsqueeze(0))
            self.regularization_terms['imp']['right_eigen_vec'][n].append(right_eigen_vec.unsqueeze(0))
            wstar = p.clone().detach().unsqueeze(-1).expand([1]+list(p.shape)+[self.config['cls_num']])
            GWstar += self.comp_GW(left_eigen_vec.unsqueeze(0), eigen_val.unsqueeze(0), right_eigen_vec.unsqueeze(0), wstar)
        self.regularization_terms['GWstar'].append(GWstar)
        self.log('storage: {}'.format(num))
        self.log('Singular value: q={}'.format(eigen_val.shape))
        self.task_count += 1

    def svd(self, imp):
        imp = torch.stack(imp)
        if imp.dim() == 3: #linear
            imp = imp.permute(1,2,0)
            left, eigen, right = [], [], []
            for o_c in range(imp.size(0)):
                l_v, e, r_v = torch.svd_lowrank(imp[o_c], q=1)
                left.append(l_v.unsqueeze(0))
                eigen.append(e.unsqueeze(0))
                right.append(r_v.t().unsqueeze(0))
            left_eigen_vec = torch.cat(left)
            eigen_val = torch.cat(eigen)
            right_eigen_vec = torch.cat(right)
            return left_eigen_vec, eigen_val, right_eigen_vec
        else:
            imp = imp.t()
            left_eigen_vec, eigen_val, right_eigen_vec = torch.svd_lowrank(imp, q=1)
            return left_eigen_vec, eigen_val, right_eigen_vec.t()

    def calculate_importance(self, dataloader, task_id):
        importance = defaultdict(list)
        for n, p in self.reg_params.items():
            for i in range(self.config['out_dim'][task_id]):#just work for multi-head
                if self.reg_params[n].dim() == 4:
                    importance[n].append(p.mean([1, 2, 3]).clone().detach().view(-1).fill_(0))
                elif self.reg_params[n].dim() == 2:
                    importance[n].append(p.clone().detach().fill_(0))
                else:
                    importance[n].append(p.clone().detach().view(-1).fill_(0))

        mode = self.model.training
        self.model.eval()
        for _, (inputs, targets, task) in enumerate(dataloader):
            if self.config['gpu']:
                inputs = inputs.cuda()

            outputs = self.model.forward(inputs)
            output = outputs[task_id].mean(dim=0)#just work for multi-head
            # svd after averaging on kernel
            for i in range(self.config['out_dim'][task_id]):
                self.model.zero_grad()
                output[i].backward(retain_graph=True if i < self.config['out_dim'][task_id] -1 else False)
                for n, p in self.reg_params.items():
                    if p.grad is not None:
                        if p.dim() == 4:#conv
                            node_imp = p.grad.mean([1,2,3]).view(-1)
                        elif p.dim() == 2:#linear
                            node_imp = p.grad
                        else:
                            node_imp = p.grad.view(-1)
                        importance[n][i] += node_imp / len(dataloader)
        self.model.train(mode=mode)
        return importance

    def comp_GW(self, left, eigen, right, w):
        rec = torch.matmul(torch.matmul(left, tensor_diag(eigen, eigen.device)), right)
        if w.dim() == 6:#task+conv-dim+class
            rec = rec.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(w.shape)
            gw = (rec * w).sum(dim=[1, 2, 3, 4])
        elif w.dim() == 4:#task+linear-dim+class
            gw = (rec * w).sum(dim=[1, 2])
        else:
            gw = (rec * w).sum(dim=1)
        return gw

    def reg_loss(self, step):
        self.reg_step += 1
        GW = torch.zeros(self.task_count, self.config['cls_num'])
        if self.config['gpu']:
            GW = GW.cuda()
        GWstar = torch.cat(self.regularization_terms['GWstar'])#t*c
        for n, p in self.reg_params.items():
            left_e_v = torch.cat(self.regularization_terms['imp']['left_eigen_vec'][n], dim=0)
            eigen = torch.cat(self.regularization_terms['imp']['eigen_val'][n], dim=0)
            right_v = torch.cat(self.regularization_terms['imp']['right_eigen_vec'][n], dim=0)
            w = p.unsqueeze(-1).expand([self.task_count] + list(p.shape)+[self.config['cls_num']])  # t*p*c
            GW += self.comp_GW(left_e_v, eigen, right_v, w)
        affinity = torch.matmul(self.prototype, self.prototype.permute(0, 2, 1))#t*c*c
        decom = self.hypermodel(affinity, self.prototype)
        nom_decom = f.normalize(decom, p=2, dim=-1)
        precision = torch.matmul(nom_decom, nom_decom.permute(0,2,1))
        identity = torch.eye(precision.size(-1)).expand(*precision.size()).to(precision.device)
        cross_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GWstar.unsqueeze(-1))#(t*1*c) matmul (t*c*c) matmul (t*c*1)
        square_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GW.unsqueeze(-1))
        square_star_term = torch.matmul(torch.matmul(GWstar.unsqueeze(1), precision), GWstar.unsqueeze(-1))
        task_reg_loss = (square_term - 2*cross_term + square_star_term).sum(dim=[1,2]) 
        
        task_reg_loss += (1/((precision+self.config['eps']*identity).det())).log()
        reg_loss = task_reg_loss.sum()# sum task
        self.summarywritter.add_scalar('reg_loss', reg_loss, self.reg_step)
        return reg_loss

    def com_prototype(self):
        self.log('with input prototype normalization')
        prototype = []
        for t in range(self.task_count):
            weight = self.model.last[str(t+1)].weight.detach()#c*f
            normalized = f.normalize(weight, p=2, dim=1)
            prototype.append((normalized).unsqueeze(0))#1*c*f
        self.prototype = torch.cat(prototype)#t*c*f



class VDFDSwin(VDFD):
    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.log('redefine regularization parameters')
        if self.with_head:
            self.reg_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self.log('Constraining head...')
        else:
            self.reg_params = {n: p for n, p in self.model.named_parameters() if not 'last' in n and not 'head.weight' in n and not 'head.bias' in n}
            self.log('len reg param {}'.format(len(self.reg_params)))
            self.log('Not constraining head...')
            if self.reset_model_optimizer is False:
                import warnings
                warnings.warn('the head for previous task may be '
                              'not constrained but updated if the '
                              'optimizer is momentum-based')
    
    def create_model(self):
        cfg = self.config
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        assert(cfg['model_type'] == 'swin')
        model = factory('models', cfg['model_type'], cfg['model_name'])(fused_window_process=cfg['fused_window_process'])
        model.last = model.head
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features  # input_dim
        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim, bias=False)
            # model.last[task].weight = model.fc.weight

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
        if self.multihead:
            head_params = list(p for n,p in self.model.named_children() if bool(re.match('last', n)))[0]
            cur_head_params = list(head_params[str(self.task_count+1)].parameters())
        else:
            cur_head_params = self.model.last['All'].parameters()
        fea_params = [p for n, p in self.model.named_parameters() if not 'last' in n and not 'head.weight' in n and not 'head.bias' in n]
        model_optimizer_arg = {'params': [{'params': cur_head_params, 'lr': self.config['head_lr'],'weight_decay': self.config['head_weight_decay']},
                                          {'params': fea_params, 'lr': self.config['model_lr'],'weight_decay': self.config['model_weight_decay']}]
                               }
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config is 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(torch.optim, self.config['model_optimizer'])(**model_optimizer_arg)
        n_iter_per_epoch = self.config['n_iter_per_epoch']
        num_steps = int(self.config['schedule'][-1] * n_iter_per_epoch)
        warmup_steps = int(self.config['warmup_epochs'] * n_iter_per_epoch)

        self.lr_scheduler = CosineLRScheduler(
            self.model_optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=self.config['min_lr'],
            warmup_lr_init=self.config['warmup_lr'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
        


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
                output = self.model.forward(inputs)
                loss = self.criterion(output, target, task, i)
                self.model_optimizer.zero_grad()
                if self.task_count > 0:
                    self.hypermodel_optimizer.zero_grad()
                loss.backward()
                
                if self.task_count > 0:
                    self.hypermodel_optimizer.step()
                    self.hypermodel_scheduler.step(epoch)
                    
                self.model_optimizer.step()
                self.lr_scheduler.step_update((epoch * len(train_loader) + i))

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
                self.validation(val_loader)
