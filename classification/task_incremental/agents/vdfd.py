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


    def init_model_optimizer(self):
        head_params = list(p for n,p in self.model.named_children() if bool(re.match('last', n)))[0]
        cur_head_params = list(head_params[str(self.task_count+1)].parameters())
        fea_params = [p for n, p in self.model.named_parameters() if not bool(re.match('last', n))]
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
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.5)


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

    def svd(self, imp):
        left_eigen_vec, eigen_val, right_eigen_vec = torch.svd_lowrank(torch.stack(imp).t(), q=1)
        return left_eigen_vec, eigen_val, right_eigen_vec.t()

    def calculate_importance(self, dataloader, task_id):
        importance = defaultdict(list)
        for n, p in self.reg_params.items():
            for i in range(self.config['out_dim'][task_id]):#just work for multi-head
                if self.reg_params[n].dim() == 4:
                    importance[n].append(p.mean([1, 2, 3]).clone().detach().view(-1).fill_(0))
                elif self.reg_params[n].dim() == 2:
                    importance[n].append(p.mean(dim=1).clone().detach().view(-1).fill_(0))
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
                            node_imp = p.grad.mean(dim=1).view(-1)
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
            rec = rec.unsqueeze(2).expand(w.shape)
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
        prototype = []
        for t in range(self.task_count):
            weight = self.model.last[str(t+1)].weight.detach()#c*f
            normalized = f.normalize(weight, p=2, dim=1)
            prototype.append((normalized).unsqueeze(0))#1*c*f
        self.prototype = torch.cat(prototype)#t*c*f

