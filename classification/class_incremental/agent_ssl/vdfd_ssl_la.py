from .vdfd import VDFD
import torch
import numpy as np
from collections import defaultdict

class VDFDLA(VDFD):
    def __init__(self, agent_config):
        super().__init__(agent_config)
    
    def load_task(self, train_loader, val_loader=None):
        self.regularization_terms[self.task_count] = {'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list), 'GWstar': None}
        importance = self.calculate_importance(train_loader, str(self.task_count+1))
        num = 0
        GWstar = torch.zeros(1, self.config['task_out_space'][str(self.task_count+1)])
        if self.config['gpu']:
            GWstar = GWstar.cuda()
        sigv = self.config['singular'] 
        for n, p in self.reg_params.items():
            left_eigen_vec, eigen_val, right_eigen_vec = self.svd(importance[n], sigv)
            num += np.prod(left_eigen_vec.shape) + np.prod(eigen_val.shape) + np.prod(right_eigen_vec.shape)
            self.regularization_terms[self.task_count]['left_eigen_vec'][n] = left_eigen_vec
            self.regularization_terms[self.task_count]['eigen_val'][n] = eigen_val
            self.regularization_terms[self.task_count]['right_eigen_vec'][n] = right_eigen_vec
            wstar = p.clone().detach().unsqueeze(-1).expand(list(p.shape)+[self.config['task_out_space'][str(self.task_count+1)]])
            GWstar += self.comp_GW(left_eigen_vec, eigen_val, right_eigen_vec, wstar).unsqueeze(0)
        self.regularization_terms[self.task_count]['GWstar'] = GWstar
        self.log('storage: {}'.format(num))
        self.log('Singular value: q={}'.format(eigen_val.shape))
        self.task_count += 1
        

    def cross_entropy(self, preds, targets, tasks):
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
                pred = preds['All'][:, self.valid_start:self.valid_out_dim]
            remap_targets = targets - self.valid_start
            loss = self.criterion_fn(pred / self.config['temp'], remap_targets)
        return loss