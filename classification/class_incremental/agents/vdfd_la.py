from .vdfd import VDFD


class VDFDLA(VDFD):
    def __init__(self, agent_config):
        super().__init__(agent_config)
        

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
