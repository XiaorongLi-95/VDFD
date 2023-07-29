import torch
import torch.nn as nn
def preprocess_adj(A):
    device = A.device
    shape = A.shape
    I = torch.eye(A.shape[1]).unsqueeze(0).to(device)#classes 1*c*c
    A_hat = A + I
    D_hat_diag = torch.sum(A_hat, dim=1)#1*c
    D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[torch.isnan(D_hat_diag_inv_sqrt)] = 0
    D_hat_diag_inv_sqrt = D_hat_diag_inv_sqrt.unsqueeze(-1).expand(shape)
    D_hat_inv_sqrt = D_hat_diag_inv_sqrt * torch.eye(A.shape[1]).to(device)
    return torch.matmul(torch.matmul(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
    def forward(self, F):
        output = self.linear(F)
        if not self.act:
            return output
        return self.act(output)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, p):
        super().__init__()
        self.gcn_layer1 = GCNLayer(in_dim, hid_dim)
        self.gcn_layer2 = GCNLayer(hid_dim, out_dim, act=False)
        self.dropout = nn.Dropout(p)

    def forward(self, A, x):
        A = preprocess_adj(A)
        x = self.dropout(x)
        F = torch.matmul(A, x)
        F = self.gcn_layer1(F)
        F = self.dropout(F)
        F = torch.matmul(A, F)
        out = self.gcn_layer2(F)
        return out
    
    def reset_model(self):
        for module_ in self.modules():
            if hasattr(module_, 'weight') and module_.weight is not None:
                module_.reset_parameters() 