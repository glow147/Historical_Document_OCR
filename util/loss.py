import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
https://github.com/clcarwin/focal_loss_pytorch
'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        assert self.reduction in ['mean','sum','none'], 'Select [\'mean\',\'sum\'\'none\']'

    def forward(self, input, target):
        '''
            input  : N * n_classes(probability) * n_boxes
            target : N * n_boxes
        '''
        if input.dim()>2:
            N = input.size(0)
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N, n_boxes, n_classes(probability)
            input = input.contiguous().view(-1,input.size(2))   # N * n_boxes, n_classes
        target = target.view(-1,1)
        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            select = (target != 0).type(target.dtype)
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.view(N, -1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else: 
            return loss.sum()

if __name__ == '__main__':
    a = FocalLoss(gamma=0, alpha=None, reduction='none')
    p = nn.CrossEntropyLoss(reduction='none')
    b = torch.randn((1,3,10))
    c = torch.empty((1,10),dtype=torch.int64).random_(2)
    a(b.float(),c)
    print(a(b.float(),c))
    print(p(b.float(),c))
    #print(torch.mean(a(b.float(),c)))
    #print(torch.mean(p(b.float(),c)))
