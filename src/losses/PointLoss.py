import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, x, y):
        '''
        Binary focal loss for point set segmentation
        Note that the positive class are very few
        '''
        # p = torch.sigmoid(x)              # from p.3 last section
        p = x
        p = p.view(y.size(0), -1)
        
        pt = (p*y + (1-p)*(1-y)).clamp(1e-8,1-1e-8)         # pt = p if y > 0 else 1-p
        w = self.alpha*y + (1-self.alpha)*(1-y)  # w = alpha if y > 0 else 1-alpha
        w = w * (1-pt)**self.gamma
        loss = -w*pt.log()
        
        return loss.sum()

if __name__=='__main__':
    x = Variable(torch.randn(3), requires_grad=True)
    #x = Variable(torch.FloatTensor(3).random_(2), requires_grad=False)
    y = Variable(torch.FloatTensor(3).random_(2), requires_grad=False)
    criteria = FocalLoss(alpha=0.25, gamma=2.0)
    print(x, y)
    print(criteria(x,y))
