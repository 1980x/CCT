'''
Aum Sri Sai Ram
Implementation of ECCT: Ensemble Consensual Collaborative Training for Facial Expression Recognition with Noisy Annotations           
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-01-2021
Email: darshangera@sssihl.edu.in
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
from model.spherenet import AngleLoss
from algorithm.mmdloss import MMD_loss
from geomloss import SamplesLoss
from functools import partial
 
def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


#Dynamic balancing of Suprevision Loss and Consistency Loss
def ECCTloss(y_1, y_2, y_3, t, co_lambda_max = 0.9, beta = 0.65, epoch_num = 1, max_epochs = 40): 
    ''' 
        y_1, y_2, y_3 are predictions of 3 networks and t is target labels. 
    '''
    e = epoch_num
      
    e_r = 0.9 * max_epochs
     
    co_lambda = co_lambda_max * math.exp(-1.0 * beta * (1.0 - e / e_r ) ** 2) #Dynamic balancing factor using Gaussian like ramp-up function
     
    loss_ce_1 = F.cross_entropy(y_1, t) 
    
    loss_ce_2 =  F.cross_entropy(y_2, t)
    
    loss_ce_3 = F.cross_entropy(y_3, t) 
    
    loss_ce =   (1 - co_lambda) * 0.33 * (loss_ce_1 + loss_ce_2 + loss_ce_3)    #Supervision Loss weighted by (1 - dynamic balancing factor)
    
    consistencyLoss =  co_lambda * 0.33 * ( kl_loss_compute(y_1, y_2) +  kl_loss_compute(y_2, y_1)  + kl_loss_compute(y_1, y_3) +  kl_loss_compute(y_3, y_1) + kl_loss_compute(y_3, y_2) +  kl_loss_compute(y_2, y_3))  #Consistency Loss weighted by dynamic balancing factor
     
    loss  =  (consistencyLoss + loss_ce).cpu()
    
    return loss     
    

