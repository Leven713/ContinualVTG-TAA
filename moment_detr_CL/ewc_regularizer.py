import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.autograd as autograd
import utils
from torch.nn import functional as F
from tqdm import tqdm
from moment_detr_CL.start_end_dataset_all import prepare_batch_inputs

USE_CUDA = torch.cuda.is_available()

def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var


class EWC_Regularizer(nn.Module):
    def __init__(self, online: bool=True):
        super().__init__()
        self.online = online
        self.tasks_encountered = []

        self.fisher = {}
        self.optpar = {}

    def forward(self, named_params):
        net_loss = Variable(torch.Tensor([0]))
        for task_id in self.tasks_encountered:
            for name, param in named_params:
                fisher = Variable(self.fisher[task_id][name])
                optpar = Variable(self.optpar[task_id][name])
                net_loss += (fisher * (optpar - param).pow(2)).sum() 

        return net_loss.squeeze(0)

    def regularize(self, named_params):
        """Calculate the EWC regularization component in the overall loss.
        For all the tasks encountered in past, L2-norm loss is calculated
        between current model parameters and optimal parameters of previous
        tasks, weighted by terms from fisher matrix.

        Arguments
        =========
        named_params : generator
            Named parameters of model to be regularized.
        """
        return self.forward(named_params)
    
    # Update the Fisher Information
    def update_fisher_optpar(self, model, criterion, current_itr, train_loader, sample_size, opt, consolidate=True):
        if consolidate:
            if self.online:
                current_itr = 1
                self.tasks_encountered = [1]
            else:
                self.tasks_encountered.append(current_itr)
        
        if sample_size != -1:
            total_iter = min(len(train_loader), sample_size // opt.bsz)
        else:
            total_iter = len(train_loader)
        losses = []
        for _, batch in tqdm(enumerate(train_loader),
                                    desc="Fisher Matrix Estimation",
                                    total=total_iter):
            
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
            outputs = model(**model_inputs)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.append(loss)

            if len(losses) >= total_iter:
                break

        # estimate the fisher information of the parameters.
        # print(len(losses))
        sample_losses = losses
        sample_grads = zip(*[autograd.grad(l, model.parameters(), retain_graph=(i < len(sample_losses)), allow_unused=True) 
                            for i, l in enumerate(sample_losses)])
        
        sample_grads_list = []
        for gs, p in zip(sample_grads, model.parameters()):
            if gs[0] is None: 
                gs = torch.zeros_like(p).unsqueeze(0)
            else:
                gs = torch.stack(gs, dim=0)
                # print(gs.shape)
            sample_grads_list.append(gs)
        # sample_grads = [torch.cat(gs, dim=0) for gs in sample_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in sample_grads_list]
        self.fisher[current_itr] = {}
        self.optpar[current_itr] = {}
        
        for (name, param), fisher in zip(model.named_parameters(), fisher_diagonals):
            # print(name, param.shape, fisher.shape)
            self.optpar[current_itr][name] = param.data.clone()
            self.fisher[current_itr][name] = fisher.detach()