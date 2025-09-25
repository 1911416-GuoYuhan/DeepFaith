import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.components.correlation import correlation_functions
from src.modules.components.perturbation_delta import perturbation_effects
from src.modules.components.perturb import perturb_methods

class SuperviseLoss(nn.Module):
    def __init__(self, corr='pearson'):
        super(SuperviseLoss, self).__init__()
        self.corr = correlation_functions[corr]

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        # B
        correlation = self.corr(inputs.reshape(batch_size,-1),targets.reshape(batch_size,-1))
        return (1 - torch.mean(correlation))/2

class BernoulliCustomSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prob):
        a = torch.bernoulli(prob)
        ctx.save_for_backward(a, prob) 
        return a
    
    @staticmethod
    def backward(ctx, grad_output):
        a, prob = ctx.saved_tensors
        grad = grad_output * a * prob
        return torch.clamp(grad, -1.0, 1.0) 

class FaithfulLoss(nn.Module):
    def __init__(self, nr_runs=100, corr='pearson', perturb_effect = 'minus', perturbation = 'replace'):
        super(FaithfulLoss, self).__init__()
        self.corr_func = correlation_functions[corr]
        self.delta_func = perturbation_effects[perturb_effect]
        self.perturb = perturb_methods[perturbation]
        self.nr_runs = nr_runs

    def forward(self, x_batch, model, a_batch):

        # device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            y_origin = torch.softmax(model(x_batch),dim=1)

        batch_size = x_batch.shape[0]
        # num_features = a_batch[0].numel()
        a_batch_flat = a_batch.reshape(batch_size, -1)
        x_batch_flat = x_batch.reshape(batch_size, -1)

        pred_deltas = []
        att_sums = []

        # print(a_batch_flat.max(),a_batch_flat.min())
        prob = torch.sigmoid(a_batch_flat)
        for _ in range(self.nr_runs):
            bernoulli_mask = BernoulliCustomSTE.apply(prob)

            x_perturbed_flat = x_batch_flat.clone()
            #x_perturbed = self.perturb(src=torch.zeros_like(x_perturbed_flat),trg=x_perturbed_flat,indices=indices).reshape(x_batch.shape)
            x_perturbed = self.perturb(src=bernoulli_mask,trg=x_perturbed_flat,indices=None).reshape(x_batch.shape)

            logits_perturb = model(x_perturbed)
            preds_perturb = torch.softmax(logits_perturb, dim=-1)
            
            pred_deltas.append(self.delta_func(y_origin,preds_perturb))

            att_sum = torch.sum(a_batch_flat*bernoulli_mask,dim=1)
            att_sums.append(att_sum)
            del x_perturbed_flat

        pred_deltas = torch.stack(pred_deltas, dim=1) # [batch, nr_runs]
        att_sums = torch.stack(att_sums, dim=1)        # [batch, nr_runs]


        corr = self.corr_func(att_sums,pred_deltas)
        # E(1-x)=E(1)-E(x)=1-E(x)
        return (1-torch.mean(corr))/2
