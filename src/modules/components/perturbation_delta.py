import torch

def minus(y_origin,y_perturb):
    # y_origin,y_perturb: B*C
    # return: B
    y_batch = torch.argmax(y_origin,dim=1)
    batch_indices = torch.arange(y_origin.shape[0], device=y_origin.device)
    y_pred = y_origin[batch_indices,y_batch]
    y_ptb = y_perturb[batch_indices,y_batch]

    return y_pred-y_ptb

def variance(y_origin,y_perturb,eps=1e-5):
    # y_origin,y_perturb: B*C
    # return: B
    y_batch = torch.argmax(y_origin,dim=1)
    batch_indices = torch.arange(y_origin.shape[0], device=y_origin.device)
    y_pred_perturb = y_perturb[batch_indices, y_batch] # B

    # inv_pred
    y_pred = y_origin[batch_indices, y_batch] # B
    inv_pred = torch.ones_like(y_pred) # B
    mask = torch.abs(y_pred) >= eps
    inv_pred[mask] = 1.0 / torch.abs(y_pred[mask])
    inv_pred = inv_pred ** 2

    var = ((y_pred_perturb - y_pred) ** 2).mean(dim=0,keepdim=True) * inv_pred # B

    return var

perturbation_effects = {
    'minus': minus,
    'variance': variance
}