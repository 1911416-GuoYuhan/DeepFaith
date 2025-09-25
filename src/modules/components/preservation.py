import torch

def target_prob(y_origin,y_perturb):
    # y_origin: B*C y_perturb:B*C
    # return: B
    y_pred = torch.argmax(y_origin,dim=1)
    batch_indices = torch.arange(y_origin.shape[0], device=y_origin.device)
    y_pred = y_perturb[batch_indices, y_pred]

    return y_pred

def prob_ratio(y_origin,y_perturb):
    # y_origin: B*C y_perturb:B*C
    # return: B
    y_pred = torch.argmax(y_origin,dim=1)
    batch_indices = torch.arange(y_origin.shape[0], device=y_origin.device)
    ratio = y_perturb[batch_indices, y_pred] / y_origin[batch_indices, y_pred]
    return ratio


preservation_effect = {
    'target': target_prob,
    'ratio': prob_ratio
}