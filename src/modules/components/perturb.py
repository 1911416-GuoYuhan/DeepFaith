import torch

def perturb_by_replacement(src, trg, indices):

    trg_modified = trg.clone()
    
    src_values = torch.gather(src, 1, indices)
    
    trg_modified.scatter_(1, indices, src_values)

    
    return trg_modified

def perturb_by_difference(src, trg, indices):
    trg_modified = trg.clone()
    
    src_values = torch.gather(src, 1, indices)
    
    trg_modified.scatter_(1, indices, trg_modified.gather(1, indices) - src_values)

    return trg_modified

def perturb_by_mask(src, trg, indices=None):

    noise = torch.randn_like(trg)
    res = src*trg+(1-src)*noise
    # res = src*trg
    return res
    
    

perturb_methods = {
    'replace': perturb_by_replacement,
    'diff': perturb_by_difference,
    'mask': perturb_by_mask
}