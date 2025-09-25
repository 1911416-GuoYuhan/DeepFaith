import torch

def pearson_mse(x,y,eps=1e-5):
    # x,y : B*N 
    # reuturn: B
    # print('x',x.max(),x.min())
    # print('y',y.max(),y.min())
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    # mean_loss =  mse(x_centered,y_centered)
    # if mean_loss.mean() <= eps:
    #     return 1 - mean_loss.abs()
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=1) + eps) * torch.sqrt((y_centered ** 2).sum(dim=1) + eps)
    corr = numerator / denominator
    # corr = torch.clamp(corr, -1 + 1e-4, 1 - 1e-4)
    return corr

def pearson(x,y,eps=1e-5):
    # x,y : B*N 
    # reuturn: B
    # print('x',x.max(),x.min())
    # print('y',y.max(),y.min())
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=1) + eps) * torch.sqrt((y_centered ** 2).sum(dim=1) + eps)
    corr = numerator / denominator
    # corr = torch.clamp(corr, -1 + 1e-4, 1 - 1e-4)
    return corr

def spearman(x,y):
    # x,y : B*N 
    # reuturn: B
    assert x.shape == y.shape
    B, N = x.shape

    x_rank = torch.argsort(x, dim=1).float()
    y_rank = torch.argsort(y, dim=1).float()
    # print(x_rank,y_rank)

    x_centered = x_rank - x_rank.mean(dim=1, keepdim=True)
    y_centered = y_rank - y_rank.mean(dim=1, keepdim=True)

    numerator = (x_centered * y_centered).sum(dim=1)  
    x_std = torch.sqrt((x_centered ** 2).sum(dim=1))  
    y_std = torch.sqrt((y_centered ** 2).sum(dim=1)) 
    denominator = x_std * y_std 

    corr = torch.where(
        denominator == 0,
        torch.zeros_like(numerator),
        numerator / denominator
    )

    return corr

def mse(x,y):
    # x,y : B*N 
    # reuturn: B
    dist = torch.mean(0.5*(x-y)**2,dim=1)
    return -dist

def mse_pearson(x,y):
    return  mse(x,y)+pearson(x,y)

correlation_functions ={
    'pearson': pearson,
    'spearman': spearman,
    'mse': mse,
    'mse_pearson': mse_pearson,
    'pearson_mse': pearson_mse
}

if __name__ == '__main__':

    x = torch.tensor([[1., 2., 3.], [1., 3., 2.]])  
    y = torch.tensor([[2., 3., 1.], [1., 2., 3.]])
    print(mse(x, y)) 