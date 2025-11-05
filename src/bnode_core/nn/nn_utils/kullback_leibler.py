import torch

def kullback_leibler(mu, logvar, per_dimension = False, reduce = True, time_series_aggregation_mode = 'mean'):
    '''
    implements the kullback leibler divergence
    based on mu and logvar of a normal distribution,
    comparing it to a distribution with mu=0 and sigma=1

    Args:
        mu (torch.tensor): mean of distribution
        logvar (torch.tensor): log variance of distribution
        per_dimension (bool, optional): if True, returns KL divergence for each dimension. Defaults to False.
        reduce (bool, optional): if True, returns mean KL divergence over all samples. Defaults to True.
    '''
    is_timeseries = len(mu.shape) == 3
    kl = -0.5 *(1 + logvar - mu.pow(2) - logvar.exp())
    
    if is_timeseries:
        if time_series_aggregation_mode == 'mean':
            kl = torch.mean(kl, dim=2)
        elif time_series_aggregation_mode == 'max':
            kl = torch.max(kl, dim=2)
        elif time_series_aggregation_mode == 'sum':
            kl = torch.sum(kl, dim=2)
        elif time_series_aggregation_mode == None:
            pass
    
    if per_dimension is False:
        kl = torch.sum(kl, dim=1)  
    
    if reduce:
        kl = torch.mean(kl, dim=0)
    
    return kl

def count_populated_dimensions(mu, logvar, threshold = 0.05, kl_timeseries_aggregation_mode = 'mean', return_idx = False):
    '''
    counts the number of dimensions of the KL divergence, where the KL divergence is larger than a threshold
    '''
    with torch.no_grad():
        kl = kullback_leibler(mu, logvar, per_dimension = True, reduce = True, time_series_aggregation_mode = kl_timeseries_aggregation_mode)
        # print histogram of kl divergence to terminal
        # print('kl divergence histogram: 0.0 - 2.0')
        # print(torch.histc(kl, bins=21, min=0, max=2.0))
        idx = kl > threshold
        n_dim_populated = torch.sum(idx)
    if return_idx:
        return n_dim_populated.detach(), idx.detach()
    else:
        return n_dim_populated.detach(), None