import torch
import torch.nn as nn
import logging

from .train_to_negative_eigenvalues import initialize_to_negative_eigenvalues

def initialize_weights_biases(net: nn.Module, method: str = None, **kwargs):
    if method == 'identity':
        # initialize weights
        # !!! I think this is not a good idea, because of the bias !!!
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0) # i think this is not good, because no gradients can be backpropagated through this
    elif method == 'xavier':
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0001)
    elif method == 'move_eigvals_matrix':
        initialized_layer=False
        for m in net.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    raise ValueError('move eigaval method only intended to use with no bias in layers')
                if initialized_layer == True:
                    raise ValueError('move eigval method only intendet to use for a module with a single layer')
                w = m.weight
                eigvals, eigvecs = torch.linalg.eig(w)
                max_eigval = torch.max(eigvals.real)
                eigvals = eigvals - max_eigval + 0
                logging.info('for layer {} moved eigvals to {}'.format(m, eigvals))
                w = eigvecs @ torch.diag(eigvals) @ torch.inverse(eigvecs)
                w = w.real
                m.weight = nn.Parameter(w)
    elif method == 'move_eigvals_net':
        initialize_to_negative_eigenvalues(net, **kwargs)
    elif method == None:
        pass
    else:
        raise ValueError('Unknown initialization method: {}'.format(method))