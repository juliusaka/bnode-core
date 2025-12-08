import torch
import logging
import time

import torch.nn as nn
import torch.optim as optim

def calculate_jacobian(model, x, in_n_args = None):
    x.requires_grad_(True)
    if in_n_args is not None:
        _x = [x[:, sum(in_n_args[:i]):sum(in_n_args[:i+1])] for i in range(len(in_n_args))]
        y = model(*_x)
    else:
        y = model(x)
    y_dim = y.size()[1]
    jacobian = torch.zeros(y_dim, y_dim)
    for i in range(y_dim):
        gradient = torch.autograd.grad(y[0, i], x, create_graph=True)[0]
        jacobian[i] = gradient[0,0:y_dim]
    return jacobian

def train_to_negative_eigenvalues(model, learning_rate, num_iterations, patience, in_n_args: list =None, out_features=None):
    # this assumes that the first entries in the input_dimension are the states and correspond to the output_dimension (state derivative)
    input_dim = model[0].in_features if in_n_args is None else sum(in_n_args)
    output_dim = model[-1].out_features if out_features is None else out_features
    if output_dim > input_dim:
        raise ValueError('output dimension must be smaller than input dimension')
    logging.info('Starting gradient descent to reduce eigenvalues of model {}'.format(model._get_name()))
    device = next(model.parameters()).device
    logging.info('Training on device: {}'.format(device))
    patience_counter = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(num_iterations):
        x = torch.randn(256, input_dim, device=next(model.parameters()).device)
        optimizer.zero_grad()
        jacobian = calculate_jacobian(model, x, in_n_args)
        eigenvalues = torch.linalg.eigvals(jacobian)
        if i==0:
            eigenvalues_0 = eigenvalues.clone()
        loss = torch.sum(torch.relu(eigenvalues.real))
        if loss == 0:
            patience_counter += 1
            logging.info('Patience counter: {}/{}'.format(patience_counter, patience))
            if patience_counter == patience:
                logging.info('Converged after {} iterations'.format(i))
                break
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            logging.info('Reducing eigenvalus of model {}: Loss: {:4f} after {} iterations'.format(model._get_name(), loss, i))
    return model, eigenvalues_0, eigenvalues

def initialize_to_negative_eigenvalues(model: torch.nn.Module, in_n_args=None, out_features=None):
    learning_rate = 1e-3
    num_iterations = 1000
    patience = 10
    device_0 = next(model.parameters()).device
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Training is faster on device: {}'.format(next(model.parameters()).device))
    model, eigenvalues_0, eigenvalues = train_to_negative_eigenvalues(model, learning_rate, num_iterations, patience, in_n_args, out_features)
    logging.info('Eigenvalues before training: {}'.format(eigenvalues_0))
    logging.info('{}/{} eigenvalues are negative'.format(torch.sum(eigenvalues_0.real < 0), len(eigenvalues_0)))
    logging.info('Eigenvalues after training: {}'.format(eigenvalues))
    logging.info('{}/{} eigenvalues are negative'.format(torch.sum(eigenvalues.real < 0), len(eigenvalues)))
    if device_0 != next(model.parameters()).device:
        model.to(device_0)
        logging.info('Moving model back to {}'.format(device_0))
    return model

if __name__ == '__main__':
    # test the model initialization
    logging.basicConfig(level=logging.INFO)
    dim = 256
    model = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, dim))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    model = initialize_to_negative_eigenvalues(model)
    t1 = time.time()
    logging.info('Time: {}'.format(t1-t0))