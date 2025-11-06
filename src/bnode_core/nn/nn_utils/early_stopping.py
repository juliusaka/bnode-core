import numpy as np
import logging
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, threshold=0, threshold_mode ='abs', path='checkpoint.pt', optimizer_path='optimizer.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            threshold (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode, `threshold` is
                            (1 - 'threshold') * best_score, in `abs` mode, `threshold` is
                            best_score - 'threshold'. Default: 'abs'
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.corresponding_score = None
        self.early_stop = False
        self.score_last_save = np.inf
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.path = path
        self.optimizer_path = optimizer_path
        self.trace_func = trace_func
        self.trace_func('EarlyStopping initialized with patience = {}, threshold = {}, threshold_mode = {}'.format(
            self.patience, self.threshold, self.threshold_mode))
        
    def reset(self):
        self.reset_counter()
        self.best_score = None
        self.corresponding_score = None
        self.early_stop = False
        self.score_last_save = np.inf

    def reset_counter(self):
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model, epoch = None, optimizer = None, corresponding_loss = None):
        '''
        loss: validation loss
        model: model to be saved
        epoch: epoch number
        corresponding_loss: training loss of the same epoch
        '''
        # if loss is not a number
        if np.isnan(loss):
            loss = np.inf
            logging.warning('EarlyStopping: loss is NaN. Setting to Inf for early stopping update.')
        score = loss
        
        # initial case
        if self.best_score is None:
            self.best_score = score
            self.corresponding_score = corresponding_loss
            self.save_checkpoint(loss, model, optimizer, epoch)
        
        _update_flag = False
        
        if self.threshold_mode == 'abs':
            if score < self.best_score - self.threshold:
                _update_flag = True
        elif self.threshold_mode == 'rel':
            if score < self.best_score * (1 - self.threshold):
                _update_flag = True
        else:
            raise ValueError('Invalid threshold mode selected.')
        
        if _update_flag:
            self.best_score = score
            self.counter = 0
            self.corresponding_score = corresponding_loss
            self.save_checkpoint(loss, model, optimizer, epoch)
        else:
            self.counter += 1
        if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, loss, model, optimizer, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func('----------------------> Epoch {} Validation loss decreased ({:.6f} --> {:.6f}).  Saving model to {}'.format(epoch, self.score_last_save, loss, self.path))
        model.save(self.path)
        self.score_last_save = loss
        if optimizer is not None:
                torch.save(optimizer.state_dict(), self.optimizer_path)