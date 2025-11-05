import logging
import numpy as np

class capacity_scheduler:
    '''
    class to schedule the capacity of the bottleneck layer of the VAE

    Parameters
    ----------
    patience : int
        How long to wait after last time validation loss improved.
    capacity_start : float
        Initial capacity of the bottleneck layer.
    capacity_increment : float
        How much to increment the capacity after patience epochs.
    capacity_increment_mode : str
        One of `abs`, `rel`. In `abs` mode, `capacity_increment` is added to the capacity, in `rel` mode, `capacity_increment` is multiplied with the capacity.
    threshold : float
        Minimum change in the monitored quantity to qualify as an improvement.
    threshold_mode : str
        One of `rel`, `abs`. In `rel` mode, `threshold` is (1 - 'threshold') * best_score, in `abs` mode, `threshold` is best_score - 'threshold'.
    trace_func : function
        trace print function.
    
    Returns
    -------
    capacity_scheduler : capacity_scheduler
        capacity_scheduler object
    '''

    def __init__(self, patience, capacity_start, capacity_max, capacity_increment, capacity_increment_mode, threshold, threshold_mode, trace_func = logging.info, enabled = True):
        self.patience = patience
        self.capacity = capacity_start
        self.capacity_max = capacity_max
        self.capacity_increment = capacity_increment
        assert capacity_increment_mode in ['abs', 'rel'], 'Invalid capacity increment mode selected.'
        self.capacity_increment_mode = capacity_increment_mode
        self.counter = 0
        self.best_score = np.inf
        self.corresponding_score = None
        self.early_stop = False
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        assert threshold_mode in ['abs', 'rel'], 'Invalid threshold mode selected.'
        self.trace_func = trace_func
        self.enabled = enabled
        self.reached_max_capacity = False
        self.trace_func('CapacityScheduler initialized with patience = {}, capacity = {}, capacity_increment = {}, capacity_increment_mode = {}, threshold = {}, threshold_mode = {}'.format(
            self.patience, self.capacity, self.capacity_increment, self.capacity_increment_mode, self.threshold, self.threshold_mode))
        if not self.enabled:
            self.trace_func('CapacityScheduler disabled.')

    def update(self, score):
        if self.enabled and not self.reached_max_capacity:
            _update_flag = False
            
            if self.threshold_mode == 'abs':
                if score < self.best_score - self.threshold:
                    _update_flag = True
            elif self.threshold_mode == 'rel':
                if score < self.best_score * (1 - self.threshold):
                    _update_flag = True
            
            if _update_flag:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
            
            if self.counter >= self.patience:
                if self.reached_max_capacity is False:
                    # update capacity
                    if self.capacity_increment_mode == 'abs':
                        new_capacity = self.capacity + self.capacity_increment
                    elif self.capacity_increment_mode == 'rel':
                        new_capacity = self.capacity * self.capacity_increment
                    if new_capacity > self.capacity_max:
                        new_capacity = self.capacity_max
                        self.reached_max_capacity = True
                        self.trace_func('\tCapacityScheduler reached maximum capacity of {}.'.format(self.capacity_max))
                    self.capacity = new_capacity

                    self.trace_func('\tCapacityScheduler updated capacity to {} after {} epochs.'.format(self.capacity, self.counter))
                    self.counter = 0

    def get_capacity(self):
        return self.capacity if self.enabled else None