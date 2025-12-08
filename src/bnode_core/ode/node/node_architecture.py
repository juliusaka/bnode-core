"""Neural ODE (NODE) Architecture Module.

This module implements a Neural Ordinary Differential Equation (NODE) architecture
for modeling dynamical systems. NODE directly learns the differential equations governing
a system's evolution by parameterizing the derivative function with neural networks. Specifically, this module
learns a State-Space representation of the system dynamics:

Attention
---------
This documentation is AI written and may contain inaccuracies. Please verify the details before use.

Overview
--------
Neural ODEs represent a continuous-depth neural network where the hidden state evolves
according to a learned ODE:

    dx/dt = f_θ(x, u, p, t)

Where:
    - x: System states
    - u: Control inputs (optional)
    - p: System parameters (optional)
    - t: Time
    - f_θ: Neural network parameterized by θ

Outputs are generated an optional output network:

    y = g_φ(x, u, p, t)

Where:
    - y: System outputs/measurements
    - g_φ: Neural network parameterized by φ

Architecture Components
-----------------------
1. **NeuralODEFunc**: Learns the state derivative function f_θ
    - Input: Current states, controls (optional), parameters (optional)
    - Output: State derivatives dx/dt
    - Uses normalization for numerical stability

2. **OutputNetwork**: Maps states to observable outputs (optional)
    - Input: States, controls (optional), parameters (optional)
    - Output: System outputs/measurements
    - Decouples internal dynamics from observations

3. **NeuralODE**: Main model combining ODE solver and output network
    - Integrates state derivatives using torchdiffeq solvers
    - Supports both training and inference modes
    - Includes pre-training on derivatives and main training with ODE solver

Key Features
------------
- **Continuous-time modeling**: No discretization artifacts
- **Variable time step**: Can predict at arbitrary time points
- **Flexible**: Supports controls, parameters, and outputs
- **Normalized**: Built-in normalization for numerical stability

Training Modes
--------------
1. **Pre-training Mode**:
    - Trains on state derivatives directly (if available in dataset, could construct these
    numerically, e.g. by finite differences)
    - Fast initial parameter estimation
    - No ODE solver required
    - Loss: ||dx/dt - f_θ(x, u, p)||²

2. **Main Training Mode**:
    - Integrates ODE from initial conditions
    - Uses torchdiffeq solvers (Euler, RK4, dopri5, etc.)
    - More accurate but slower
    - Loss: ||x(t) - ∫f_θ(x, u, p)dt||²

Supported Solvers
-----------------
See torchdiffeq documentation.


"""
from __future__ import annotations

import torch
import torch.nn as nn
import logging
import warnings
import h5py
from pathlib import Path
import time as pyTime
from torchdiffeq import odeint, odeint_adjoint
import torchdiffeq as torchdiffeq


from bnode_core.ode.ode_utils.initialize_model import initialize_weights_biases
from bnode_core.nn.nn_utils.normalization import NormalizationLayer1D
from bnode_core.ode.ode_utils.mixed_norm_for_torchdiffeq import _mixed_norm_tensor
from typing import Tuple, Optional


class NeuralODEFunc(nn.Module):
    """Neural network function representing ODE right-hand side f_θ(x, u, p).
    
    This module learns the state derivative function for a dynamical system:
        
        dx/dt = f_θ(x, u, p)
    
    The network includes normalization layers for numerical stability and supports
    optional control inputs and system parameters.
    
    Args:
        states_dim (int): Dimension of state vector x.
        controls_dim (int, optional): Dimension of control input vector u. Default: 0.
        parameters_dim (int, optional): Dimension of parameter vector p. Default: 0.
        hidden_dim (int, optional): Hidden layer dimension. Default: 20.
        n_layers (int, optional): Number of layers (minimum 2). Default: 3.
        activation (nn.Module, optional): Activation function class. Default: nn.ELU.
        intialization (str, optional): Weight initialization method
            ('identity', 'xavier', 'kaiming', 'orthogonal'). Default: 'identity'.
    
    Attributes:
        states_dim (int): Dimension of states.
        controls_dim (int): Dimension of controls.
        parameters_dim (int): Dimension of parameters.
        include_controls (bool): Whether controls are used.
        include_parameters (bool): Whether parameters are used.
        normalization_states (NormalizationLayer1D): Normalizes input states.
        normalization_states_der (NormalizationLayer1D): Normalizes output derivatives.
        normalization_controls (NormalizationLayer1D): Normalizes control inputs.
        normalization_parameters (NormalizationLayer1D): Normalizes parameters.
        system_nn (nn.Sequential): Neural network for derivative function.
    
    Forward Args:

        states (torch.Tensor): State tensor of shape [batch_size, states_dim].
        parameters (torch.Tensor, optional): Parameters [batch_size, parameters_dim].
        controls (torch.Tensor, optional): Controls [batch_size, controls_dim].

    Returns:
        Tuple of (states_der, states_der_norm):
            - states_der (torch.Tensor): Denormalized state derivatives [batch_size, states_dim]
            - states_der_norm (torch.Tensor): Normalized state derivatives [batch_size, states_dim]
    
    Raises:
        AssertionError: If normalization layers of states_der are not initialized before 
            forward pass. If no pre-training is done, this normalization layer must not 
            be initialized.

    """
    
    def __init__(self,
                states_dim,
                controls_dim: int = 0,
                parameters_dim: int = 0,
                hidden_dim: int = 20,
                n_layers: int = 3,
                activation: nn.Module = nn.ELU,
                intialization: str = 'identity',
                ):
        super().__init__()
        self.states_dim = states_dim
        
        self.controls_dim = controls_dim
        self.parameters_dim = parameters_dim

        self.include_controls = True if controls_dim > 0 else False
        self.include_parameters = True if parameters_dim > 0 else False

        self.normalization_states = NormalizationLayer1D(num_features=states_dim)
        self.normalization_states_der = NormalizationLayer1D(num_features=states_dim)
        self.normalization_controls = NormalizationLayer1D(num_features=controls_dim) if self.include_controls else None
        self.normalization_parameters = NormalizationLayer1D(num_features=parameters_dim) if self.include_parameters else None

        # initialize system dynamics
        modules =[
            nn.Linear(states_dim + controls_dim + parameters_dim, hidden_dim),
            activation()
        ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')#
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, states_dim))
        self.system_nn = nn.Sequential(*modules)

        initialize_weights_biases(self.system_nn, method=intialization)
    
    def forward(self, states: torch.Tensor, parameters: Optional[torch.Tensor] = None, controls: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        assert self.normalization_states_der._initialized, "the states derivative normalization layer must be initialized before calling the forward pass"

        # normalize inputs
        states = self.normalization_states(states)
        if self.include_controls:
            controls = self.normalization_controls(controls)
        if self.include_parameters:
            parameters = self.normalization_parameters(parameters)

        # concatenate inputs
        if self.include_controls and self.include_parameters:
            x = torch.cat((states, controls, parameters), dim=1)
        elif self.include_controls:
            x = torch.cat((states, controls), dim=1)
        elif self.include_parameters:
            x = torch.cat((states, parameters), dim=1)
        else:
            x = states
        
        # forward pass system dynamics
        states_der_norm = self.system_nn(x)

        # denormalize output
        states_der = self.normalization_states_der(states_der_norm, denormalize=True)
        
        return states_der, states_der_norm
    

class OutputNetwork(nn.Module):
    """Neural network mapping states to observable outputs.
    
    This module learns the observation function that maps internal states to
    measurable outputs:
        y = g_θ(x, u, p)
    
    Useful when system states are not directly observable or when outputs
    represent derived quantities.
    
    Args:
        states_dim (int): Dimension of state vector x.
        outputs_dim (int): Dimension of output vector y.
        controls_dim (int, optional): Dimension of control inputs. Default: 0.
        parameters_dim (int, optional): Dimension of parameters. Default: 0.
        controls_to_output (bool, optional): Whether to include controls in
            output mapping. Default: False.
        hidden_dim (int, optional): Hidden layer dimension. Default: 20.
        n_layers (int, optional): Number of layers (minimum 2). Default: 3.
        activation (nn.Module, optional): Activation function. Default: nn.ELU.
        intialization (str, optional): Weight initialization method. Default: 'identity'.
    
    Attributes:
        states_dim (int): Dimension of states.
        outputs_dim (int): Dimension of outputs.
        controls_dim (int): Dimension of controls (0 if not used).
        parameters_dim (int): Dimension of parameters.
        include_parameters (bool): Whether parameters are used.
        controls_to_output (bool): Whether controls feed into output network.
        normalization_states (NormalizationLayer1D): Normalizes states.
        normalization_controls (NormalizationLayer1D): Normalizes controls.
        normalization_parameters (NormalizationLayer1D): Normalizes parameters.
        normalization_outputs (NormalizationLayer1D): Normalizes outputs.
        output_nn (nn.Sequential): Neural network for output mapping.
    
    Forward Args:
        states (torch.Tensor): States [batch_size, states_dim].
        parameters (torch.Tensor, optional): Parameters [batch_size, parameters_dim].
        controls (torch.Tensor, optional): Controls [batch_size, controls_dim].
    
    Returns:
        Tuple of (outputs, outputs_norm):
            - outputs (torch.Tensor): Denormalized outputs [batch_size, outputs_dim]
            - outputs_norm (torch.Tensor): Normalized outputs [batch_size, outputs_dim]
    
    Raises:
        AssertionError: If normalization of outputs are not initialized before forward pass.
    

    """

    def __init__(self,
                states_dim,
                outputs_dim,
                controls_dim: int = 0,
                parameters_dim: int = 0,
                controls_to_output: bool = False,
                hidden_dim: int = 20,
                n_layers: int = 3,
                activation: nn.Module = nn.ELU,
                intialization: str = 'identity',
                ):
        super().__init__()

        self.states_dim = states_dim
        self.outputs_dim = outputs_dim
        self.controls_dim = controls_dim if controls_to_output else 0
        self.parameters_dim = parameters_dim
        
        self.include_parameters = True if parameters_dim > 0 else False
        self.controls_to_output = controls_to_output if controls_dim > 0 else False

        self.normalization_states = NormalizationLayer1D(num_features=states_dim)
        self.normalization_controls = NormalizationLayer1D(num_features=controls_dim) if self.controls_to_output else None
        self.normalization_parameters = NormalizationLayer1D(num_features=parameters_dim) if self.include_parameters else None
        self.normalization_outputs = NormalizationLayer1D(num_features=outputs_dim)

        # initialize output nn
        modules = [
            nn.Linear(self.states_dim + self.controls_dim + self.parameters_dim, hidden_dim),
            activation()
        ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, outputs_dim))
        self.output_nn = nn.Sequential(*modules)

        initialize_weights_biases(self.output_nn, method=intialization)

    def forward(self, states: torch.Tensor, parameters: Optional[torch.Tensor] = None, controls: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.normalization_outputs._initialized, "the outputs normalization layer must be initialized before calling the forward pass"

        # normalize inputs
        states = self.normalization_states(states)
        if self.controls_to_output:
            controls = self.normalization_controls(controls)
        if self.include_parameters:
            parameters = self.normalization_parameters(parameters)

        # concatenate inputs
        if self.controls_to_output and self.include_parameters:
            x = torch.cat((states, controls, parameters), dim=1)
        elif self.controls_to_output:
            x = torch.cat((states, controls), dim=1)
        elif self.include_parameters:
            x = torch.cat((states, parameters), dim=1)
        else:
            x = states
        
        # forward pass system dynamics
        outputs_norm = self.output_nn(x)

        # denormalize output
        outputs = self.normalization_outputs(outputs_norm, denormalize=True)
        
        return outputs, outputs_norm
    
class NeuralODE(nn.Module):
    """Complete Neural ODE model for dynamical system learning.
    
    Main class that combines the ODE function, output network, and ODE solver
    for training and inference on continuous-time dynamical systems.
    
    The model learns:
        dx/dt = f_θ(x, u, p)    (ODE function)
        y = g_φ(x, u, p)        (Output function, optional)
    
    And integrates: x(t) = x(t₀) + ∫[t₀,t] f_θ(x(τ), u(τ), p) dτ
    
    Args:
        states_dim (int): Dimension of state vector x.
        controls_dim (int, optional): Dimension of control inputs u. Default: 0.
        parameters_dim (int, optional): Dimension of parameters p. Default: 0.
        outputs_dim (int, optional): Dimension of outputs y. Default: 0.
        controls_to_output_nn (bool, optional): Include controls in output mapping. Default: False.
        hidden_dim (int, optional): Hidden dimension for ODE network. Default: 20.
        n_layers (int, optional): Layers in ODE network. Default: 3.
        hidden_dim_output_nn (int, optional): Hidden dimension for output network. Default: 20.
        n_layers_output_nn (int, optional): Layers in output network. Default: 2.
        activation (nn.Module, optional): Activation function. Default: nn.ELU.
        intialization (str, optional): Initialization for output network. Default: 'identity'.
        initialization_ode (str, optional): Initialization for ODE network. Default: 'identity'.
    
    Attributes:
        include_controls (bool): Whether model uses controls.
        include_parameters (bool): Whether model uses parameters.
        include_outputs (bool): Whether model has output network.
        ode_fun_count (int): Counter for ODE function evaluations.
        NeuralODEFunc (NeuralODEFunc): ODE right-hand side function.
        OutputNetwork (OutputNetwork): Output mapping network (if outputs_dim > 0).
        current_controls (torch.Tensor): Controls for current integration.
        current_times (torch.Tensor): Time points for current integration.
        current_parameters (torch.Tensor): Parameters for current integration.
    
    Methods:
        normalization_init(dataset): Initialize normalization from HDF5 dataset.
        forward(states, parameters, controls, pre_training, times): Forward pass.
        set_input(controls, times, parameters): Set inputs for ODE integration.
        forward_ODE(t, x): ODE function compatible with torchdiffeq.
        model_and_loss_evaluation(...): Compute loss for training/testing.
        get_progress_string(...): Format training progress string.
        save(path): Save model checkpoint.
        load(path, device): Load model checkpoint.
    
    Training Modes:
        **Pre-training (pre_training=True)**:
            - Uses state derivatives directly from data
            - Bypasses ODE solver for fast training
            - Loss: ||dx/dt - f_θ(x)||² + ||y - g_φ(x)||²
            - Requires 'states_der' in dataset
        
        **Main training (pre_training=False)**:
            - Integrates ODE from initial conditions
            - Uses torchdiffeq solver (Euler, RK4, dopri5, etc.)
            - Loss: ||x(t) - x̂(t)||² + ||y(t) - ŷ(t)||²
            - More accurate but computationally expensive
    
    Loss Components:
        - **loss_states**: MSE between true and predicted states (normalized)
        - **loss_outputs**: MSE between true and predicted outputs (normalized)
        - **loss_states_der**: MSE for state derivatives (pre-training only)
        - **rmse_states**: RMSE for states (main training only)
        - **rmse_outputs**: RMSE for outputs (main training only)
    
    Solver Options:
        Configured via train_cfg:
            - **solver**: Solver name ('euler', 'rk4', 'dopri5', etc.)
            - **solver_rtol**: Relative tolerance for adaptive solvers
            - **solver_atol**: Absolute tolerance for adaptive solvers
            - **solver_norm**: Norm for step size control ('max', 'mixed')
            - **use_adjoint**: Use adjoint method for backpropagation
            - **evaluate_at_control_times**: Force evaluation at control time points
    
    
    Notes:
        - Normalization must be initialized before any forward pass
        - Adjoint method saves memory but may be slower for small models
        - Control interpolation uses nearest-neighbor for robustness
        - ODE function counter tracks solver efficiency
        - All tensors use normalized values internally for stability
    
    See Also:
        - ``NeuralODEFunc``: ODE right-hand side network
        - ``OutputNetwork``: Output mapping network
        - ``bnode_core.ode.trainer``: Training pipeline
        - ``torchdiffeq.odeint``: ODE solver
    """
    
    def __init__(self,
                states_dim,
                controls_dim: int = 0,
                parameters_dim: int = 0,
                outputs_dim: int = 0, 
                controls_to_output_nn: bool = False,
                hidden_dim: int = 20,
                n_layers: int = 3,
                hidden_dim_output_nn: int = 20,
                n_layers_output_nn: int = 2,
                activation: nn.Module = nn.ELU,
                intialization: str = 'identity',
                initialization_ode: str = 'identity',
                ): 
        super().__init__()

        self.include_controls = True if controls_dim > 0 else False
        self.include_parameters = True if parameters_dim > 0 else False
        self.include_outputs = True if outputs_dim > 0 else False

        self.ode_fun_count = 0

        self.NeuralODEFunc = NeuralODEFunc(states_dim=states_dim,
                                            controls_dim=controls_dim,
                                            parameters_dim=parameters_dim,
                                            hidden_dim=hidden_dim,
                                            n_layers=n_layers,
                                            activation=activation,
                                            intialization=initialization_ode)
        self.OutputNetwork = OutputNetwork(states_dim=states_dim,
                                            outputs_dim=outputs_dim,
                                            controls_dim=controls_dim,
                                            parameters_dim=parameters_dim,
                                            controls_to_output=controls_to_output_nn,
                                            hidden_dim=hidden_dim_output_nn,
                                            n_layers=n_layers_output_nn,
                                            activation=activation,
                                            intialization=intialization) if outputs_dim > 0 else None
        
    def normalization_init(self, dataset: h5py.File):
        # initialize normalization layers
        def reshape_array(array):
            arr = array.transpose(1,0,2).reshape(array.shape[1],-1).transpose(1,0)
            return arr
        
        # states
        self.NeuralODEFunc.normalization_states.initialize_normalization(reshape_array(dataset['train']['states'][:]))
        if self.include_outputs:
            self.OutputNetwork.normalization_states.initialize_normalization(reshape_array(dataset['train']['states'][:]))

        # states derivative
        if False: # 'states_der' in dataset['train'].keys(): # removed on 2024-11-25 as states_der could be approximated badly by FMU
            _data = dataset['train']['states_der'][:] # TODO maybe remove this as states_der can be approximated badly by FMU?
        else:
            _data = (dataset['train']['states'][:,:,1:] - dataset['train']['states'][:,:,:-1]) / (dataset['time'][1:] - dataset['time'][:-1]).reshape(1,1,-1)
        self.NeuralODEFunc.normalization_states_der.initialize_normalization(reshape_array(_data))

        # controls
        if self.include_controls:
            self.NeuralODEFunc.normalization_controls.initialize_normalization(reshape_array(dataset['train']['controls'][:]))
            if self.include_outputs:
                if self.OutputNetwork.controls_to_output:
                    self.OutputNetwork.normalization_controls.initialize_normalization(reshape_array(dataset['train']['controls'][:]))
        # parameters
        if self.include_parameters:
            self.NeuralODEFunc.normalization_parameters.initialize_normalization(dataset['train']['parameters'][:])
            self.OutputNetwork.normalization_parameters.initialize_normalization(dataset['train']['parameters'][:])

        # outputs
        if self.include_outputs:
            self.OutputNetwork.normalization_outputs.initialize_normalization(reshape_array(dataset['train']['outputs'][:]))

        logging.info('Initialized normalization layers')

    def forward(self, 
                states, 
                parameters = None, 
                controls = None, 
                pre_training = False,
                times = None):
        if pre_training is True:
            states_der, states_der_norm = self.NeuralODEFunc(states, parameters, controls)
            outputs, outputs_norm = self.OutputNetwork(states, parameters, controls) if self.include_outputs else (None, None)
            return states_der, outputs, states_der_norm, outputs_norm
        else:
            states_der, _ = self.NeuralODEFunc(states, parameters, controls)
            # the output network needs to be called from external with NeuralOde.OutputNetwork(states, parameters, controls)
            return states_der
    
    def set_input(self, controls=None, times=None, parameters=None):
        self.current_controls = controls
        self.current_times = times
        self.current_parameters = parameters

    def forward_ODE(self, t, x):
        if self.include_controls:
            try:
                idx = torch.nonzero(self.current_times[0][0] > t)
                if len(idx) == 0:
                    idx = -1
                else:
                    idx = idx[0][0] - 1
                u = self.current_controls[:,:,idx]
            except:
                u = self.current_controls[:,:,-1]
                warnings.warn('something went wrong when trying to get the control input at time t, using the last control input instead')
        else:
            u = None
        # call
        x_dot = self.__call__(x.swapaxes(0,1), self.current_parameters, u)
        x_dot = x_dot.swapaxes(0,1)
        self.ode_fun_count += 1
        return x_dot
    
    def model_and_loss_evaluation(self, data_batch, train_cfg, pre_train, device, return_model_outputs, test = False, last_batch=True, activate_deterministic_mode = False): # last two arguments for compatibility with trainer
        # get data
        time = data_batch['time'].to(device)
        states = data_batch['states'].to(device)
        if 'states_der' in data_batch.keys():
            states_der = data_batch['states_der'].to(device)
        parameters = data_batch['parameters'].to(device) if 'parameters' in data_batch.keys() else None
        controls = data_batch['controls'].to(device) if 'controls' in data_batch.keys() else None
        outputs = data_batch['outputs'].to(device) if 'outputs' in data_batch.keys() else None
        # squeeze data if in pre_train
        if pre_train is True:
            time = time.squeeze(2)
            states = states.squeeze(2)
            states_der = states_der.squeeze(2)
            controls = controls.squeeze(2) if controls is not None else None
            outputs = outputs.squeeze(2) if outputs is not None else None
                
        """pre-training"""
        if pre_train is True:
            # forward pass
            states_der_hat, outputs_hat, states_der_norm_hat, outputs_norm_hat = self.__call__(states, parameters, controls, pre_training = True)
            # get norms
            states_der_norm = self.NeuralODEFunc.normalization_states_der(states_der).detach()
            outputs_norm = self.OutputNetwork.normalization_outputs(outputs).detach() if self.include_outputs else None
            # calculate losses
            loss_states_der = torch.mean(torch.square((states_der_norm - states_der_norm_hat)))
            loss_outputs = torch.mean(torch.square(outputs_norm - outputs_norm_hat)) if self.include_outputs else torch.zeros(1).to(device)
            loss = loss_states_der + loss_outputs
            # make return values
            ret_val = {
                'loss': loss,
                'loss_states_der': loss_states_der,
                'loss_outputs': loss_outputs,
            }
            if return_model_outputs:
                model_outputs = {
                    'states_der_hat': states_der_hat,
                    'outputs_hat': outputs_hat,
                }
            if test is True:
                # call value.item() for each value in return_value
                ret_val = dict({key: value.item() for key, value in ret_val.items()})
            # detach model outputs from computational graph
            if return_model_outputs:
                model_outputs = dict({key: value.cpu().detach().numpy() for key, value in model_outputs.items()})
            return ret_val if return_model_outputs is False else (ret_val, model_outputs)
        
        """training"""
        if pre_train is False:
            _time_logging0 = pyTime.time()
            # provide inputs to NeuralODE
            self.set_input(controls, time, parameters)
            self.ode_fun_count = 0
            # forward pass
            x0 = states[:, :, 0].swapaxes(0,1) # x is shape [batch_size, states_dim, timeseries_length], but for odeint it must be [states_dim, batch_size]
            time = time[0,0,:] # as we used equidistant time steps in data generation, we can use the first time vector
            # specify options for odeint
            _fixed_step_solvers = ['euler', 'midpoint', 'rk4', 'implicit_adams', 'explicit_adams']
            _base_options = {}
            if train_cfg.solver_norm == 'max':
                _base_options['norm'] = torchdiffeq._impl.misc._linf_norm
            elif train_cfg.solver_norm == 'mixed':
                _base_options['norm'] = _mixed_norm_tensor
            if self.include_controls or train_cfg.evaluate_at_control_times is True:
                if train_cfg.evaluate_at_control_times is True:
                    if train_cfg.solver in _fixed_step_solvers:
                        _base_options['perturb'] = True
                    else:
                        _base_options['jump_t'] = time
            options = _base_options.copy()

            if train_cfg.use_adjoint is True and train_cfg.solver not in ['euler', 'midpoint', 'rk4', 'implicit_adams', 'explicit_adams']:
                adjoint_options = _base_options.copy()
                adjoint_options['norm'] = 'seminorm'
                states_hat = odeint_adjoint(self.forward_ODE, x0, time,
                                method=train_cfg.solver, 
                                rtol = train_cfg.solver_rtol, 
                                atol = train_cfg.solver_atol,
                                adjoint_params=self.parameters(),
                                adjoint_options=adjoint_options,
                                options=options)
            else:
                states_hat = odeint(self.forward_ODE, x0, time, 
                                method=train_cfg.solver,
                                rtol = train_cfg.solver_rtol, 
                                atol = train_cfg.solver_atol,
                                options=options)
            time_odeint = pyTime.time() - _time_logging0
            _time_logging0 = pyTime.time()
            ode_calls_forward = self.ode_fun_count
            self.ode_fun_count = 0 # reset ode_fun_count for adjoint pass
            # x is of shape [timeseries_length, states_dim, batch_size], but we need [batch_size, states_dim, timeseries_length]
            states_hat = states_hat.swapaxes(0,2)
            # calculate outputs
            if self.include_outputs:
                outputs_hat = torch.empty((states_hat.shape[0], self.OutputNetwork.outputs_dim , states_hat.shape[2])).to(device)
                for i in range(states_hat.shape[2]):
                    if self.OutputNetwork.controls_to_output is True:
                        outputs_hat[:,:,i], _ = self.OutputNetwork(states = states_hat[:,:,i], controls = controls[:,:,i], parameters = parameters)
                    else:
                        outputs_hat[:,:,i], _ = self.OutputNetwork(states_hat[:,:,i], controls = None, parameters = parameters)
            time_outputs = pyTime.time() - _time_logging0
            # maybe something like this is faster:
            # test = x_norm.reshape(2,-1).swapaxes(0,1)
            # print(test.shape)
            # test = test.swapaxes(0,1).reshape(512,2,1000)
            # print(test.shape)

            # calculate loss
            # normalize states and outputs
            states_norm = torch.empty_like(states).to(device)
            outputs_norm = torch.empty_like(outputs).to(device) if self.include_outputs else None
            states_hat_norm = torch.empty_like(states_hat).to(device)
            outputs_hat_norm = torch.empty_like(outputs_hat).to(device) if self.include_outputs else None

            for i in range(states.shape[2]):
                states_norm[:,:,i] = self.NeuralODEFunc.normalization_states(states[:,:,i])
                states_hat_norm[:,:,i] = self.NeuralODEFunc.normalization_states(states_hat[:,:,i])
                if self.include_outputs:
                    outputs_norm[:,:,i] = self.OutputNetwork.normalization_outputs(outputs[:,:,i])
                    outputs_hat_norm[:,:,i] = self.OutputNetwork.normalization_outputs(outputs_hat[:,:,i])

            loss_states = torch.mean(torch.square((states_norm - states_hat_norm)))
            loss_outputs = torch.mean(torch.square(outputs_norm - outputs_hat_norm)) if self.include_outputs else torch.zeros(1).to(device)
            loss = loss_states + loss_outputs
            rmse_states = torch.sqrt(torch.mean(torch.square((states_norm - states_hat_norm))))
            rmse_outputs = torch.sqrt(torch.mean(torch.square(outputs_norm - outputs_hat_norm))) if self.include_outputs else torch.zeros(1).to(device)
            # make ret_vals
            ret_val = {
                'loss': loss,
                'loss_states': loss_states,
                'loss_outputs': loss_outputs,
                'rmse_states': rmse_states,
                'rmse_outputs': rmse_outputs,
            }
            if test is True:
                # call value.item() for each value in return_value
                ret_val = dict({key: value.item() for key, value in ret_val.items()})
            ret_val['time_odeint'] = time_odeint
            ret_val['time_outputs'] = time_outputs
            ret_val['ode_calls_forward'] = ode_calls_forward
            # append model output if necessary
            if return_model_outputs:
                model_outputs = {
                    'states_hat': states_hat,
                }
                if self.include_outputs:
                    model_outputs['outputs_hat'] = outputs_hat
            # detach model outputs from computational graph
            if return_model_outputs:
                model_outputs = dict({key: value.cpu().detach().numpy() for key, value in model_outputs.items()})
            return ret_val if return_model_outputs is False else (ret_val, model_outputs)
    
    def get_progress_string(self, ret_vals_train, ret_vals_validation, ret_vals_test, pre_train):
        if pre_train is True:
            _str = '[train/val/test] loss: {:.5f}/{:.5f}/{:.5f} | loss_states_der: {:.5f}/{:.5f}/{:.5f} | loss_outputs: {:.5f}/{:.5f}/{:.5f}'.format(
            ret_vals_train['loss'], ret_vals_validation['loss'], ret_vals_test['loss'],
            ret_vals_train['loss_states_der'], ret_vals_validation['loss_states_der'], ret_vals_test['loss_states_der'],
            ret_vals_train['loss_outputs'], ret_vals_validation['loss_outputs'], ret_vals_test['loss_outputs']
        )
        else:
            try:
                _str =  '[train/val/test] loss: {:.5f}/{:.5f}/{:.5f} | loss_states: {:.5f}/{:.5f}/{:.5f} | loss_outputs: {:.5f}/{:.5f}/{:.5f} | \n \t \t|rmse_states: {:.3f}/{:.3f}/{:.3f} | rmse_outputs: {:.3f}/{:.3f}/{:.3f} | time_forward: {:.5f} | time_backward: {:.5f}'.format(
                    ret_vals_train['loss'], ret_vals_validation['loss'], ret_vals_test['loss'],
                    ret_vals_train['loss_states'], ret_vals_validation['loss_states'], ret_vals_test['loss_states'],
                    ret_vals_train['loss_outputs'], ret_vals_validation['loss_outputs'], ret_vals_test['loss_outputs'],
                    ret_vals_train['rmse_states'], ret_vals_validation['rmse_states'], ret_vals_test['rmse_states'],
                    ret_vals_train['rmse_outputs'], ret_vals_validation['rmse_outputs'], ret_vals_test['rmse_outputs'],
                    ret_vals_train['time_forward'], ret_vals_train['time_backward']
                )
            except:
                _str = 'error in get_progress_string'
        return _str
    
    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)
        logging.info('\t \t \tSaved model to {}'.format(path))
    
    def load(self, path: Path, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))
        logging.info('\tLoaded model from {}'.format(path))