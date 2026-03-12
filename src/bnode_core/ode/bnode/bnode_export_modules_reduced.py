"""
In this file, we implement wrappers for the ONNX export of a BNODE trained in deterministic mode. 
This means, that the trained BNODE still uses the initial high-dimensional latent space, but the 
exported ONNX model has an interface that only provides the low-dimensional latent space as input and output.
This allows for more efficient inference.
"""
import torch 
import torch.nn as nn


def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
     return mask.clone().detach().to(dtype=torch.bool)

class encoder_wrapped(nn.Module):
     def __init__(self, encoder, mask_states, mask_controls = None, mask_parameters = None):
          super(encoder_wrapped, self).__init__()
          self.encoder = encoder
          mask_states = _to_bool_mask(mask_states)
          self.lat_state_dim = mask_states.sum().item()  # Number of True values in the state mask
          self.register_buffer('mask_states', mask_states)
          if mask_controls is not None:
               mask_controls = _to_bool_mask(mask_controls)
               self.lat_control_dim = mask_controls.sum().item()  # Number of True values in the control mask
               self.register_buffer('mask_controls', mask_controls)
          else:
               self.lat_control_dim = 0
               self.mask_controls = None
          if mask_parameters is not None:
               mask_parameters = _to_bool_mask(mask_parameters)
               self.lat_parameter_dim = mask_parameters.sum().item()  # Number of True values in the parameter mask
               self.register_buffer('mask_parameters', mask_parameters)
          else:
               self.lat_parameter_dim = 0
               self.mask_parameters = None
          
          # unset mask in encoder, since we will handle it here
          self.encoder.mask_set.set_(torch.tensor(False, device=self.mask_states.device))

     def forward(self, x, params = None, controls = None):
          mu, logvar = self.encoder(x, params, controls)
          return mu[:, self.mask_states]

class decoder_wrapped(nn.Module):
     def __init__(self, decoder: nn.Module, mask_states: torch.Tensor, mask_controls: torch.Tensor = None, mask_parameters: torch.Tensor = None):
          super(decoder_wrapped, self).__init__()
          self.decoder = decoder
          mask_states = _to_bool_mask(mask_states)
          self.register_buffer('mask_states', mask_states)
          if mask_controls is not None:
               mask_controls = _to_bool_mask(mask_controls)
               self.register_buffer('mask_controls', mask_controls)
          else:
               self.mask_controls = None
          if mask_parameters is not None:
               mask_parameters = _to_bool_mask(mask_parameters)
               self.register_buffer('mask_parameters', mask_parameters)
          else:
               self.mask_parameters = None

     def forward(self, lat_state, lat_parameters = None, lat_controls = None, feedthrough_controls = None):
          lat_states_full = torch.zeros((lat_state.shape[0], self.mask_states.shape[0]), device=lat_state.device)
          lat_states_full[:, self.mask_states] = lat_state
          if self.mask_controls is not None and lat_controls is not None:
               lat_controls_full = torch.zeros((lat_controls.shape[0], self.mask_controls.shape[0]), device=lat_controls.device)
               lat_controls_full[:, self.mask_controls] = lat_controls
          else:
               lat_controls_full = None
          if self.mask_parameters is not None and lat_parameters is not None:
               lat_parameters_full = torch.zeros((lat_parameters.shape[0], self.mask_parameters.shape[0]), device=lat_parameters.device)
               lat_parameters_full[:, self.mask_parameters] = lat_parameters
          else:
               lat_parameters_full = None
          
          return self.decoder(lat_states_full, lat_parameters_full, lat_controls_full, feedthrough_controls)          

class ssm_from_param_wrapped(nn.Module):
     def __init__(self):
          raise NotImplementedError("The ssm_from_param wrapper is not implemented yet. Please implement it if you want to export a BNODE with ssm_from_param decoder.")

class ode_wrapped(nn.Module):
     def __init__(self, ode, mask_states, mask_controls = None, mask_parameters = None):
          super(ode_wrapped, self).__init__()
          self.ode = ode
          if self.ode.lat_ode_type != 'variance_constant':
               raise ValueError("The ODE wrapper is only implemented for BNODEs with variance_constant latent ODE type.")
          mask_states = _to_bool_mask(mask_states)
          self.register_buffer('mask_states', mask_states)
          if mask_controls is not None:
               mask_controls = _to_bool_mask(mask_controls)
               self.register_buffer('mask_controls', mask_controls)
          else:
               self.mask_controls = None
          if mask_parameters is not None:
               mask_parameters = _to_bool_mask(mask_parameters)
               self.register_buffer('mask_parameters', mask_parameters)
          else:
               self.mask_parameters = None
          
          # unset mask in ode, since we will handle it here
          self.ode.mask_set.set_(torch.tensor(False, device=self.mask_states.device))

     def forward(self, lat_states, lat_parameters = None, lat_controls = None, A_from_param: torch.Tensor = None, B_from_param: torch.Tensor = None):
          if A_from_param is not None:
               raise NotImplementedError("The ODE wrapper is not implemented for BNODEs with A_from_param. Please implement it if you want to export a BNODE with A_from_param.")

          lat_states_full = torch.zeros((lat_states.shape[0], self.mask_states.shape[0]), device=lat_states.device)
          lat_states_full[:, self.mask_states] = lat_states
          if self.mask_controls is not None and lat_controls is not None:
               lat_controls_full = torch.zeros((lat_controls.shape[0], self.mask_controls.shape[0]), device=lat_controls.device)
               lat_controls_full[:, self.mask_controls] = lat_controls
          else:
               lat_controls_full = None
          if self.mask_parameters is not None and lat_parameters is not None:
               lat_parameters_full = torch.zeros((lat_parameters.shape[0], self.mask_parameters.shape[0]), device=lat_parameters.device)
               lat_parameters_full[:, self.mask_parameters] = lat_parameters
          else:
               lat_parameters_full = None
          
          lat_states_dot_full = self.ode(lat_states_full, lat_parameters_full, lat_controls_full, A_from_param, B_from_param)
          
          return lat_states_dot_full[:, self.mask_states]