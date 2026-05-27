"""Sampling methods for raw data generation.

Each sub-module implements one sampling strategy:

- :mod:`parameters` — uniform random sampling of model parameters
- :mod:`initial_states` — uniform random sampling of initial state values
- :mod:`controls_random` — uniform random sampling of control inputs (strategy ``R``)
- :mod:`controls_random_offset` — random controls with bounded offset (strategy ``RO``)
- :mod:`controls_rocs` — cubic-spline controls with manual clipping (strategy ``ROCS``)
- :mod:`controls_rrocs` — cubic-spline controls with random rescaling (strategy ``RROCS``)
- :mod:`controls_steps` — step-change controls for step-response tests (strategy ``RS``)
- :mod:`controls_frequency_response` — chirp/frequency-sweep controls (strategy ``RF``)
- :mod:`controls_from_file` — controls loaded from a CSV file (strategy ``file``)
- :mod:`controls_from_excel` — constant controls loaded from Excel (strategy ``constantInput``)
"""

from bnode_core.data_generation.sampling.parameters import random_sampling_parameters
from bnode_core.data_generation.sampling.initial_states import random_sampling_initial_states
from bnode_core.data_generation.sampling.controls_random import random_sampling_controls
from bnode_core.data_generation.sampling.controls_random_offset import random_sampling_controls_w_offset
from bnode_core.data_generation.sampling.controls_rocs import random_sampling_controls_w_offset_cubic_splines_old_clip_manual
from bnode_core.data_generation.sampling.controls_rrocs import random_sampling_controls_w_offset_cubic_splines_clip_random
from bnode_core.data_generation.sampling.controls_steps import random_steps_sampling_controls
from bnode_core.data_generation.sampling.controls_sequential_steps import random_sequential_steps_sampling_controls
from bnode_core.data_generation.sampling.controls_frequency_response import random_frequency_response_sampling_controls
from bnode_core.data_generation.sampling.controls_fourrier import random_fourrier_sampling_controls
from bnode_core.data_generation.sampling.controls_from_file import load_controls_from_file
from bnode_core.data_generation.sampling.controls_from_excel import constant_input_simulation_from_excel

__all__ = [
    "random_sampling_parameters",
    "random_sampling_initial_states",
    "random_sampling_controls",
    "random_sampling_controls_w_offset",
    "random_sampling_controls_w_offset_cubic_splines_old_clip_manual",
    "random_sampling_controls_w_offset_cubic_splines_clip_random",
    "random_steps_sampling_controls",
    "random_sequential_steps_sampling_controls",
    "random_frequency_response_sampling_controls",
    "random_fourrier_sampling_controls",
    "load_controls_from_file",
    "constant_input_simulation_from_excel",
]
