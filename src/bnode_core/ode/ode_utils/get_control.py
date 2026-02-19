
import torch
import logging

def get_control_input_at_t(
    t: float,
    ctrl_times: torch.Tensor,
    ctrl_inputs: torch.Tensor,
    use_input_smoother: bool = False
) -> torch.Tensor:
    """Return control input at time ``t``.

    Semantics (for monotonically increasing 1D ``ctrl_times``):

    - Without smoothing (``use_input_smoother=False``):
      the control is piecewise constant with a one-step delay:
      for example, for ``ctrl_times = [0, 1, 2, 3, ...]``
      we obtain
        * ``t in [0, 1]   -> u(t=0)``
        * ``t in (1, 2]   -> u(t=1)``
        * ``t in (2, 3]   -> u(t=2)``

    - With smoothing (``use_input_smoother=True``):
      for intervals beyond the first one we linearly interpolate
      between the *previous* and the *current* control value.
      With ``ctrl_times = [0, 1, 2, 3, ...]`` and controls
      ``u(0), u(1), u(2), ...`` this gives, e.g.::

          t = 1.5  -> 0.5 * (u(0) + u(1))

      which matches the desired behaviour you described.
    """

    # Guard against degenerate ctrl_times
    if ctrl_times.numel() == 0:
        raise ValueError("ctrl_times must contain at least one element")

    t_eff = float(t)
    last_time = float(ctrl_times[-1])
    if t_eff > last_time:
        logging.warning(
            "t is larger than the last time point in ctrl_times, using the last control value"
        )
        t_eff = last_time

    # --- No input smoother: piecewise constant with value ctrl[i] on [t_i, t_{i+1}) ---
    if not use_input_smoother or ctrl_times.numel() < 2:
        # Use right=True so that t == ctrl_times[i] maps to index i
        idx = torch.searchsorted(ctrl_times, torch.tensor(t_eff, dtype=ctrl_times.dtype), right=True).item() - 1
        # Clamp into valid range [0, n-1]
        idx = max(0, min(idx, ctrl_times.numel() - 1))
        u_now = ctrl_inputs[:, :, idx]
    
    else:
        # --- With input smoother ---
        # We want for i >= 1:
        #   - at t = ctrl_times[i]          -> u = ctrl[i-1]
        #   - at t_mid between t_i, t_{i+1} -> u = (ctrl[i-1] + ctrl[i]) / 2
        # To achieve this, for t in [t_i, t_{i+1}) we linearly interpolate
        # between ctrl[i-1] (at t_i) and ctrl[i] (at t_{i+1}).

        # First determine the interval index i such that t in [t_i, t_{i+1})
        idx = torch.searchsorted(ctrl_times, torch.tensor(t_eff, dtype=ctrl_times.dtype), right=True).item() - 1
        idx = max(0, min(idx, ctrl_times.numel() - 1))

        # For the very first interval we cannot look back; just use ctrl[0].
        if idx == 0:
             u_now = ctrl_inputs[:, :, 0]
        else:
            # Define interpolation between ctrl[idx-1] at t_idx and ctrl[idx] at t_{idx+1}
            t_i = float(ctrl_times[idx])
            if idx + 1 < ctrl_times.numel():
                t_ip1 = float(ctrl_times[idx + 1])
                if t_ip1 == t_i:
                    raise ValueError(
                        f"ctrl_times contains duplicate entries at indices {idx} and {idx+1}, which is not allowed for smoothing"
                    )
                alpha = (t_eff - t_i) / (t_ip1 - t_i)
                u_prev = ctrl_inputs[:, :, idx - 1]
                u_curr = ctrl_inputs[:, :, idx]
                u_now =  (1.0 - alpha) * u_prev + alpha * u_curr
            else:
                # If there is no next time point, fall back to constant ctrl[idx]
                u_now = ctrl_inputs[:, :, idx-1]
    return u_now, idx