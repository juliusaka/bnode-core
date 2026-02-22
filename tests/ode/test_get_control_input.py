import torch

from bnode_core.ode.ode_utils.get_control import get_control_input_at_t


def _make_simple_ctrl_data(batch_size: int = 3, channels: int = 2):
    """Create simple control data: value equals time index for all batches/channels."""
    ctrl_times = torch.linspace(0.0, 5.0, steps=10, dtype=torch.float32)  # 10 time points from 0 to 5
    ctrl_inputs = torch.rand(batch_size, channels, ctrl_times.shape[0], dtype=torch.float32)  # random control values
    return ctrl_times, ctrl_inputs


def test_no_smoother_piecewise_constant():
    """Without input smoother, u stays constant on (t_i, t_{i+1}] with value ctrl[i]."""
    ctrl_times, ctrl_inputs = _make_simple_ctrl_data()

    for i in range(ctrl_times.numel() - 1):
        t_i = float(ctrl_times[i])
        t_ip1 = float(ctrl_times[i + 1])
        t_mid = 0.5 * (t_i + t_ip1)

        u, _ = get_control_input_at_t(t_i, ctrl_times, ctrl_inputs, use_input_smoother=False)
        expected = ctrl_inputs[:, :, i]
        assert torch.allclose(u, expected), f"no-smoother: expected u[{i}] at t = ctrl_times[{i}]"

        # For each interval (t_i, t_{i+1}], mid-point should return ctrl[i].
        u, _ = get_control_input_at_t(t_mid, ctrl_times, ctrl_inputs, use_input_smoother=False)
        expected = ctrl_inputs[:, :, i]
        assert torch.allclose(u, expected), f"no-smoother: expected u[{i}] at t_mid between t[{i}] and t[{i+1}], got {u} instead of {expected}"

        


def test_smoother_matches_conditions_at_sample_and_midpoint():
    """With input smoother, check only the key conditions.

    For i >= 1:
      - at t = ctrl_times[i], we expect u = ctrl[i-1]
      - at t_mid = (ctrl_times[i] + ctrl_times[i+1]) / 2,
        we expect u = (ctrl[i-1] + ctrl[i]) / 2
    """
    ctrl_times, ctrl_inputs = _make_simple_ctrl_data()

    for i in range(1, ctrl_times.numel() - 1):
        
        # Condition 1: t = ctrl_times[i] -> u = ctrl[i-1]
        t_sample = float(ctrl_times[i])
        u_sample, _ = get_control_input_at_t(t_sample, ctrl_times, ctrl_inputs, use_input_smoother=True)
        expected_sample = ctrl_inputs[:, :, i - 1]
        assert torch.allclose(
            u_sample, expected_sample
        ), f"smoother: expected u[{i-1}] at t = ctrl_times[{i}]"

        # Condition 2: midpoint between ctrl_times[i] and ctrl_times[i+1]
        t_i = float(ctrl_times[i])
        t_ip1 = float(ctrl_times[i + 1])
        t_mid = 0.5 * (t_i + t_ip1)
        u_mid, _ = get_control_input_at_t(t_mid, ctrl_times, ctrl_inputs, use_input_smoother=True)
        expected_mid = 0.5 * (ctrl_inputs[:, :, i - 1] + ctrl_inputs[:, :, i])
        assert torch.allclose(
            u_mid, expected_mid
        ), f"smoother: expected (u[{i-1}] + u[{i}])/2 at midpoint between t[{i}] and t[{i+1}]"


def test_get_control_input_batches_and_channels():
    """ctrl_inputs with multiple batches and channels, verifying broadcasting over them."""
    batch_size, channels = 3, 4
    ctrl_times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    t_steps = ctrl_times.numel()

    # Each batch/channel has a distinct linear pattern so we can detect mis-indexing.
    ctrl_inputs = torch.zeros(batch_size, channels, t_steps, dtype=torch.float32)
    for b in range(batch_size):
        for c in range(channels):
            base = b * 10.0 + c * 1.0
            ctrl_inputs[b, c, :] = base + torch.arange(t_steps, dtype=torch.float32)

    # Use a mid-interval time between 1 and 2.
    i = 1
    t_i = float(ctrl_times[i])
    t_ip1 = float(ctrl_times[i + 1])
    t_mid = 0.5 * (t_i + t_ip1)

    # Without smoothing: should pick ctrl_inputs[:, :, i]
    u_no_smooth, _ = get_control_input_at_t(t_mid, ctrl_times, ctrl_inputs, use_input_smoother=False)
    assert u_no_smooth.shape == (batch_size, channels)
    expected_no_smooth = ctrl_inputs[:, :, i]
    assert torch.allclose(u_no_smooth, expected_no_smooth)

    # With smoothing: should give the average of ctrl[:, :, i-1] and ctrl[:, :, i]
    u_smooth, _ = get_control_input_at_t(t_mid, ctrl_times, ctrl_inputs, use_input_smoother=True)
    assert u_smooth.shape == (batch_size, channels)
    expected_smooth = 0.5 * (ctrl_inputs[:, :, i - 1] + ctrl_inputs[:, :, i])
    assert torch.allclose(u_smooth, expected_smooth)


def test_kwargs_tensors_match_as_ctrl_inputs():
    """Tensors passed via kwargs should behave like ctrl_inputs.

    For a given auxiliary tensor `aux`, using

        get_control_input_at_t(..., ctrl_inputs, aux=aux)

    must produce the same `aux` output (in kwargs_res) as when
    calling

        get_control_input_at_t(..., aux)

    directly as `ctrl_inputs`, for the same `t` and `use_input_smoother`.
    """
    ctrl_times, ctrl_inputs = _make_simple_ctrl_data(batch_size=2, channels=3)
    aux = torch.randn_like(ctrl_inputs)

    # Use a few representative times: first sample, a midpoint, and a later sample.
    t_samples = [
        float(ctrl_times[1]),
        0.5 * (float(ctrl_times[2]) + float(ctrl_times[3])),
        float(ctrl_times[5]),
    ]

    for use_smoother in (False, True):
        for t in t_samples:
            # Call with auxiliary tensor in kwargs
            u_main, idx1, aux_main = get_control_input_at_t(
                t,
                ctrl_times,
                ctrl_inputs,
                use_input_smoother=use_smoother,
                aux=aux,
            )

            # Call treating the same auxiliary tensor as the primary ctrl_inputs
            u_aux, idx2 = get_control_input_at_t(
                t,
                ctrl_times,
                aux,
                use_input_smoother=use_smoother,
            )

            # Indices must match, and kwarg result must equal the
            # value we would obtain when using the tensor as ctrl_inputs.
            assert idx1 == idx2
            assert torch.allclose(aux_main, u_aux)


if __name__ == "__main__":
    # Visualization: compare input smoother vs. usual mode for a sign-like control signal.
    import numpy as np
    import matplotlib.pyplot as plt

    # Define control times and a sign-like control sequence (-1, +1, -1, +1, ...).
    ctrl_times = torch.linspace(0.0, 1.6, steps=20, dtype=torch.float32)
    def ctr_func(t):
        return np.sin(1.0 * np.pi * t / ctrl_times[-1])  # oscillates between -1 and +1 over [0, 1.6]
    ctrl_inputs = torch.tensor([ctr_func(t) for t in ctrl_times], dtype=torch.float32).view(1, 1, -1)  # (batch=1, channels=1, time)
    eps_inputs = torch.randn_like(ctrl_inputs)  # small noise for eps if needed
    ctrl_inputs_noise = ctrl_inputs + 0.1 * eps_inputs

    # Dense time grid for plotting.
    t_dense = torch.linspace(float(ctrl_times[0]), float(ctrl_times[-1]), steps=1000)
    ctrl_dense = torch.tensor([ctr_func(t) for t in t_dense], dtype=torch.float32).view(1, 1, -1)

    print(" Control times:", ctrl_times[:10], "..."
          "\n Control inputs:", ctrl_inputs.squeeze()[:10], "...")

    u_step = []
    u_smooth = []
    u_step_noise = []
    u_smooth_noise = []
    eps_step = []
    eps_smooth = []
    
    for t in t_dense:
        t_float = float(t)
        u, _ = get_control_input_at_t(t_float, ctrl_times, ctrl_inputs, use_input_smoother=False)
        u_step.append(u.squeeze().item())        
        u, _ = get_control_input_at_t(t_float, ctrl_times, ctrl_inputs, use_input_smoother=True)
        u_smooth.append(u.squeeze().item())
        ## noise case
        u, _ = get_control_input_at_t(t_float, ctrl_times, ctrl_inputs_noise, use_input_smoother=False)
        u_step_noise.append(u.squeeze().item())
        u, _ = get_control_input_at_t(t_float, ctrl_times, ctrl_inputs_noise, use_input_smoother=True)
        u_smooth_noise.append(u.squeeze().item())
        ## eps case
        eps, _ = get_control_input_at_t(t_float, ctrl_times, eps_inputs, use_input_smoother=False)
        eps_step.append(eps.squeeze().item())
        eps, _ = get_control_input_at_t(t_float, ctrl_times, eps_inputs, use_input_smoother=True)
        eps_smooth.append(eps.squeeze().item())

    t_np = t_dense.detach().cpu().numpy()
    u_step_np = np.array(u_step)
    u_smooth_np = np.array(u_smooth)
    u_step_noise_np = np.array(u_step_noise)
    u_smooth_noise_np = np.array(u_smooth_noise)
    eps_step_np = np.array(eps_step)
    eps_smooth_np = np.array(eps_smooth)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].step(t_np, u_step_np, label="usual (step)")
    ax[0].plot(t_np, u_smooth_np, label="input smoother", linestyle="--")
    ax[0].scatter(
        ctrl_times.detach().cpu().numpy(),
        ctrl_inputs.squeeze().detach().cpu().numpy(),
        color="black",
        zorder=1,
        label="control samples",
    )
    ax[0].plot(t_np, ctrl_dense.squeeze().detach().cpu().numpy(), color="gray", alpha=0.5, label="true control")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("u(t)")
    ax[0].set_title("Control input: usual mode vs. input smoother")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].step(t_np, u_step_noise_np, label="usual mode + noise")
    ax[1].plot(t_np, u_smooth_noise_np, label="input smoother + noise", linestyle="--")
    ax[1].scatter(
        ctrl_times.detach().cpu().numpy(),
        ctrl_inputs_noise.squeeze().detach().cpu().numpy(),
        color="black",
        zorder=1,
        label="control samples + noise",
    )
    ax[1].plot(t_np, ctrl_dense.squeeze().detach().cpu().numpy(), color="gray", alpha=0.5, label="true control")

    ax[1].set_xlabel("t")
    ax[1].set_ylabel("u(t)")
    ax[1].set_title("Control input with noise: usual mode vs. input smoother")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].step(t_np, eps_step_np, label="usual mode")
    ax[2].plot(t_np, eps_smooth_np, label="input smoother", linestyle="--")
    ax[2].scatter(
        ctrl_times.detach().cpu().numpy(),
        eps_inputs.squeeze().detach().cpu().numpy(),
        color="black",
        zorder=1,
        label="eps samples",
    )
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("eps(t)")
    ax[2].set_title("Noise input: usual mode vs. input smoother")
    ax[2].legend()
    ax[2].grid(True)




    plt.tight_layout()
    plt.show()
