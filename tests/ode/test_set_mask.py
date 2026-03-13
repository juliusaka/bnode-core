"""
Unit tests for set_mask weight-trimming in GeneralEncoder, LatentODEFunc, and Decoder.

Verifies that:
- Output shapes are correct after trimming
- Numerical equivalence: trimmed forward on reduced inputs == full forward with zeros at unpopulated dims
"""
import copy
import pytest
import torch
import torch.nn as nn
from bnode_core.ode.bnode.bnode_modules import GeneralEncoder, LatentODEFunc, Decoder


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_mask(full_dim: int, populated: list[int]) -> torch.Tensor:
    """Create a binary mask with 1s at *populated* indices."""
    mask = torch.zeros(full_dim)
    mask[populated] = 1.0
    return mask


def _embed_reduced(full_dim: int, populated: list[int], reduced: torch.Tensor) -> torch.Tensor:
    """Scatter *reduced* tensor into a full-size tensor at *populated* indices (rest zero)."""
    batch = reduced.shape[0]
    full = torch.zeros(batch, full_dim)
    full[:, populated] = reduced
    return full


BATCH = 4
SEED = 42

# ─── GeneralEncoder ──────────────────────────────────────────────────────────


class TestGeneralEncoderSetMask:
    """Tests for GeneralEncoder.set_mask weight trimming."""

    @pytest.fixture()
    def encoder(self):
        torch.manual_seed(SEED)
        return GeneralEncoder(
            input_dim=10, lat_dim=8, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None,
        )

    @pytest.fixture()
    def encoder_with_params_controls(self):
        torch.manual_seed(SEED)
        return GeneralEncoder(
            input_dim=10, lat_dim=8, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None,
            include_parameters=True, param_dim=3,
            include_controls=True, controls_dim=5,
        )

    def test_output_shape(self, encoder):
        populated = [0, 2, 5]
        mask = _make_mask(8, populated)
        encoder.set_mask(mask)
        x = torch.randn(BATCH, 10)
        out = encoder(x)
        assert out.shape == (BATCH, len(populated))

    def test_numerical_equivalence(self, encoder):
        """Trimmed output should match indexing mu from full forward."""
        populated = [1, 3, 7]
        mask = _make_mask(8, populated)
        enc_full = copy.deepcopy(encoder)
        # full forward (normal mode returns mu, logvar)
        x = torch.randn(BATCH, 10)
        mu_full, _ = enc_full(x)
        mu_selected = mu_full[:, populated]
        # trimmed forward
        encoder.set_mask(mask)
        mu_trimmed = encoder(x)
        torch.testing.assert_close(mu_trimmed, mu_selected, atol=1e-5, rtol=1e-5)

    def test_numerical_equivalence_with_params_controls(self, encoder_with_params_controls):
        enc = encoder_with_params_controls
        populated = [0, 4, 6]
        mask = _make_mask(8, populated)
        enc_full = copy.deepcopy(enc)
        x = torch.randn(BATCH, 10)
        params = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 5)
        mu_full, _ = enc_full(x, params=params, controls=controls)
        mu_selected = mu_full[:, populated]
        enc.set_mask(mask)
        mu_trimmed = enc(x, params=params, controls=controls)
        torch.testing.assert_close(mu_trimmed, mu_selected, atol=1e-5, rtol=1e-5)

    def test_lat_dim_updated(self, encoder):
        populated = [2, 5]
        mask = _make_mask(8, populated)
        encoder.set_mask(mask)
        assert encoder.lat_dim == 2

    def test_last_layer_shapes(self, encoder):
        populated = [0, 3, 4]
        mask = _make_mask(8, populated)
        encoder.set_mask(mask)
        last = encoder.net[-1]
        assert last.weight.shape[0] == 3
        assert last.bias.shape[0] == 3


# ─── LatentODEFunc (nonlinear) ────────────────────────────────────────────────


class TestLatentODEFuncNonlinearSetMask:
    """Tests for LatentODEFunc.set_mask — nonlinear case."""

    @pytest.fixture()
    def ode_func(self):
        torch.manual_seed(SEED)
        return LatentODEFunc(
            lat_state_mu_dim=8, lat_control_dim=4, lat_parameter_dim=3,
            hidden_dim=16, n_layers=3, activation=nn.ELU,
            initialization_ode=None, lat_ode_type='variance_constant',
            linear=False,
        )

    @pytest.fixture()
    def ode_func_no_controls_no_params(self):
        torch.manual_seed(SEED)
        return LatentODEFunc(
            lat_state_mu_dim=8, lat_control_dim=0, lat_parameter_dim=0,
            hidden_dim=16, n_layers=3, activation=nn.ELU,
            initialization_ode=None, lat_ode_type='variance_constant',
            linear=False,
        )

    def test_output_shape(self, ode_func):
        s_pop = [1, 3, 5]
        c_pop = [0, 2]
        p_pop = [1]
        ode_func.set_mask(
            _make_mask(8, s_pop),
            mask_controls=_make_mask(4, c_pop),
            mask_parameters=_make_mask(3, p_pop),
        )
        states = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 2)
        params = torch.randn(BATCH, 1)
        out = ode_func(states, lat_parameters=params, lat_controls=controls)
        assert out.shape == (BATCH, 3)

    def test_numerical_equivalence(self, ode_func):
        s_pop = [0, 2, 7]
        c_pop = [1, 3]
        p_pop = [0, 2]
        mask_s = _make_mask(8, s_pop)
        mask_c = _make_mask(4, c_pop)
        mask_p = _make_mask(3, p_pop)
        full = copy.deepcopy(ode_func)

        states_reduced = torch.randn(BATCH, 3)
        controls_reduced = torch.randn(BATCH, 2)
        params_reduced = torch.randn(BATCH, 2)

        states_full = _embed_reduced(8, s_pop, states_reduced)
        controls_full = _embed_reduced(4, c_pop, controls_reduced)
        params_full = _embed_reduced(3, p_pop, params_reduced)
        out_full = full(states_full, lat_parameters=params_full, lat_controls=controls_full)
        out_full_selected = out_full[:, s_pop]

        ode_func.set_mask(mask_s, mask_controls=mask_c, mask_parameters=mask_p)
        out_trimmed = ode_func(states_reduced, lat_parameters=params_reduced, lat_controls=controls_reduced)
        torch.testing.assert_close(out_trimmed, out_full_selected, atol=1e-5, rtol=1e-5)

    def test_numerical_equivalence_states_only(self, ode_func_no_controls_no_params):
        s_pop = [0, 4, 6]
        mask_s = _make_mask(8, s_pop)
        full = copy.deepcopy(ode_func_no_controls_no_params)

        states_reduced = torch.randn(BATCH, 3)
        states_full = _embed_reduced(8, s_pop, states_reduced)
        out_full = full(states_full)
        out_full_selected = out_full[:, s_pop]

        ode_func_no_controls_no_params.set_mask(mask_s)
        out_trimmed = ode_func_no_controls_no_params(states_reduced)
        torch.testing.assert_close(out_trimmed, out_full_selected, atol=1e-5, rtol=1e-5)

    def test_dims_updated(self, ode_func):
        ode_func.set_mask(
            _make_mask(8, [0, 2]),
            mask_controls=_make_mask(4, [1]),
            mask_parameters=_make_mask(3, [0, 2]),
        )
        assert ode_func.lat_state_mu_dim == 2
        assert ode_func.lat_state_dim == 2
        assert ode_func.lat_control_dim == 1
        assert ode_func.lat_parameter_dim == 2

    def test_first_last_layer_shapes(self, ode_func):
        s_pop = [0, 5]
        c_pop = [1]
        p_pop = [2]
        ode_func.set_mask(
            _make_mask(8, s_pop),
            mask_controls=_make_mask(4, c_pop),
            mask_parameters=_make_mask(3, p_pop),
        )
        first = ode_func.net[0]
        last = ode_func.net[-1]
        # input: 2 states + 1 param + 1 control = 4 columns
        assert first.weight.shape[1] == 4
        # output: 2 state rows
        assert last.weight.shape[0] == 2
        assert last.bias.shape[0] == 2


# ─── LatentODEFunc (linear, no params) ───────────────────────────────────────


class TestLatentODEFuncLinearSetMask:
    """Tests for LatentODEFunc.set_mask — linear SSM case (no params)."""

    @pytest.fixture()
    def ode_func_linear(self):
        torch.manual_seed(SEED)
        return LatentODEFunc(
            lat_state_mu_dim=6, lat_control_dim=4, lat_parameter_dim=0,
            hidden_dim=16, n_layers=3, activation=nn.ELU,
            initialization_ode=None, lat_ode_type='variance_constant',
            linear=True,
        )

    @pytest.fixture()
    def ode_func_linear_no_ctrl(self):
        torch.manual_seed(SEED)
        return LatentODEFunc(
            lat_state_mu_dim=6, lat_control_dim=0, lat_parameter_dim=0,
            hidden_dim=16, n_layers=3, activation=nn.ELU,
            initialization_ode=None, lat_ode_type='variance_constant',
            linear=True,
        )

    def test_output_shape(self, ode_func_linear):
        s_pop = [0, 3, 5]
        c_pop = [1, 2]
        ode_func_linear.set_mask(_make_mask(6, s_pop), mask_controls=_make_mask(4, c_pop))
        states = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 2)
        out = ode_func_linear(states, lat_controls=controls)
        assert out.shape == (BATCH, 3)

    def test_numerical_equivalence(self, ode_func_linear):
        s_pop = [1, 2, 4]
        c_pop = [0, 3]
        mask_s = _make_mask(6, s_pop)
        mask_c = _make_mask(4, c_pop)
        full = copy.deepcopy(ode_func_linear)

        states_r = torch.randn(BATCH, 3)
        controls_r = torch.randn(BATCH, 2)
        states_f = _embed_reduced(6, s_pop, states_r)
        controls_f = _embed_reduced(4, c_pop, controls_r)
        out_full = full(states_f, lat_controls=controls_f)
        out_selected = out_full[:, s_pop]

        ode_func_linear.set_mask(mask_s, mask_controls=mask_c)
        out_trimmed = ode_func_linear(states_r, lat_controls=controls_r)
        torch.testing.assert_close(out_trimmed, out_selected, atol=1e-5, rtol=1e-5)

    def test_numerical_equivalence_no_controls(self, ode_func_linear_no_ctrl):
        s_pop = [0, 3, 5]
        mask_s = _make_mask(6, s_pop)
        full = copy.deepcopy(ode_func_linear_no_ctrl)

        states_r = torch.randn(BATCH, 3)
        states_f = _embed_reduced(6, s_pop, states_r)
        out_full = full(states_f)
        out_selected = out_full[:, s_pop]

        ode_func_linear_no_ctrl.set_mask(mask_s)
        out_trimmed = ode_func_linear_no_ctrl(states_r)
        torch.testing.assert_close(out_trimmed, out_selected, atol=1e-5, rtol=1e-5)

    def test_A_B_shapes(self, ode_func_linear):
        s_pop = [0, 4]
        c_pop = [2]
        ode_func_linear.set_mask(_make_mask(6, s_pop), mask_controls=_make_mask(4, c_pop))
        assert ode_func_linear.A.weight.shape == (2, 2)
        assert ode_func_linear.B.weight.shape == (2, 1)


# ─── Decoder (nonlinear) ─────────────────────────────────────────────────────


class TestDecoderNonlinearSetMask:
    """Tests for Decoder.set_mask — nonlinear case."""

    @pytest.fixture()
    def decoder(self):
        torch.manual_seed(SEED)
        return Decoder(
            lat_state_mu_dim=8, lat_control_dim=4, lat_parameter_dim=3,
            state_dim=5, outputs_dim=3, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None, linear=False,
        )

    @pytest.fixture()
    def decoder_with_feedthrough(self):
        torch.manual_seed(SEED)
        return Decoder(
            lat_state_mu_dim=8, lat_control_dim=4, lat_parameter_dim=0,
            state_dim=5, outputs_dim=3, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None, linear=False,
            feedthrough_controls_dim=2,
        )

    def test_output_shape(self, decoder):
        s_pop = [0, 3, 7]
        c_pop = [1, 2]
        p_pop = [0]
        decoder.set_mask(
            _make_mask(8, s_pop),
            mask_controls=_make_mask(4, c_pop),
            mask_parameters=_make_mask(3, p_pop),
        )
        states = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 2)
        params = torch.randn(BATCH, 1)
        out = decoder(states, lat_parameters=params, lat_controls=controls)
        # output dim = (state_dim + outputs_dim) * 2 = (5+3)*2 = 16
        assert out.shape == (BATCH, 16)

    def test_numerical_equivalence(self, decoder):
        s_pop = [1, 5, 6]
        c_pop = [0, 3]
        p_pop = [2]
        mask_s = _make_mask(8, s_pop)
        mask_c = _make_mask(4, c_pop)
        mask_p = _make_mask(3, p_pop)
        full = copy.deepcopy(decoder)

        sr = torch.randn(BATCH, 3)
        cr = torch.randn(BATCH, 2)
        pr = torch.randn(BATCH, 1)
        sf = _embed_reduced(8, s_pop, sr)
        cf = _embed_reduced(4, c_pop, cr)
        pf = _embed_reduced(3, p_pop, pr)
        out_full = full(sf, lat_parameters=pf, lat_controls=cf)
        decoder.set_mask(mask_s, mask_controls=mask_c, mask_parameters=mask_p)
        out_trimmed = decoder(sr, lat_parameters=pr, lat_controls=cr)
        torch.testing.assert_close(out_trimmed, out_full, atol=1e-5, rtol=1e-5)

    def test_numerical_equivalence_with_feedthrough(self, decoder_with_feedthrough):
        s_pop = [0, 2, 4]
        c_pop = [1, 3]
        mask_s = _make_mask(8, s_pop)
        mask_c = _make_mask(4, c_pop)
        full = copy.deepcopy(decoder_with_feedthrough)

        sr = torch.randn(BATCH, 3)
        cr = torch.randn(BATCH, 2)
        ft = torch.randn(BATCH, 2)
        sf = _embed_reduced(8, s_pop, sr)
        cf = _embed_reduced(4, c_pop, cr)
        out_full = full(sf, lat_controls=cf, feedthrough_controls=ft)
        decoder_with_feedthrough.set_mask(mask_s, mask_controls=mask_c)
        out_trimmed = decoder_with_feedthrough(sr, lat_controls=cr, feedthrough_controls=ft)
        torch.testing.assert_close(out_trimmed, out_full, atol=1e-5, rtol=1e-5)

    def test_first_layer_shape(self, decoder):
        s_pop = [0, 7]
        c_pop = [2]
        p_pop = [0, 1]
        decoder.set_mask(
            _make_mask(8, s_pop),
            mask_controls=_make_mask(4, c_pop),
            mask_parameters=_make_mask(3, p_pop),
        )
        first = decoder.net[0]
        # 2 states + 2 params + 1 control = 5
        assert first.weight.shape[1] == 5

    def test_dims_updated(self, decoder):
        decoder.set_mask(
            _make_mask(8, [0, 2]),
            mask_controls=_make_mask(4, [1]),
            mask_parameters=_make_mask(3, [0]),
        )
        assert decoder.lat_state_mu_dim == 2
        assert decoder.lat_control_dim == 1
        assert decoder.lat_parameter_dim == 1


# ─── Decoder (linear, no params) ─────────────────────────────────────────────


class TestDecoderLinearSetMask:
    """Tests for Decoder.set_mask — linear case (no params)."""

    @pytest.fixture()
    def decoder_linear(self):
        torch.manual_seed(SEED)
        return Decoder(
            lat_state_mu_dim=6, lat_control_dim=4, lat_parameter_dim=0,
            state_dim=5, outputs_dim=3, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None, linear=True,
        )

    def test_output_shape(self, decoder_linear):
        s_pop = [0, 2, 5]
        c_pop = [1, 3]
        decoder_linear.set_mask(_make_mask(6, s_pop), mask_controls=_make_mask(4, c_pop))
        states = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 2)
        out = decoder_linear(states, lat_controls=controls)
        assert out.shape == (BATCH, 16)

    def test_numerical_equivalence(self, decoder_linear):
        s_pop = [1, 3, 4]
        c_pop = [0, 2]
        mask_s = _make_mask(6, s_pop)
        mask_c = _make_mask(4, c_pop)
        full = copy.deepcopy(decoder_linear)

        sr = torch.randn(BATCH, 3)
        cr = torch.randn(BATCH, 2)
        sf = _embed_reduced(6, s_pop, sr)
        cf = _embed_reduced(4, c_pop, cr)
        out_full = full(sf, lat_controls=cf)
        decoder_linear.set_mask(mask_s, mask_controls=mask_c)
        out_trimmed = decoder_linear(sr, lat_controls=cr)
        torch.testing.assert_close(out_trimmed, out_full, atol=1e-5, rtol=1e-5)

    def test_C_D_shapes(self, decoder_linear):
        s_pop = [0, 5]
        c_pop = [2]
        decoder_linear.set_mask(_make_mask(6, s_pop), mask_controls=_make_mask(4, c_pop))
        assert decoder_linear.C.weight.shape == (8, 2)  # out_dim=8, reduced_state=2
        assert decoder_linear.D.weight.shape == (8, 1)  # out_dim=8, reduced_ctrl=1


# ─── Decoder (linear, with params) ───────────────────────────────────────────


class TestDecoderLinearParamSetMask:
    """Tests for Decoder.set_mask — linear param-dependent case."""

    @pytest.fixture()
    def decoder_lin_param(self):
        torch.manual_seed(SEED)
        return Decoder(
            lat_state_mu_dim=6, lat_control_dim=4, lat_parameter_dim=3,
            state_dim=5, outputs_dim=3, hidden_dim=16, n_layers=3,
            activation=nn.ELU, initialization=None, linear=True,
        )

    def test_output_shape(self, decoder_lin_param):
        s_pop = [0, 2, 5]
        c_pop = [1]
        p_pop = [0, 2]
        decoder_lin_param.set_mask(
            _make_mask(6, s_pop),
            mask_controls=_make_mask(4, c_pop),
            mask_parameters=_make_mask(3, p_pop),
        )
        states = torch.randn(BATCH, 3)
        controls = torch.randn(BATCH, 1)
        params = torch.randn(BATCH, 2)
        out = decoder_lin_param(states, lat_parameters=params, lat_controls=controls)
        assert out.shape == (BATCH, 16)

    def test_numerical_equivalence(self, decoder_lin_param):
        s_pop = [1, 4]
        c_pop = [0, 3]
        p_pop = [2]
        mask_s = _make_mask(6, s_pop)
        mask_c = _make_mask(4, c_pop)
        mask_p = _make_mask(3, p_pop)
        full = copy.deepcopy(decoder_lin_param)

        sr = torch.randn(BATCH, 2)
        cr = torch.randn(BATCH, 2)
        pr = torch.randn(BATCH, 1)
        sf = _embed_reduced(6, s_pop, sr)
        cf = _embed_reduced(4, c_pop, cr)
        pf = _embed_reduced(3, p_pop, pr)
        out_full = full(sf, lat_parameters=pf, lat_controls=cf)
        decoder_lin_param.set_mask(mask_s, mask_controls=mask_c, mask_parameters=mask_p)
        out_trimmed = decoder_lin_param(sr, lat_parameters=pr, lat_controls=cr)
        torch.testing.assert_close(out_trimmed, out_full, atol=1e-4, rtol=1e-4)

    def test_C_and_constant_from_param_shape(self, decoder_lin_param):
        s_pop = [0, 3]
        decoder_lin_param.set_mask(_make_mask(6, s_pop))
        # out_dim = 8, reduced_state = 2 → weight rows = 8*2 + 8 = 24
        w = decoder_lin_param.C_and_constant_from_param.weight
        b = decoder_lin_param.C_and_constant_from_param.bias
        assert w.shape[0] == 8 * 2 + 8  # 24
        assert b.shape[0] == 8 * 2 + 8  # 24
