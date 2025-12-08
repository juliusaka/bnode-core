from bnode_core import config
import pytest
from omegaconf import OmegaConf

def test_config_store():
    cs = config.get_config_store()

def test_config_store_singleton():
    # ConfigStore is a singleton
    cs = config.get_config_store()
    cs2 = config.get_config_store()
    assert cs is cs2

def test_solver_sequence_length_computed():
    s = config.SolverClass(simulationStartTime=0.0, simulationEndTime=1.0, timestep=0.2)
    # ceil((1-0)/0.2) + 1 = ceil(5) + 1 = 5
    assert s.sequence_length == 6

def test_pels_vae_activation_validation_failure():
    with pytest.raises(ValueError):
        config.pels_vae_network_class(activation='not.a.module')

def test_lat_ode_type_invalid():
    with pytest.raises(ValueError):
        config.latent_ode_network_class(lat_ode_type='invalid_type')

def test_convert_cfg_to_dataclass():
    dc = OmegaConf.create({'x': 1, 'y': {'z': 2}})
    obj = config.convert_cfg_to_dataclass(dc)
    assert isinstance(obj, dict)
    assert obj['x'] == 1
    assert obj['y']['z'] == 2

def test_bnode_linear_mode():
    cls = config.latent_ode_network_class(linear_mode='mpc_mode')
    assert cls.state_encoder_linear is False
    assert cls.control_encoder_linear is True
    assert cls.parameter_encoder_linear is True
    assert cls.ode_linear is True
    assert cls.decoder_linear is True

def test_bnode_linear_mode_for_controls():
    cls = config.latent_ode_network_class(linear_mode='mpc_mode_for_controls')
    assert cls.state_encoder_linear is False
    assert cls.control_encoder_linear is True
    assert cls.parameter_encoder_linear is False
    assert cls.ode_linear is True
    assert cls.decoder_linear is True

def test_bnode_deep_koopman_mode():
    cls = config.latent_ode_network_class(linear_mode='deep_koopman')
    assert cls.state_encoder_linear is False
    assert cls.control_encoder_linear is False
    assert cls.parameter_encoder_linear is False
    assert cls.ode_linear is True
    assert cls.decoder_linear is False
    
def test_rawdata_fmu_path_required_when_not_external():
    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=False, fmuPath=None)

def test_rawdata_fmu_path_invalid_suffix(tmp_path):
    p = tmp_path / "model.txt"
    # existence doesn't matter, suffix check should raise
    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=True, fmuPath=str(p))

def test_rawdata_fmu_path_accepts_valid_suffix_even_if_nonexistent(tmp_path):
    p = tmp_path / "model.fmu"
    rd = config.RawDataClass(raw_data_from_external_source=True, fmuPath=str(p))
    assert isinstance(rd.fmuPath, str)
    assert rd.fmuPath == str(p.as_posix())

def test_rawdata_fmu_path_accepts_existing_file(tmp_path):
    p = tmp_path / "model.fmu"
    p.write_text("dummy")
    rd = config.RawDataClass(raw_data_from_external_source=True, fmuPath=str(p))
    assert rd.fmuPath == str(p.as_posix())

def test_parameters_conversion_and_defaults():
    rd = config.RawDataClass(raw_data_from_external_source=True, parameters={'p1': 2})
    assert 'p1' in rd.parameters
    assert rd.parameters['p1'][2] == 2
    assert rd.parameters['p1'][0] == pytest.approx(0.2 * 2)
    assert rd.parameters['p1'][1] == pytest.approx(5.0 * 2)

def test_parameters_invalid_length_raises():
    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=True, parameters={'p1': [1, 2]})

def test_controls_defaults_and_validation():
    rd = config.RawDataClass(raw_data_from_external_source=True, controls={'u': None})
    assert rd.controls['u'] == [rd.controls_default_lower_value, rd.controls_default_upper_value]

    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=True, controls={'u': [5, 0]})

def test_states_defaults_and_validation():
    rd = config.RawDataClass(raw_data_from_external_source=True, states={'s': None})
    assert rd.states['s'] == [rd.states_default_lower_value, rd.states_default_upper_value]

    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=True, states={'s': [2, 1]})

def test_states_length_validation():
    with pytest.raises(ValueError):
        config.RawDataClass(raw_data_from_external_source=True, states={'s': [1, 2, 3]})

def test_controls_frequency_validation_requires_min_and_max():
    with pytest.raises(ValueError):
        config.RawDataClass(
            raw_data_from_external_source=True,
            controls_sampling_strategy='ROCS',
            controls_frequency_min_in_timesteps=None,
            controls_frequency_max_in_timesteps=None
        )

    with pytest.raises(ValueError):
        config.RawDataClass(
            raw_data_from_external_source=True,
            controls_sampling_strategy='ROCS',
            controls_frequency_min_in_timesteps=5,
            controls_frequency_max_in_timesteps=2
        )

def test_controls_file_path_validator(tmp_path):
    p = tmp_path / "controls.csv"
    p.write_text("time,ctrl\n0,1\n")
    rd = config.RawDataClass(
        raw_data_from_external_source=True,
        controls_sampling_strategy='file',
        controls_file_path=str(p)
    )
    assert rd.controls_file_path == str(p.as_posix())

    # when strategy is not file/constantInput, provided path should be ignored and result in None
    rd2 = config.RawDataClass(
        raw_data_from_external_source=True,
        controls_sampling_strategy='R',
        controls_file_path='some/nonexistent/path.csv'
    )
    assert rd2.controls_file_path is None