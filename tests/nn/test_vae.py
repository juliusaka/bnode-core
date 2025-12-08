import hydra
import shutil
import sys
import os
from pathlib import Path
from bnode_core.nn.vae.vae_train_test import main as vae_main
from bnode_core.config import get_config_store

def vae_training(test_case: str, overrides: list[str] = [],):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    cs = get_config_store()
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    test_dir = Path('./tests/_results/nn/vae') / ('test_' + test_case)
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True) 
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=train_test_vae',
                f"hydra.run.dir={str(test_dir.absolute())}"
                ]
    sys.argv += overrides
    vae_main()
    sys.argv = orig_argv
    return test_dir

def test_vae():
    """Test basic VAE training with default configuration."""
    vae_training("test_vae", overrides=[
                # "nn_model.training.max_epochs=2",
                 ]
                 )

def test_vae_standard_mode():
    """Test standard VAE mode (encoder -> latent -> decoder)."""
    test_dir = vae_training("vae_standard", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.network.params_to_decoder=false",
        "nn_model.network.feed_forward_nn=false",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_pels_mode():
    """Test PELS-VAE mode (parameters concatenated to decoder input)."""
    test_dir = vae_training("vae_pels", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.network.params_to_decoder=true",
        "nn_model.network.feed_forward_nn=false",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_feedforward_mode():
    """Test feed-forward NN mode (parameters -> decoder, no latent space)."""
    test_dir = vae_training("vae_feedforward", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.network.feed_forward_nn=true",
        "nn_model.network.params_to_decoder=true",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_from_encoder():
    """Test VAE with testing from encoder (not regressor)."""
    test_dir = vae_training("vae_from_encoder", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.training.test_from_regressor=false",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_from_regressor():
    """Test VAE with testing from regressor (parameter-based prediction)."""
    test_dir = vae_training("vae_from_regressor", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.training.test_from_regressor=true",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_multi_pass():
    """Test VAE with multiple stochastic forward passes for prediction."""
    test_dir = vae_training("vae_multi_pass", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.training.n_passes_train=3",
        "nn_model.training.n_passes_test=5",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_with_capacity():
    """Test VAE with capacity scheduling for controlled KL divergence."""
    test_dir = vae_training("vae_capacity", overrides=[
        "nn_model.training.max_epochs=3",
        "nn_model.training.use_capacity=true",
        "nn_model.training.capacity_start=0.0",
        "nn_model.training.capacity_max=10.0",
        "nn_model.training.capacity_increment=2.0",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()

def test_vae_zero_eps():
    """Test VAE with deterministic testing (zero epsilon, no sampling variance)."""
    test_dir = vae_training("vae_zero_eps", overrides=[
        "nn_model.training.max_epochs=2",
        "nn_model.training.test_with_zero_eps=true",
    ])
    assert test_dir.exists()
    assert (test_dir / "model.pt").exists()