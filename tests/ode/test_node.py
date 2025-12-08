from test_bnode import ode_training, ode_training_initial_states, ode_training_params

def test_node_training():
    """Test basic BNODE training with controls and parameters."""
    ode_training('bnode_node_training', overrides=[
        'nn_model=node_pytest'
    ])

def test_node_training_with_pretrain():
    """Test BNODE training with pretraining phase."""
    ode_training('bnode_node_training_pretrain', overrides=[
        'nn_model=node_pytest',
        'nn_model.training.pre_train=true',
    ])

def test_node_training_with_parameters():
    """Test BNODE training with parameters."""
    ode_training_params('bnode_node_training_params', overrides=[
        'nn_model=node_pytest',
    ])

def test_node_training_with_parameters_and_pretrain():
    """Test BNODE training with parameters and pretraining phase."""
    ode_training_params('bnode_node_training_params_pretrain', overrides=[
        'nn_model=node_pytest',
        'nn_model.training.pre_train=true',
    ])

def test_node_trainin_initial_states():
    """Test BNODE training with only initial states as parameters."""
    ode_training_initial_states('bnode_node_training_initial_states', overrides=[
        'nn_model=node_pytest',
    ])

def test_node_training_initial_states_with_pretrain():
    """Test BNODE training with only initial states as parameters and pretraining phase."""
    ode_training_initial_states('bnode_node_training_initial_states_pretrain', overrides=[
        'nn_model=node_pytest',
        'nn_model.training.pre_train=true',
    ])