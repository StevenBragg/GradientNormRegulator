import pytest
import numpy as np
import tensorflow as tf
from gradientnormregulator import GradientNormRegulator

def test_initialization():
    regulator = GradientNormRegulator()
    assert regulator.Kp == 5.0
    assert regulator.Ki == 1.0
    assert regulator.Kd == 0.5
    assert regulator.target_grad_norm == 1.0

def test_update():
    regulator = GradientNormRegulator(Kp=1.0, Ki=0.1, Kd=0.01, target_grad_norm=1.0, min_lr=1e-4, max_lr=1e-2)
    lr = regulator.update(0.5)  # Grad norm is less than target
    assert 1e-4 <= lr <= 1e-2
    
    lr = regulator.update(1.5)  # Grad norm is greater than target
    assert 1e-4 <= lr <= 1e-2

def test_reset():
    regulator = GradientNormRegulator()
    regulator.update(0.5)
    regulator.reset()
    assert regulator.integral == 0.0
    assert regulator.prev_error is None
    assert regulator.step_count == 0
    assert len(regulator.history) == 0

def test_warmup():
    regulator = GradientNormRegulator(warmup_steps=10, min_lr=1e-4, max_lr=1e-2)
    lrs = [regulator.update(1.0) for _ in range(15)]
    assert lrs[0] == 1e-4
    assert lrs[9] == 1e-2
    assert all(1e-4 <= lr <= 1e-2 for lr in lrs)

def test_cooldown():
    regulator = GradientNormRegulator(warmup_steps=5, cooldown_steps=5, min_lr=1e-4, max_lr=1e-2)
    _ = [regulator.update(1.0) for _ in range(5)]  # Warmup
    lrs = [regulator.update(1.0) for _ in range(5)]  # Cooldown
    assert lrs[0] == 1e-2
    assert lrs[-1] == 1e-4
    assert all(1e-4 <= lr <= 1e-2 for lr in lrs)

def test_invalid_inputs():
    with pytest.raises(ValueError):
        GradientNormRegulator(Kp=-1.0)
    
    with pytest.raises(ValueError):
        GradientNormRegulator(min_lr=1.0, max_lr=0.1)

    regulator = GradientNormRegulator()
    with pytest.raises(ValueError):
        regulator.update(-1.0)

@pytest.mark.parametrize("Kp,Ki,Kd,target_grad_norm", [
    (1.0, 0.1, 0.01, 0.5),
    (2.0, 0.2, 0.02, 1.5),
    (0.5, 0.05, 0.005, 2.0),
])
def test_auto_tune(Kp, Ki, Kd, target_grad_norm):
    x = np.random.random((100, 5))
    y = np.random.randint(2, size=(100, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    search_space = {
        'Kp': (Kp/2, Kp*2),
        'Ki': (Ki/2, Ki*2),
        'Kd': (Kd/2, Kd*2),
        'target_grad_norm': (target_grad_norm/2, target_grad_norm*2)
    }
    
    regulator = GradientNormRegulator.auto_tune(model, dataset, epochs=1, search_space=search_space, n_trials=5)
    
    assert Kp/2 <= regulator.Kp <= Kp*2
    assert Ki/2 <= regulator.Ki <= Ki*2
    assert Kd/2 <= regulator.Kd <= Kd*2
    assert target_grad_norm/2 <= regulator.target_grad_norm <= target_grad_norm*2

if __name__ == "__main__":
    pytest.main()
