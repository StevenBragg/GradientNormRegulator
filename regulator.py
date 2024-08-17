import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Union, Tuple

class GradientNormRegulator:
    """
    A Gradient Norm Regulator that dynamically adjusts the learning rate based on the gradient norm.
    This regulator uses a Proportional-Integral-Derivative (PID) control method to keep the gradient norm close to a desired target.
    """

    def __init__(self, Kp: float = 5.0, Ki: float = 1.0, Kd: float = 0.5, 
                 target_grad_norm: float = 1.0, min_lr: float = 1e-6, max_lr: float = 1e-2,
                 warmup_steps: int = 0, cooldown_steps: int = 0):
        """Initialize the GradientNormRegulator."""
        self._validate_inputs(Kp, Ki, Kd, target_grad_norm, min_lr, max_lr, warmup_steps, cooldown_steps)
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target_grad_norm = target_grad_norm
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps

        self.integral = 0.0
        self.prev_error = None
        self.step_count = 0
        self.history = []

    def _validate_inputs(self, Kp, Ki, Kd, target_grad_norm, min_lr, max_lr, warmup_steps, cooldown_steps):
        """Validate input parameters."""
        if not all(isinstance(arg, (float, int)) and arg >= 0 for arg in [Kp, Ki, Kd, target_grad_norm, min_lr, max_lr]):
            raise ValueError("Kp, Ki, Kd, target_grad_norm, min_lr, and max_lr must be non-negative numbers.")
        if not all(isinstance(arg, int) and arg >= 0 for arg in [warmup_steps, cooldown_steps]):
            raise ValueError("warmup_steps and cooldown_steps must be non-negative integers.")
        if min_lr >= max_lr:
            raise ValueError("min_lr must be less than max_lr.")

    def update(self, current_grad_norm: float) -> float:
        """Update the learning rate based on the current gradient norm."""
        if not isinstance(current_grad_norm, (float, int)) or current_grad_norm < 0:
            raise ValueError("current_grad_norm must be a non-negative number.")

        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            return self._warmup_lr()
        elif self.step_count > self.warmup_steps and self.step_count <= (self.warmup_steps + self.cooldown_steps):
            return self._cooldown_lr()

        error = self.target_grad_norm - current_grad_norm
        self.integral += error
        derivative = 0.0 if self.prev_error is None else error - self.prev_error

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error

        adjusted_lr = np.clip(output, self.min_lr, self.max_lr)
        self.history.append((self.step_count, current_grad_norm, adjusted_lr))
        return adjusted_lr

    def _warmup_lr(self) -> float:
        """Calculate learning rate during warmup period."""
        return self.min_lr + (self.max_lr - self.min_lr) * (self.step_count / self.warmup_steps)

    def _cooldown_lr(self) -> float:
        """Calculate learning rate during cooldown period."""
        cooldown_progress = (self.step_count - self.warmup_steps) / self.cooldown_steps
        return self.max_lr - (self.max_lr - self.min_lr) * cooldown_progress

    def reset(self):
        """Reset the regulator's internal state."""
        self.integral = 0.0
        self.prev_error = None
        self.step_count = 0
        self.history.clear()

    def get_lr_callback(self) -> tf.keras.callbacks.LambdaCallback:
        """Get a Keras callback for dynamic learning rate adjustment."""
        def on_batch_end(batch, logs):
            if logs is None:
                logs = {}
            grad_norm = logs.get('grad_norm', None)
            if grad_norm is not None:
                new_lr = self.update(grad_norm)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

        return tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)

    def plot_history(self):
        """Plot the history of gradient norms and learning rates."""
        import matplotlib.pyplot as plt

        if not self.history:
            print("No history to plot. Train the model first.")
            return

        steps, grad_norms, lrs = zip(*self.history)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        ax1.plot(steps, grad_norms, label='Gradient Norm')
        ax1.axhline(y=self.target_grad_norm, color='r', linestyle='--', label='Target Norm')
        ax1.set_ylabel('Gradient Norm')
        ax1.legend()

        ax2.plot(steps, lrs, label='Learning Rate')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    @classmethod
    def auto_tune(cls, model: tf.keras.Model, train_dataset: tf.data.Dataset, 
                  epochs: int = 1, search_space: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None, 
                  n_trials: int = 20) -> 'GradientNormRegulator':
        """Auto-tune the PID coefficients using Bayesian optimization."""
        from sklearn.model_selection import train_test_split
        from skopt import gp_minimize
        from skopt.space import Real

        if search_space is None:
            search_space = {
                'Kp': (0.1, 10.0),
                'Ki': (0.0, 1.0),
                'Kd': (0.0, 1.0),
                'target_grad_norm': (0.1, 2.0)
            }

        space = [Real(low, high, name=name) for name, (low, high) in search_space.items()]

        def objective(params):
            regulator = cls(**dict(zip(search_space.keys(), params)))
            callback = regulator.get_lr_callback()
            
            history = model.fit(train_dataset, epochs=epochs, callbacks=[callback], verbose=0)
            return -np.mean(history.history['val_accuracy'])

        result = gp_minimize(objective, space, n_calls=n_trials, random_state=42)

        best_params = dict(zip(search_space.keys(), result.x))
        return cls(**best_params)
