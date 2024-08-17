# GradientNormRegulator

## Overview

The `GradientNormRegulator` is a Python class that implements a Proportional-Integral-Derivative (PID) controller to dynamically adjust the learning rate during neural network training. It aims to maintain a target gradient norm, which can help stabilize training and potentially improve model performance.

## Features

- Dynamic learning rate adjustment based on gradient norm
- Auto-tuning of PID coefficients using first-epoch training data
- Support for both grid search and random search in auto-tuning
- Easy integration with TensorFlow/Keras training loops

## Installation

To use the `GradientNormRegulator`, you need to have TensorFlow and NumPy installed. You can install these dependencies using pip:

```bash
pip install tensorflow numpy
```

Then, simply copy the `GradientNormRegulator` class into your project.

## Usage

### Basic Usage

1. Import the necessary libraries and the `GradientNormRegulator` class:

```python
import tensorflow as tf
from gradient_norm_regulator import GradientNormRegulator
```

2. Create an instance of the `GradientNormRegulator`:

```python
regulator = GradientNormRegulator(target_grad_norm=1.0, min_lr=1e-6, max_lr=1e-2)
```

3. Auto-tune the PID coefficients using the first epoch of your training data:

```python
best_params = regulator.auto_tune(model, train_dataset, initial_lr=1e-3)
```

4. Use the regulator during training by adding its callback to your model's `fit` method:

```python
lr_callback = regulator.get_lr_callback()
model.fit(train_dataset, epochs=num_epochs, callbacks=[lr_callback])
```

### Advanced Usage

You can customize the auto-tuning process by specifying the search type and search space:

```python
best_params = regulator.auto_tune(
    model, 
    train_dataset, 
    initial_lr=1e-3,
    search_type='random',
    search_space={
        'Kp': (0.1, 5.0),
        'Ki': (0.0, 0.5),
        'Kd': (0.0, 0.5)
    },
    n_trials=100
)
```

## API Reference

### `GradientNormRegulator`

#### `__init__(self, Kp=1.0, Ki=0.0, Kd=0.0, target_grad_norm=1.0, min_lr=1e-6, max_lr=1e-2)`

Initializes the GradientNormRegulator.

- `Kp` (float): Proportional gain
- `Ki` (float): Integral gain
- `Kd` (float): Derivative gain
- `target_grad_norm` (float): The target gradient norm to maintain
- `min_lr` (float): Minimum learning rate
- `max_lr` (float): Maximum learning rate

#### `update(self, current_grad_norm)`

Updates the learning rate based on the current gradient norm.

- `current_grad_norm` (float): The current gradient norm

Returns:
- `float`: The adjusted learning rate

#### `reset(self)`

Resets the regulator's internal state.

#### `auto_tune(self, model, train_dataset, initial_lr, search_type='grid', search_space=None, n_trials=50)`

Auto-tunes the PID coefficients using the first epoch of training data.

- `model`: The Keras model to train
- `train_dataset`: The training dataset
- `initial_lr` (float): Initial learning rate
- `search_type` (str): 'grid' for grid search, 'random' for random search
- `search_space` (dict): Search space for PID coefficients
- `n_trials` (int): Number of trials for random search

Returns:
- `dict`: The best found PID coefficients

#### `get_lr_callback(self)`

Creates a Keras callback for dynamic learning rate adjustment.

Returns:
- `tf.keras.callbacks.Callback`: A callback to use during model training

## Examples

### Basic Training Loop

```python
import tensorflow as tf
from gradient_norm_regulator import GradientNormRegulator

# Prepare your model and dataset
model = tf.keras.Sequential([...])
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# Initialize and auto-tune the regulator
regulator = GradientNormRegulator(target_grad_norm=1.0)
best_params = regulator.auto_tune(model, train_dataset, initial_lr=1e-3)

print(f"Best PID parameters: {best_params}")

# Train the model with the regulator
lr_callback = regulator.get_lr_callback()
model.fit(train_dataset, epochs=10, callbacks=[lr_callback])
```

### Custom Training Loop

```python
import tensorflow as tf
from gradient_norm_regulator import GradientNormRegulator

# Prepare your model and dataset
model = tf.keras.Sequential([...])
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# Initialize and auto-tune the regulator
regulator = GradientNormRegulator(target_grad_norm=1.0)
best_params = regulator.auto_tune(model, train_dataset, initial_lr=1e-3)

# Custom training loop
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm(gradients)
    lr = regulator.update(grad_norm)
    optimizer.learning_rate.assign(lr)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, grad_norm

optimizer = tf.keras.optimizers.Adam()

for epoch in range(10):
    for inputs, labels in train_dataset:
        loss, grad_norm = train_step(inputs, labels)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}, Grad Norm: {grad_norm.numpy()}")
```

## Contributing

Contributions to improve the `GradientNormRegulator` are welcome. Please feel free to submit issues or pull requests on the GitHub repository.

## License

GradientNormRegulator is licensed under the Apache License, Version 2.0. See the LICENSE file for the full license text.
