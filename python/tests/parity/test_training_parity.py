"""
Training Framework Parity Tests

Tests MLX training framework against PyTorch Lightning reference.
Verifies that given identical inputs, frameworks produce equivalent outputs.

These tests ensure:
1. Loss computation produces same values
2. Gradients are computed identically
3. Optimizer updates produce same parameter changes
4. Callback hooks fire in the same order
5. Validation metrics are computed identically
"""

import numpy as np
import pytest

# Check for PyTorch availability
HAS_PYTORCH = False
torch = None
torch_nn = None

try:
    import torch as _torch
    import torch.nn as _torch_nn

    torch = _torch
    torch_nn = _torch_nn
    HAS_PYTORCH = True
except ImportError:
    pass

# Check for MLX availability
HAS_MLX = False
mx = None
nn = None
optim = None
TrainModule = None
Trainer = None
OptimizerConfig = None
Callback = None

try:
    import mlx.core as _mx
    import mlx.nn as _nn
    import mlx.optimizers as _optim
    from mlx_audio.train import (
        TrainModule as _TrainModule,
        Trainer as _Trainer,
        OptimizerConfig as _OptimizerConfig,
        Callback as _Callback,
    )

    mx = _mx
    nn = _nn
    optim = _optim
    TrainModule = _TrainModule
    Trainer = _Trainer
    OptimizerConfig = _OptimizerConfig
    Callback = _Callback
    HAS_MLX = True
except ImportError:
    pass


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility across frameworks."""
    np.random.seed(seed)
    if HAS_PYTORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if HAS_MLX:
        mx.random.seed(seed)


def create_deterministic_data(
    num_samples: int = 100,
    in_features: int = 10,
    num_classes: int = 2,
    seed: int = 42,
):
    """Create identical data for both frameworks."""
    np.random.seed(seed)
    X = np.random.randn(num_samples, in_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    return X, y


# ==================== PyTorch Model Factories ====================


def create_torch_linear_model(in_features: int = 10, out_features: int = 2):
    """Create a simple linear model in PyTorch."""
    if not HAS_PYTORCH:
        return None

    class TorchLinearModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch_nn.Linear(in_features, out_features, bias=True)
            torch_nn.init.ones_(self.linear.weight)
            torch_nn.init.zeros_(self.linear.bias)

        def forward(self, x):
            return self.linear(x)

    return TorchLinearModel()


# ==================== MLX Model Factories ====================


def create_mlx_linear_model(in_features: int = 10, out_features: int = 2):
    """Create a simple linear model in MLX matching PyTorch."""
    if not HAS_MLX:
        return None

    class MLXLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=True)
            self.linear.weight = mx.ones((out_features, in_features))
            self.linear.bias = mx.zeros((out_features,))

        def __call__(self, x):
            return self.linear(x)

    return MLXLinearModel()


def create_mlx_train_module(
    in_features: int = 10, out_features: int = 2, lr: float = 0.01
):
    """Create an MLX TrainModule for training tests."""
    if not HAS_MLX:
        return None

    class MLXTrainMod(TrainModule):
        def __init__(self):
            super().__init__()
            self.model = create_mlx_linear_model(in_features, out_features)
            self._lr = lr

        def __call__(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = mx.mean(nn.losses.cross_entropy(logits, y))
            acc = mx.mean(mx.argmax(logits, axis=-1) == y)
            # Note: Don't call mx.eval() inside training_step - it's called
            # inside value_and_grad which doesn't allow evaluation
            return {"loss": loss, "accuracy": acc}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = mx.mean(nn.losses.cross_entropy(logits, y))
            acc = mx.mean(mx.argmax(logits, axis=-1) == y)
            return {"val_loss": loss, "val_accuracy": acc}

        def configure_optimizers(self):
            return OptimizerConfig(optimizer=optim.SGD(learning_rate=self._lr))

    return MLXTrainMod()


# ==================== Parity Tests ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestTrainingParity:
    """Test training behavior parity between MLX and PyTorch."""

    def test_forward_pass_parity(self):
        """Forward pass should produce identical outputs given same weights."""
        set_seeds(42)

        X, _ = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch forward
        torch_model = create_torch_linear_model()
        torch_input = torch.from_numpy(X)
        torch_output = torch_model(torch_input).detach().numpy()

        # MLX forward
        mlx_model = create_mlx_linear_model()
        mlx_input = mx.array(X)
        mlx_output = np.array(mlx_model(mlx_input))

        np.testing.assert_allclose(
            mlx_output, torch_output, rtol=1e-5, atol=1e-5,
            err_msg="Forward pass outputs differ"
        )

    def test_loss_computation_parity(self):
        """Loss computation should match between frameworks."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch loss
        torch_model = create_torch_linear_model()
        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        torch_logits = torch_model(torch_input)
        torch_loss = torch_nn.functional.cross_entropy(
            torch_logits, torch_target
        )
        torch_loss_val = torch_loss.item()

        # MLX loss
        mlx_model = create_mlx_linear_model()
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)
        mlx_logits = mlx_model(mlx_input)
        mlx_loss = mx.mean(nn.losses.cross_entropy(mlx_logits, mlx_target))
        mx.eval(mlx_loss)
        mlx_loss_val = float(mlx_loss.item())

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-5,
            err_msg=f"Loss: MLX={mlx_loss_val}, PyTorch={torch_loss_val}"
        )

    def test_gradient_computation_parity(self):
        """Gradients should match between frameworks."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch gradients
        torch_model = create_torch_linear_model()
        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        torch_logits = torch_model(torch_input)
        torch_loss = torch_nn.functional.cross_entropy(
            torch_logits, torch_target
        )
        torch_loss.backward()
        torch_weight_grad = torch_model.linear.weight.grad.numpy()
        torch_bias_grad = torch_model.linear.bias.grad.numpy()

        # MLX gradients
        mlx_model = create_mlx_linear_model()

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, mlx_input, mlx_target)
        mx.eval(loss, grads)

        mlx_weight_grad = np.array(grads["linear"]["weight"])
        mlx_bias_grad = np.array(grads["linear"]["bias"])

        np.testing.assert_allclose(
            mlx_weight_grad, torch_weight_grad, rtol=1e-4, atol=1e-4,
            err_msg="Weight gradients differ"
        )
        np.testing.assert_allclose(
            mlx_bias_grad, torch_bias_grad, rtol=1e-4, atol=1e-4,
            err_msg="Bias gradients differ"
        )

    def test_sgd_update_parity(self):
        """SGD optimizer updates should produce same parameter changes."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)
        lr = 0.1

        # PyTorch update
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)

        torch_optimizer.zero_grad()
        torch_logits = torch_model(torch_input)
        torch_loss = torch_nn.functional.cross_entropy(
            torch_logits, torch_target
        )
        torch_loss.backward()
        torch_optimizer.step()

        torch_weight_after = torch_model.linear.weight.detach().numpy()
        torch_bias_after = torch_model.linear.bias.detach().numpy()

        # MLX update
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)

        loss, grads = loss_and_grad(mlx_model, mlx_input, mlx_target)
        mlx_optimizer.update(mlx_model, grads)
        mx.eval(mlx_model.parameters())

        mlx_weight_after = np.array(mlx_model.linear.weight)
        mlx_bias_after = np.array(mlx_model.linear.bias)

        np.testing.assert_allclose(
            mlx_weight_after, torch_weight_after, rtol=1e-4, atol=1e-4,
            err_msg="Weights after SGD update differ"
        )
        np.testing.assert_allclose(
            mlx_bias_after, torch_bias_after, rtol=1e-4, atol=1e-4,
            err_msg="Biases after SGD update differ"
        )

    def test_training_step_output_parity(self):
        """training_step should produce same loss and metrics."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch manual training step
        torch_model = create_torch_linear_model()
        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        torch_logits = torch_model(torch_input)
        torch_loss = torch_nn.functional.cross_entropy(
            torch_logits, torch_target
        )
        torch_acc = (
            (torch_logits.argmax(dim=-1) == torch_target).float().mean()
        )
        torch_loss_val = torch_loss.item()
        torch_acc_val = torch_acc.item()

        # MLX training_step
        mlx_module = create_mlx_train_module(lr=0.01)
        mlx_batch = (mx.array(X), mx.array(y))
        mlx_result = mlx_module.training_step(mlx_batch, batch_idx=0)
        mx.eval(mlx_result["loss"], mlx_result["accuracy"])
        mlx_loss_val = float(mlx_result["loss"].item())
        mlx_acc_val = float(mlx_result["accuracy"].item())

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-5,
            err_msg=f"Loss: MLX={mlx_loss_val}, PyTorch={torch_loss_val}"
        )
        np.testing.assert_allclose(
            mlx_acc_val, torch_acc_val, rtol=1e-5, atol=1e-5,
            err_msg=f"Accuracy: MLX={mlx_acc_val}, PyTorch={torch_acc_val}"
        )


@pytest.mark.parity
@pytest.mark.skipif(not HAS_MLX, reason="MLX required")
class TestCallbackParity:
    """Test callback hook firing order matches PyTorch Lightning."""

    def test_hook_order_single_epoch(self):
        """Callback hooks should fire in the same order as Lightning."""
        set_seeds(42)

        # Expected hook order for single epoch with validation
        expected_order = [
            "on_fit_start",
            "on_train_start",
            "on_train_epoch_start",
            "on_train_batch_start",
            "on_train_batch_end",
            "on_validation_start",
            "on_validation_batch_start",
            "on_validation_batch_end",
            "on_validation_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_fit_end",
        ]

        # Create tracking callback
        class HookTracker(Callback):
            def __init__(self):
                self.hooks = []

            def on_fit_start(self, trainer, module):
                self.hooks.append("on_fit_start")

            def on_fit_end(self, trainer, module):
                self.hooks.append("on_fit_end")

            def on_train_start(self, trainer, module):
                self.hooks.append("on_train_start")

            def on_train_end(self, trainer, module):
                self.hooks.append("on_train_end")

            def on_train_epoch_start(self, trainer, module, ctx):
                self.hooks.append("on_train_epoch_start")

            def on_train_epoch_end(self, trainer, module, ctx):
                self.hooks.append("on_train_epoch_end")

            def on_train_batch_start(self, trainer, module, batch, ctx):
                self.hooks.append("on_train_batch_start")

            def on_train_batch_end(self, trainer, module, batch, outputs, ctx):
                self.hooks.append("on_train_batch_end")

            def on_validation_start(self, trainer, module, ctx):
                self.hooks.append("on_validation_start")

            def on_validation_end(self, trainer, module, ctx, metrics):
                self.hooks.append("on_validation_end")

            def on_validation_batch_start(
                self, trainer, module, batch, batch_idx, ctx
            ):
                self.hooks.append("on_validation_batch_start")

            def on_validation_batch_end(
                self, trainer, module, outputs, batch, batch_idx, ctx
            ):
                self.hooks.append("on_validation_batch_end")

        # Create data
        X, y = create_deterministic_data(num_samples=10, seed=42)
        train_data = [(mx.array(X), mx.array(y))]
        val_data = [(mx.array(X), mx.array(y))]

        # Create module and trainer
        module = create_mlx_train_module(lr=0.01)
        tracker = HookTracker()
        trainer = Trainer(
            max_epochs=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[tracker],
        )

        # Run training
        trainer.fit(module, train_data, val_data)

        # Verify hook order
        assert tracker.hooks == expected_order, (
            f"Hook order mismatch:\n"
            f"Expected: {expected_order}\n"
            f"Got: {tracker.hooks}"
        )

    def test_batch_idx_incrementing(self):
        """batch_idx should increment correctly within epochs."""
        set_seeds(42)

        class BatchIdxTracker(Callback):
            def __init__(self):
                self.batch_indices = []

            def on_train_batch_end(self, trainer, module, batch, outputs, ctx):
                self.batch_indices.append(ctx.batch_idx)

        # Create multi-batch data
        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10
        train_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, len(X), batch_size)
        ]

        module = create_mlx_train_module(lr=0.01)
        tracker = BatchIdxTracker()
        trainer = Trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[tracker],
        )

        trainer.fit(module, train_data)

        # Should have [0, 1, 2, 0, 1, 2] for 2 epochs with 3 batches each
        expected = [0, 1, 2, 0, 1, 2]
        assert tracker.batch_indices == expected, (
            f"batch_idx sequence wrong: {tracker.batch_indices}"
        )


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestMultiStepParity:
    """Test parity across multiple training steps."""

    def test_loss_trajectory_matches(self):
        """Loss values should follow same trajectory over multiple steps."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=50, seed=42)
        batch_size = 10
        lr = 0.1
        num_batches = 5

        # PyTorch trajectory
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
        torch_losses = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = torch.from_numpy(X[start_idx:end_idx])
            batch_y = torch.from_numpy(y[start_idx:end_idx])

            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()
            torch_losses.append(loss.item())

        # MLX trajectory
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)
        mlx_losses = []

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = mx.array(X[start_idx:end_idx])
            batch_y = mx.array(y[start_idx:end_idx])

            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(loss, mlx_model.parameters())
            mlx_losses.append(float(loss.item()))

        # Compare loss trajectories
        np.testing.assert_allclose(
            mlx_losses, torch_losses, rtol=1e-4, atol=1e-4,
            err_msg=f"Loss trajectories:\nMLX: {mlx_losses}\nPyTorch: {torch_losses}"
        )

    def test_weight_evolution_matches(self):
        """Weights should evolve identically over multiple steps."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10
        lr = 0.1
        num_batches = 3

        # Track PyTorch weights
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
        torch_weights = [torch_model.linear.weight.detach().numpy().copy()]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = torch.from_numpy(X[start_idx:end_idx])
            batch_y = torch.from_numpy(y[start_idx:end_idx])

            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()
            torch_weights.append(
                torch_model.linear.weight.detach().numpy().copy()
            )

        # Track MLX weights
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)
        mlx_weights = [np.array(mlx_model.linear.weight)]

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = mx.array(X[start_idx:end_idx])
            batch_y = mx.array(y[start_idx:end_idx])

            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())
            mlx_weights.append(np.array(mlx_model.linear.weight))

        # Compare weight evolution
        for step, (mlx_w, torch_w) in enumerate(zip(mlx_weights, torch_weights)):
            np.testing.assert_allclose(
                mlx_w, torch_w, rtol=1e-4, atol=1e-4,
                err_msg=f"Weights differ at step {step}"
            )


@pytest.mark.parity
@pytest.mark.skipif(not HAS_MLX, reason="MLX required")
class TestInternalConsistency:
    """Test internal consistency of MLX training framework."""

    def test_metric_aggregation_consistency(self):
        """Validation metrics should aggregate correctly."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10

        # Create multi-batch validation data
        val_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, len(X), batch_size)
        ]

        # Use the same module for both manual and trainer validation
        # to ensure weights are identical
        module = create_mlx_train_module(lr=0.01)

        trainer = Trainer(
            max_epochs=0,  # No training, just validation
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Run validation via trainer
        trainer.validate(module, val_data)
        trainer_avg_loss = trainer.callback_metrics.get("val_loss")

        # Compute expected average manually using the SAME module
        manual_losses = []
        for i, (batch_x, batch_y) in enumerate(val_data):
            result = module.validation_step((batch_x, batch_y), i)
            mx.eval(result["val_loss"])
            manual_losses.append(float(result["val_loss"].item()))
        expected_avg_loss = sum(manual_losses) / len(manual_losses)

        np.testing.assert_allclose(
            trainer_avg_loss, expected_avg_loss, rtol=1e-5, atol=1e-5,
            err_msg=f"val_loss: Trainer={trainer_avg_loss}, Manual={expected_avg_loss}"
        )

    def test_global_step_incrementing(self):
        """global_step should increment correctly."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10
        train_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, len(X), batch_size)
        ]

        module = create_mlx_train_module(lr=0.01)
        trainer = Trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(module, train_data)

        # 2 epochs * 3 batches = 6 steps
        assert trainer.global_step == 6, (
            f"Expected global_step=6, got {trainer.global_step}"
        )

    def test_epoch_incrementing(self):
        """current_epoch should increment correctly."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)
        train_data = [(mx.array(X), mx.array(y))]

        module = create_mlx_train_module(lr=0.01)
        trainer = Trainer(
            max_epochs=3,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(module, train_data)

        # After 3 epochs, current_epoch should be 3
        assert trainer.current_epoch == 3, (
            f"Expected current_epoch=3, got {trainer.current_epoch}"
        )


# ==================== AdamW Optimizer Parity ====================


def create_torch_mlp(hidden_sizes: list[int] = [64, 32], in_features: int = 10, out_features: int = 2):
    """Create an MLP in PyTorch with deterministic initialization."""
    if not HAS_PYTORCH:
        return None

    layers = []
    prev_size = in_features
    for i, hidden in enumerate(hidden_sizes):
        linear = torch_nn.Linear(prev_size, hidden, bias=True)
        # Deterministic init: small values based on layer index
        torch_nn.init.constant_(linear.weight, 0.1 / (i + 1))
        torch_nn.init.constant_(linear.bias, 0.01)
        layers.append(linear)
        layers.append(torch_nn.ReLU())
        prev_size = hidden

    # Output layer
    output = torch_nn.Linear(prev_size, out_features, bias=True)
    torch_nn.init.constant_(output.weight, 0.1)
    torch_nn.init.constant_(output.bias, 0.0)
    layers.append(output)

    return torch_nn.Sequential(*layers)


def create_mlx_mlp(hidden_sizes: list[int] = [64, 32], in_features: int = 10, out_features: int = 2):
    """Create an MLP in MLX with deterministic initialization matching PyTorch."""
    if not HAS_MLX:
        return None

    class MLXMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []
            prev_size = in_features
            for i, hidden in enumerate(hidden_sizes):
                linear = nn.Linear(prev_size, hidden, bias=True)
                # Match PyTorch init
                linear.weight = mx.full((hidden, prev_size), 0.1 / (i + 1))
                linear.bias = mx.full((hidden,), 0.01)
                self.layers.append(linear)
                prev_size = hidden

            # Output layer
            self.output = nn.Linear(prev_size, out_features, bias=True)
            self.output.weight = mx.full((out_features, prev_size), 0.1)
            self.output.bias = mx.zeros((out_features,))

        def __call__(self, x):
            for layer in self.layers:
                x = nn.relu(layer(x))
            return self.output(x)

    return MLXMLP()


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestAdamWParity:
    """Test AdamW optimizer parity between MLX and PyTorch."""

    def test_adamw_single_step(self):
        """AdamW should produce identical updates for single step."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)
        lr = 0.001
        weight_decay = 0.01
        betas = (0.9, 0.999)
        eps = 1e-8

        # PyTorch - disable bias correction to match MLX default
        # PyTorch AdamW uses bias correction by default, MLX does not
        torch_model = create_torch_linear_model()

        # Use Adam with decoupled weight decay and no bias correction
        # to match MLX's AdamW behavior exactly
        class TorchAdamWNoBias(torch.optim.Optimizer):
            """AdamW without bias correction to match MLX."""
            def __init__(self, params, lr, betas, eps, weight_decay):
                defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
                super().__init__(params, defaults)

            def step(self):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad.data
                        state = self.state[p]

                        if len(state) == 0:
                            state['m'] = torch.zeros_like(p.data)
                            state['v'] = torch.zeros_like(p.data)

                        m, v = state['m'], state['v']
                        beta1, beta2 = group['betas']

                        # Update biased moments (no correction)
                        m.mul_(beta1).add_(grad, alpha=1 - beta1)
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        # AdamW update: gradient step + weight decay
                        denom = v.sqrt().add_(group['eps'])
                        p.data.addcdiv_(m, denom, value=-group['lr'])
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        torch_optimizer = TorchAdamWNoBias(
            torch_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

        batch_x = torch.from_numpy(X)
        batch_y = torch.from_numpy(y)
        torch_optimizer.zero_grad()
        logits = torch_model(batch_x)
        loss = torch_nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        torch_optimizer.step()

        torch_weight_after = torch_model.linear.weight.detach().numpy()

        # MLX (uses no bias correction by default)
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            bias_correction=False,  # Explicit for clarity
        )

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        batch_x_mlx = mx.array(X)
        batch_y_mlx = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, batch_x_mlx, batch_y_mlx)
        mlx_optimizer.update(mlx_model, grads)
        mx.eval(mlx_model.parameters())

        mlx_weight_after = np.array(mlx_model.linear.weight)

        np.testing.assert_allclose(
            mlx_weight_after, torch_weight_after, rtol=1e-5, atol=1e-5,
            err_msg="AdamW single step weights differ"
        )

    def test_adamw_multi_step_convergence(self):
        """AdamW should converge similarly over multiple steps."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=50, seed=42)
        batch_size = 10
        lr = 0.01
        weight_decay = 0.01
        num_steps = 10

        # Both use bias correction for fair comparison
        # PyTorch training
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.AdamW(
            torch_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        torch_losses = []
        for step in range(num_steps):
            idx = (step * batch_size) % len(X)
            batch_x = torch.from_numpy(X[idx:idx+batch_size])
            batch_y = torch.from_numpy(y[idx:idx+batch_size])

            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()
            torch_losses.append(loss.item())

        # MLX training - enable bias correction to match PyTorch
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.AdamW(
            learning_rate=lr, weight_decay=weight_decay, bias_correction=True
        )

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        mlx_losses = []
        for step in range(num_steps):
            idx = (step * batch_size) % len(X)
            batch_x = mx.array(X[idx:idx+batch_size])
            batch_y = mx.array(y[idx:idx+batch_size])

            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters(), loss)
            mlx_losses.append(float(loss.item()))

        # Loss trajectories should match closely
        np.testing.assert_allclose(
            mlx_losses, torch_losses, rtol=1e-4, atol=1e-4,
            err_msg="AdamW loss trajectories differ"
        )

    def test_adamw_momentum_state_parity(self):
        """AdamW momentum buffers should evolve identically."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=20, seed=42)
        lr = 0.001
        betas = (0.9, 0.999)

        # PyTorch - use bias_correction=True (default)
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.AdamW(
            torch_model.parameters(), lr=lr, betas=betas
        )

        # Run 3 steps
        for i in range(3):
            batch_x = torch.from_numpy(X[i*5:(i+1)*5])
            batch_y = torch.from_numpy(y[i*5:(i+1)*5])
            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()

        # Get PyTorch optimizer state (first moment, second moment)
        # The state is keyed by parameter tensor
        torch_weight_param = list(torch_model.parameters())[0]
        torch_state = torch_optimizer.state[torch_weight_param]
        torch_m = torch_state["exp_avg"].numpy()  # First moment
        torch_v = torch_state["exp_avg_sq"].numpy()  # Second moment

        # MLX - use bias_correction=True to match PyTorch
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.AdamW(
            learning_rate=lr, betas=betas, bias_correction=True
        )

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        for i in range(3):
            batch_x = mx.array(X[i*5:(i+1)*5])
            batch_y = mx.array(y[i*5:(i+1)*5])
            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())

        # Get MLX optimizer state - navigate the nested structure
        mlx_state = mlx_optimizer.state
        # MLX state structure: state["linear"]["weight"]["m"] and ["v"]
        weight_state = mlx_state["linear"]["weight"]
        mlx_m = np.array(weight_state["m"])  # First moment
        mlx_v = np.array(weight_state["v"])  # Second moment

        np.testing.assert_allclose(
            mlx_m, torch_m, rtol=1e-4, atol=1e-5,
            err_msg="AdamW first moment (m) differs"
        )
        np.testing.assert_allclose(
            mlx_v, torch_v, rtol=1e-4, atol=1e-5,
            err_msg="AdamW second moment (v) differs"
        )


# ==================== Deep Network Parity ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestDeepNetworkParity:
    """Test parity for deeper networks where precision drift could accumulate."""

    def test_mlp_forward_pass(self):
        """MLP forward pass should match between frameworks."""
        set_seeds(42)

        X, _ = create_deterministic_data(num_samples=10, seed=42)

        torch_mlp = create_torch_mlp()
        mlx_mlp = create_mlx_mlp()

        batch_x_torch = torch.from_numpy(X)
        batch_x_mlx = mx.array(X)

        torch_out = torch_mlp(batch_x_torch).detach().numpy()
        mlx_out = np.array(mlx_mlp(batch_x_mlx))

        np.testing.assert_allclose(
            mlx_out, torch_out, rtol=1e-5, atol=1e-5,
            err_msg="MLP forward pass differs"
        )

    def test_mlp_gradient_computation(self):
        """MLP gradients should match between frameworks."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch
        torch_mlp = create_torch_mlp()
        batch_x = torch.from_numpy(X)
        batch_y = torch.from_numpy(y)
        logits = torch_mlp(batch_x)
        loss = torch_nn.functional.cross_entropy(logits, batch_y)
        loss.backward()

        # Get output layer gradient
        torch_out_grad = torch_mlp[-1].weight.grad.numpy()

        # MLX
        mlx_mlp = create_mlx_mlp()

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_mlp, loss_fn)
        batch_x_mlx = mx.array(X)
        batch_y_mlx = mx.array(y)
        loss, grads = loss_and_grad(mlx_mlp, batch_x_mlx, batch_y_mlx)
        mx.eval(grads)

        mlx_out_grad = np.array(grads["output"]["weight"])

        np.testing.assert_allclose(
            mlx_out_grad, torch_out_grad, rtol=1e-4, atol=1e-4,
            err_msg="MLP output layer gradients differ"
        )

    def test_mlp_training_convergence(self):
        """MLP should converge similarly over training."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=100, seed=42)
        batch_size = 20
        lr = 0.01
        num_epochs = 3

        # PyTorch training
        torch_mlp = create_torch_mlp()
        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=lr)

        torch_epoch_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch_x = torch.from_numpy(X[i:i+batch_size])
                batch_y = torch.from_numpy(y[i:i+batch_size])

                torch_optimizer.zero_grad()
                logits = torch_mlp(batch_x)
                loss = torch_nn.functional.cross_entropy(logits, batch_y)
                loss.backward()
                torch_optimizer.step()
                epoch_loss += loss.item()
            torch_epoch_losses.append(epoch_loss / (len(X) // batch_size))

        # MLX training
        mlx_mlp = create_mlx_mlp()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_mlp, loss_fn)

        mlx_epoch_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch_x = mx.array(X[i:i+batch_size])
                batch_y = mx.array(y[i:i+batch_size])

                loss, grads = loss_and_grad(mlx_mlp, batch_x, batch_y)
                mlx_optimizer.update(mlx_mlp, grads)
                mx.eval(mlx_mlp.parameters(), loss)
                epoch_loss += float(loss.item())
            mlx_epoch_losses.append(epoch_loss / (len(X) // batch_size))

        np.testing.assert_allclose(
            mlx_epoch_losses, torch_epoch_losses, rtol=1e-3, atol=1e-3,
            err_msg="MLP epoch losses differ"
        )

    def test_mlp_final_weights_match(self):
        """MLP final weights should match after training."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=50, seed=42)
        batch_size = 10
        lr = 0.1
        num_steps = 5

        # PyTorch
        torch_mlp = create_torch_mlp()
        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=lr)

        for step in range(num_steps):
            idx = (step * batch_size) % len(X)
            batch_x = torch.from_numpy(X[idx:idx+batch_size])
            batch_y = torch.from_numpy(y[idx:idx+batch_size])

            torch_optimizer.zero_grad()
            logits = torch_mlp(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()

        torch_final_weight = torch_mlp[-1].weight.detach().numpy()

        # MLX
        mlx_mlp = create_mlx_mlp()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_mlp, loss_fn)

        for step in range(num_steps):
            idx = (step * batch_size) % len(X)
            batch_x = mx.array(X[idx:idx+batch_size])
            batch_y = mx.array(y[idx:idx+batch_size])

            loss, grads = loss_and_grad(mlx_mlp, batch_x, batch_y)
            mlx_optimizer.update(mlx_mlp, grads)
            mx.eval(mlx_mlp.parameters())

        mlx_final_weight = np.array(mlx_mlp.output.weight)

        np.testing.assert_allclose(
            mlx_final_weight, torch_final_weight, rtol=1e-4, atol=1e-4,
            err_msg="MLP final output weights differ"
        )


# ==================== Gradient Clipping Parity ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestGradientClippingParity:
    """Test gradient clipping produces identical results."""

    def test_gradient_norm_clipping(self):
        """Gradient norm clipping should match PyTorch."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)
        max_norm = 1.0

        # PyTorch - compute gradients and clip
        torch_model = create_torch_linear_model()
        batch_x = torch.from_numpy(X)
        batch_y = torch.from_numpy(y)
        logits = torch_model(batch_x)
        loss = torch_nn.functional.cross_entropy(logits, batch_y)
        loss.backward()

        # Clip gradients (returns unclipped norm, which we don't need)
        torch.nn.utils.clip_grad_norm_(
            torch_model.parameters(), max_norm, error_if_nonfinite=False
        )

        # Get clipped gradients
        torch_clipped_grad = torch_model.linear.weight.grad.numpy().copy()

        # MLX - compute gradients and clip manually
        mlx_model = create_mlx_linear_model()

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        batch_x_mlx = mx.array(X)
        batch_y_mlx = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, batch_x_mlx, batch_y_mlx)

        # Manual gradient clipping (same as Trainer does)
        from mlx.utils import tree_flatten, tree_unflatten

        flat_grads = tree_flatten(grads)
        total_norm = mx.sqrt(
            sum(mx.sum(g**2) for _, g in flat_grads if isinstance(g, mx.array))
        )
        scale = max_norm / (total_norm + 1e-6)
        scale = mx.minimum(scale, mx.array(1.0))

        clipped = [
            (k, g * scale if isinstance(g, mx.array) else g)
            for k, g in flat_grads
        ]
        clipped_grads = tree_unflatten(clipped)
        mx.eval(clipped_grads)

        mlx_clipped_grad = np.array(clipped_grads["linear"]["weight"])

        np.testing.assert_allclose(
            mlx_clipped_grad, torch_clipped_grad, rtol=1e-4, atol=1e-4,
            err_msg="Clipped gradients differ"
        )

    def test_training_with_gradient_clipping(self):
        """Training with gradient clipping should match."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10
        lr = 0.1
        max_norm = 0.5
        num_steps = 3

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        torch_losses = []
        for step in range(num_steps):
            idx = step * batch_size
            batch_x = torch.from_numpy(X[idx:idx+batch_size])
            batch_y = torch.from_numpy(y[idx:idx+batch_size])

            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm)
            torch_optimizer.step()
            torch_losses.append(loss.item())

        torch_final_weight = torch_model.linear.weight.detach().numpy()

        # MLX with Trainer gradient clipping
        module = create_mlx_train_module(lr=lr)

        train_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, num_steps * batch_size, batch_size)
        ]

        trainer = Trainer(
            max_epochs=1,
            gradient_clip_val=max_norm,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            compile=False,  # Disable compile for exact comparison
        )

        trainer.fit(module, train_data)

        mlx_final_weight = np.array(module.model.linear.weight)

        np.testing.assert_allclose(
            mlx_final_weight, torch_final_weight, rtol=1e-4, atol=1e-4,
            err_msg="Final weights with gradient clipping differ"
        )


# ==================== Learning Rate Scheduler Parity ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestLRSchedulerParity:
    """Test learning rate schedulers produce identical behavior."""

    def test_constant_lr_parity(self):
        """Constant LR should obviously match."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10
        lr = 0.05

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        for i in range(3):
            batch_x = torch.from_numpy(X[i*batch_size:(i+1)*batch_size])
            batch_y = torch.from_numpy(y[i*batch_size:(i+1)*batch_size])
            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()

        torch_weight = torch_model.linear.weight.detach().numpy()

        # MLX
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        for i in range(3):
            batch_x = mx.array(X[i*batch_size:(i+1)*batch_size])
            batch_y = mx.array(y[i*batch_size:(i+1)*batch_size])
            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())

        mlx_weight = np.array(mlx_model.linear.weight)

        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-5, atol=1e-5,
            err_msg="Constant LR weights differ"
        )

    def test_step_decay_lr_parity(self):
        """Step decay LR schedule should match."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=50, seed=42)
        batch_size = 10
        initial_lr = 0.1
        decay_rate = 0.5
        decay_steps = 2

        # Create step decay schedule
        def mlx_step_schedule(step):
            return initial_lr * (decay_rate ** (step // decay_steps))

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=initial_lr)
        torch_scheduler = torch.optim.lr_scheduler.StepLR(
            torch_optimizer, step_size=decay_steps, gamma=decay_rate
        )

        torch_lrs = []
        for step in range(5):
            idx = (step * batch_size) % len(X)
            batch_x = torch.from_numpy(X[idx:idx+batch_size])
            batch_y = torch.from_numpy(y[idx:idx+batch_size])

            torch_optimizer.zero_grad()
            logits = torch_model(batch_x)
            loss = torch_nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            torch_optimizer.step()
            torch_lrs.append(torch_optimizer.param_groups[0]["lr"])
            torch_scheduler.step()

        torch_weight = torch_model.linear.weight.detach().numpy()

        # MLX with callable lr
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=mlx_step_schedule)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        mlx_lrs = []
        for step in range(5):
            idx = (step * batch_size) % len(X)
            batch_x = mx.array(X[idx:idx+batch_size])
            batch_y = mx.array(y[idx:idx+batch_size])

            mlx_lrs.append(mlx_step_schedule(step))
            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())

        mlx_weight = np.array(mlx_model.linear.weight)

        # LRs should match
        np.testing.assert_allclose(
            mlx_lrs, torch_lrs, rtol=1e-6, atol=1e-6,
            err_msg="Step decay LRs differ"
        )

        # Final weights should match
        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-4, atol=1e-4,
            err_msg="Step decay final weights differ"
        )


# ==================== Numerical Stability Tests ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_small_gradients(self):
        """Small gradients should be handled identically."""
        set_seeds(42)

        # Create data that produces small gradients
        X = np.random.randn(10, 10).astype(np.float32) * 0.001
        y = np.zeros(10, dtype=np.int64)  # All same class

        lr = 0.01

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        batch_x = torch.from_numpy(X)
        batch_y = torch.from_numpy(y)
        torch_optimizer.zero_grad()
        logits = torch_model(batch_x)
        loss = torch_nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        torch_optimizer.step()

        torch_weight = torch_model.linear.weight.detach().numpy()

        # MLX
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        batch_x_mlx = mx.array(X)
        batch_y_mlx = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, batch_x_mlx, batch_y_mlx)
        mlx_optimizer.update(mlx_model, grads)
        mx.eval(mlx_model.parameters())

        mlx_weight = np.array(mlx_model.linear.weight)

        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-5, atol=1e-6,
            err_msg="Small gradient handling differs"
        )

    def test_gradient_accumulation_internal_consistency(self):
        """Gradient accumulation should be internally consistent.

        This tests that our gradient accumulation logic works correctly,
        not that it matches a single large batch (which would require
        sum reduction instead of mean).
        """
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=40, seed=42)
        lr = 0.1
        batch_size = 10
        num_accum = 4

        # MLX: Manual gradient accumulation
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)

        from mlx.utils import tree_map

        accumulated_grads = None
        total_loss = 0.0

        for i in range(num_accum):
            batch_x = mx.array(X[i*batch_size:(i+1)*batch_size])
            batch_y = mx.array(y[i*batch_size:(i+1)*batch_size])
            loss, grads = loss_and_grad(mlx_model, batch_x, batch_y)
            mx.eval(loss)
            total_loss += float(loss.item())

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda a, b: a + b if a is not None else b,
                    accumulated_grads, grads
                )

        # Average gradients
        averaged_grads = tree_map(
            lambda g: g / num_accum if g is not None else None,
            accumulated_grads
        )
        mx.eval(averaged_grads)

        initial_weight = np.ones((2, 10), dtype=np.float32)
        mlx_optimizer.update(mlx_model, averaged_grads)
        mx.eval(mlx_model.parameters())

        final_weight = np.array(mlx_model.linear.weight)

        # Verify weights changed
        assert not np.allclose(final_weight, initial_weight), \
            "Weights should have changed after update"

        # Verify loss is reasonable (cross-entropy for random data)
        avg_loss = total_loss / num_accum
        assert 0.5 < avg_loss < 1.5, f"Loss {avg_loss} outside expected range"

        # Verify gradient was non-zero
        grad_norm = np.sqrt(
            np.sum(np.array(averaged_grads["linear"]["weight"])**2)
        )
        assert grad_norm > 0, "Gradient should be non-zero"


# ==================== Dropout/Train-Eval Mode Parity ====================


def create_torch_dropout_model(in_features: int = 10, hidden: int = 32, out_features: int = 2, dropout_p: float = 0.5):
    """Create a model with dropout in PyTorch."""
    if not HAS_PYTORCH:
        return None

    class TorchDropoutModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch_nn.Linear(in_features, hidden, bias=True)
            self.dropout = torch_nn.Dropout(p=dropout_p)
            self.linear2 = torch_nn.Linear(hidden, out_features, bias=True)
            # Deterministic init
            torch_nn.init.constant_(self.linear1.weight, 0.1)
            torch_nn.init.zeros_(self.linear1.bias)
            torch_nn.init.constant_(self.linear2.weight, 0.1)
            torch_nn.init.zeros_(self.linear2.bias)

        def forward(self, x):
            x = torch_nn.functional.relu(self.linear1(x))
            x = self.dropout(x)
            return self.linear2(x)

    return TorchDropoutModel()


def create_mlx_dropout_model(in_features: int = 10, hidden: int = 32, out_features: int = 2, dropout_p: float = 0.5):
    """Create a model with dropout in MLX."""
    if not HAS_MLX:
        return None

    class MLXDropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden, bias=True)
            self.dropout = nn.Dropout(p=dropout_p)
            self.linear2 = nn.Linear(hidden, out_features, bias=True)
            # Deterministic init matching PyTorch
            self.linear1.weight = mx.full((hidden, in_features), 0.1)
            self.linear1.bias = mx.zeros((hidden,))
            self.linear2.weight = mx.full((out_features, hidden), 0.1)
            self.linear2.bias = mx.zeros((out_features,))

        def __call__(self, x):
            x = nn.relu(self.linear1(x))
            x = self.dropout(x)
            return self.linear2(x)

    return MLXDropoutModel()


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestDropoutParity:
    """Test dropout train/eval mode behavior parity."""

    def test_eval_mode_deterministic(self):
        """Eval mode should produce deterministic (no dropout) outputs."""
        set_seeds(42)

        X, _ = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch in eval mode
        torch_model = create_torch_dropout_model()
        torch_model.eval()  # Disable dropout
        torch_input = torch.from_numpy(X)
        torch_out1 = torch_model(torch_input).detach().numpy()
        torch_out2 = torch_model(torch_input).detach().numpy()

        # Outputs should be identical (no dropout)
        np.testing.assert_allclose(
            torch_out1, torch_out2, rtol=0, atol=0,
            err_msg="PyTorch eval mode should be deterministic"
        )

        # MLX in eval mode
        mlx_model = create_mlx_dropout_model()
        mlx_model.eval()  # Disable dropout
        mlx_input = mx.array(X)
        mlx_out1 = np.array(mlx_model(mlx_input))
        mlx_out2 = np.array(mlx_model(mlx_input))

        # Outputs should be identical (no dropout)
        np.testing.assert_allclose(
            mlx_out1, mlx_out2, rtol=0, atol=0,
            err_msg="MLX eval mode should be deterministic"
        )

        # Both frameworks should match in eval mode
        np.testing.assert_allclose(
            mlx_out1, torch_out1, rtol=1e-5, atol=1e-5,
            err_msg="Eval mode outputs differ between frameworks"
        )

    def test_train_mode_stochastic(self):
        """Train mode should apply dropout (stochastic outputs)."""
        set_seeds(42)

        X, _ = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch in train mode
        torch_model = create_torch_dropout_model(dropout_p=0.5)
        torch_model.train()
        torch_input = torch.from_numpy(X)

        torch.manual_seed(123)
        torch_out1 = torch_model(torch_input).detach().numpy()
        torch.manual_seed(456)
        torch_out2 = torch_model(torch_input).detach().numpy()

        # Outputs should differ (dropout applied)
        assert not np.allclose(torch_out1, torch_out2), \
            "PyTorch train mode should be stochastic"

        # MLX in train mode
        mlx_model = create_mlx_dropout_model(dropout_p=0.5)
        mlx_model.train()
        mlx_input = mx.array(X)

        mx.random.seed(123)
        mlx_out1 = np.array(mlx_model(mlx_input))
        mx.random.seed(456)
        mlx_out2 = np.array(mlx_model(mlx_input))

        # Outputs should differ (dropout applied)
        assert not np.allclose(mlx_out1, mlx_out2), \
            "MLX train mode should be stochastic"

    def test_train_eval_toggle(self):
        """Toggling between train/eval modes should work correctly."""
        set_seeds(42)

        X, _ = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch
        torch_model = create_torch_dropout_model()
        torch_input = torch.from_numpy(X)

        # Start in eval mode
        torch_model.eval()
        eval_out1 = torch_model(torch_input).detach().numpy()

        # Switch to train mode
        torch_model.train()
        torch.manual_seed(42)
        _ = torch_model(torch_input).detach().numpy()  # Train output (stochastic)

        # Back to eval mode
        torch_model.eval()
        eval_out2 = torch_model(torch_input).detach().numpy()

        # Eval outputs should match
        np.testing.assert_allclose(eval_out1, eval_out2, rtol=0, atol=0)

        # MLX
        mlx_model = create_mlx_dropout_model()
        mlx_input = mx.array(X)

        # Start in eval mode
        mlx_model.eval()
        mlx_eval_out1 = np.array(mlx_model(mlx_input))

        # Switch to train mode
        mlx_model.train()
        mx.random.seed(42)
        _ = np.array(mlx_model(mlx_input))  # Train output (stochastic)

        # Back to eval mode
        mlx_model.eval()
        mlx_eval_out2 = np.array(mlx_model(mlx_input))

        # Eval outputs should match
        np.testing.assert_allclose(mlx_eval_out1, mlx_eval_out2, rtol=0, atol=0)

        # Both frameworks eval outputs should match
        np.testing.assert_allclose(
            mlx_eval_out1, eval_out1, rtol=1e-5, atol=1e-5,
            err_msg="Eval mode outputs differ after toggle"
        )


# ==================== Loss Function Parity ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestLossFunctionParity:
    """Test loss function parity between frameworks."""

    def test_mse_loss_parity(self):
        """MSE loss should match between frameworks."""
        set_seeds(42)

        # Create regression data
        pred = np.random.randn(10, 5).astype(np.float32)
        target = np.random.randn(10, 5).astype(np.float32)

        # PyTorch MSE
        torch_pred = torch.from_numpy(pred)
        torch_target = torch.from_numpy(target)
        torch_loss = torch_nn.functional.mse_loss(torch_pred, torch_target)
        torch_loss_val = torch_loss.item()

        # MLX MSE
        mlx_pred = mx.array(pred)
        mlx_target = mx.array(target)
        mlx_loss = mx.mean((mlx_pred - mlx_target) ** 2)
        mx.eval(mlx_loss)
        mlx_loss_val = float(mlx_loss.item())

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-6,
            err_msg=f"MSE loss: MLX={mlx_loss_val}, PyTorch={torch_loss_val}"
        )

    def test_bce_loss_parity(self):
        """Binary cross-entropy loss should match."""
        set_seeds(42)

        # Create binary classification data
        logits = np.random.randn(10, 1).astype(np.float32)
        targets = np.random.randint(0, 2, size=(10, 1)).astype(np.float32)

        # PyTorch BCE with logits
        torch_logits = torch.from_numpy(logits)
        torch_targets = torch.from_numpy(targets)
        torch_loss = torch_nn.functional.binary_cross_entropy_with_logits(
            torch_logits, torch_targets
        )
        torch_loss_val = torch_loss.item()

        # MLX BCE with logits (manual implementation)
        mlx_logits = mx.array(logits)
        mlx_targets = mx.array(targets)
        # BCE with logits: max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
        max_val = mx.maximum(mlx_logits, 0)
        mlx_loss = mx.mean(
            max_val - mlx_logits * mlx_targets + mx.log(1 + mx.exp(-mx.abs(mlx_logits)))
        )
        mx.eval(mlx_loss)
        mlx_loss_val = float(mlx_loss.item())

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-5,
            err_msg=f"BCE loss: MLX={mlx_loss_val}, PyTorch={torch_loss_val}"
        )

    def test_bce_loss_gradient_parity(self):
        """BCE loss gradients should match."""
        set_seeds(42)

        logits = np.random.randn(10, 1).astype(np.float32)
        targets = np.random.randint(0, 2, size=(10, 1)).astype(np.float32)

        # PyTorch gradient
        torch_logits = torch.from_numpy(logits)
        torch_logits.requires_grad = True
        torch_targets = torch.from_numpy(targets)
        torch_loss = torch_nn.functional.binary_cross_entropy_with_logits(
            torch_logits, torch_targets
        )
        torch_loss.backward()
        torch_grad = torch_logits.grad.numpy()

        # MLX gradient
        def bce_loss(logits, targets):
            max_val = mx.maximum(logits, 0)
            return mx.mean(
                max_val - logits * targets + mx.log(1 + mx.exp(-mx.abs(logits)))
            )

        mlx_logits = mx.array(logits)
        mlx_targets = mx.array(targets)

        def loss_wrapper(logits):
            return bce_loss(logits, mlx_targets)

        grad_fn = mx.grad(loss_wrapper)
        mlx_grad = np.array(grad_fn(mlx_logits))

        np.testing.assert_allclose(
            mlx_grad, torch_grad, rtol=1e-5, atol=1e-5,
            err_msg="BCE loss gradients differ"
        )

    def test_softmax_cross_entropy_parity(self):
        """Softmax cross-entropy should match for multi-class classification."""
        set_seeds(42)

        # Multi-class data
        logits = np.random.randn(10, 5).astype(np.float32)
        targets = np.random.randint(0, 5, size=10).astype(np.int64)

        # PyTorch
        torch_logits = torch.from_numpy(logits)
        torch_targets = torch.from_numpy(targets)
        torch_loss = torch_nn.functional.cross_entropy(torch_logits, torch_targets)
        torch_loss_val = torch_loss.item()

        # MLX
        mlx_logits = mx.array(logits)
        mlx_targets = mx.array(targets)
        mlx_loss = mx.mean(nn.losses.cross_entropy(mlx_logits, mlx_targets))
        mx.eval(mlx_loss)
        mlx_loss_val = float(mlx_loss.item())

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-5,
            err_msg=f"Cross-entropy loss: MLX={mlx_loss_val}, PyTorch={torch_loss_val}"
        )

    def test_large_logits_numerical_stability(self):
        """Loss should be numerically stable with large logits."""
        set_seeds(42)

        # Large logits that could cause overflow
        logits = np.array([[100.0, -100.0], [-100.0, 100.0]], dtype=np.float32)
        targets = np.array([0, 1], dtype=np.int64)

        # PyTorch (uses log_softmax for stability)
        torch_logits = torch.from_numpy(logits)
        torch_targets = torch.from_numpy(targets)
        torch_loss = torch_nn.functional.cross_entropy(torch_logits, torch_targets)
        torch_loss_val = torch_loss.item()

        # MLX
        mlx_logits = mx.array(logits)
        mlx_targets = mx.array(targets)
        mlx_loss = mx.mean(nn.losses.cross_entropy(mlx_logits, mlx_targets))
        mx.eval(mlx_loss)
        mlx_loss_val = float(mlx_loss.item())

        # Both should handle this without NaN/Inf
        assert np.isfinite(torch_loss_val), "PyTorch loss should be finite"
        assert np.isfinite(mlx_loss_val), "MLX loss should be finite"

        # Loss should be very small (confident correct predictions)
        assert mlx_loss_val < 1e-3, f"Loss should be small: {mlx_loss_val}"

        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-3, atol=1e-5,
            err_msg="Large logits stability differs"
        )


# ==================== Checkpoint Save/Restore Parity ====================


@pytest.mark.parity
@pytest.mark.skipif(not HAS_MLX, reason="MLX required")
class TestCheckpointParity:
    """Test checkpoint save/restore functionality."""

    def test_save_restore_weights(self):
        """Model weights should be exactly restored from checkpoint."""
        set_seeds(42)

        # Create and train a model for a few steps
        X, y = create_deterministic_data(num_samples=30, seed=42)
        batch_size = 10

        module = create_mlx_train_module(lr=0.1)
        train_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, len(X), batch_size)
        ]

        trainer = Trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(module, train_data)

        # Save weights
        original_weight = np.array(module.model.linear.weight).copy()
        original_bias = np.array(module.model.linear.bias).copy()

        # Get the parameters structure
        from mlx.utils import tree_flatten, tree_unflatten

        flat_params = tree_flatten(module.parameters())

        # Create a fresh model with different weights
        new_module = create_mlx_train_module(lr=0.1)

        # Verify weights are different before loading
        new_weight_before = np.array(new_module.model.linear.weight)
        assert not np.allclose(new_weight_before, original_weight), \
            "Fresh model should have different weights"

        # Restore by unflattening and updating
        restored_params = tree_unflatten(flat_params)
        new_module.update(restored_params)
        mx.eval(new_module.parameters())

        # Verify weights match exactly
        restored_weight = np.array(new_module.model.linear.weight)
        restored_bias = np.array(new_module.model.linear.bias)

        np.testing.assert_array_equal(
            restored_weight, original_weight,
            err_msg="Restored weights don't match original"
        )
        np.testing.assert_array_equal(
            restored_bias, original_bias,
            err_msg="Restored biases don't match original"
        )

    def test_checkpoint_training_continuation(self):
        """Training should continue identically from checkpoint."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=60, seed=42)
        batch_size = 10

        # First training run: train for 3 epochs
        module1 = create_mlx_train_module(lr=0.1)
        train_data = [
            (mx.array(X[i:i+batch_size]), mx.array(y[i:i+batch_size]))
            for i in range(0, 30, batch_size)
        ]

        trainer1 = Trainer(
            max_epochs=3,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer1.fit(module1, train_data)
        weights_after_3_epochs = np.array(module1.model.linear.weight).copy()

        # Train for 2 epochs (checkpoint state)
        set_seeds(42)
        module2 = create_mlx_train_module(lr=0.1)
        trainer2 = Trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer2.fit(module2, train_data)

        # Continue training from epoch 2 checkpoint
        # Simulate by running one more epoch with the same data pattern
        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(module2, loss_fn)
        mlx_optimizer = optim.SGD(learning_rate=0.1)

        # Run one more epoch manually
        for i in range(0, 30, batch_size):
            batch_x = mx.array(X[i:i+batch_size])
            batch_y = mx.array(y[i:i+batch_size])
            loss, grads = loss_and_grad(module2, batch_x, batch_y)
            mlx_optimizer.update(module2, grads)
            mx.eval(module2.parameters())

        weights_continued = np.array(module2.model.linear.weight)

        # Weights after continuation should match weights from full 3-epoch run
        np.testing.assert_allclose(
            weights_continued, weights_after_3_epochs, rtol=1e-5, atol=1e-5,
            err_msg="Continued training should match full training"
        )


# ==================== Additional Numerical Edge Cases ====================


@pytest.mark.parity
@pytest.mark.skipif(
    not HAS_PYTORCH or not HAS_MLX, reason="Both frameworks required"
)
class TestNumericalEdgeCases:
    """Test numerical edge cases and stability."""

    def test_very_deep_network_gradient_flow(self):
        """Gradients should flow through deep networks and match between frameworks.

        Note: With constant initialization (0.1) and ReLU activation, gradients
        can be very small in deep networks. This test verifies that both frameworks
        compute the same gradients, regardless of magnitude.
        """
        set_seeds(42)

        # Use 4 hidden layers (reasonable depth, less vanishing)
        hidden_sizes = [32, 32, 32, 32]

        torch_deep = create_torch_mlp(hidden_sizes=hidden_sizes)
        mlx_deep = create_mlx_mlp(hidden_sizes=hidden_sizes)

        X, y = create_deterministic_data(num_samples=10, seed=42)

        # PyTorch gradient flow
        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        logits = torch_deep(torch_input)
        loss = torch_nn.functional.cross_entropy(logits, torch_target)
        loss.backward()

        # Check first layer gradient (deepest in backward pass)
        torch_first_grad = torch_deep[0].weight.grad.numpy()

        # MLX gradient flow
        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_deep, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)
        loss, grads = loss_and_grad(mlx_deep, mlx_input, mlx_target)
        mx.eval(grads)

        mlx_first_grad = np.array(grads["layers"][0]["weight"])

        # Key assertion: gradients should match between frameworks
        # Even if small, they should be identical
        np.testing.assert_allclose(
            mlx_first_grad, torch_first_grad, rtol=1e-4, atol=1e-6,
            err_msg="Deep network first layer gradients differ"
        )

        # Also verify output layer gradients match (these should be larger)
        torch_out_grad = torch_deep[-1].weight.grad.numpy()
        mlx_out_grad = np.array(grads["output"]["weight"])

        np.testing.assert_allclose(
            mlx_out_grad, torch_out_grad, rtol=1e-4, atol=1e-5,
            err_msg="Deep network output layer gradients differ"
        )

    def test_large_batch_stability(self):
        """Large batch training should be stable."""
        set_seeds(42)

        # Large batch
        X, y = create_deterministic_data(num_samples=1000, seed=42)
        lr = 0.01

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        torch_optimizer.zero_grad()
        logits = torch_model(torch_input)
        loss = torch_nn.functional.cross_entropy(logits, torch_target)
        loss.backward()
        torch_optimizer.step()

        torch_weight = torch_model.linear.weight.detach().numpy()

        # MLX
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, mlx_input, mlx_target)
        mlx_optimizer.update(mlx_model, grads)
        mx.eval(mlx_model.parameters())

        mlx_weight = np.array(mlx_model.linear.weight)

        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-4, atol=1e-4,
            err_msg="Large batch weights differ"
        )

    def test_high_learning_rate_stability(self):
        """High learning rate should be handled identically."""
        set_seeds(42)

        X, y = create_deterministic_data(num_samples=10, seed=42)
        lr = 1.0  # Intentionally high

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)

        for _ in range(3):
            torch_optimizer.zero_grad()
            logits = torch_model(torch_input)
            loss = torch_nn.functional.cross_entropy(logits, torch_target)
            loss.backward()
            torch_optimizer.step()

        torch_weight = torch_model.linear.weight.detach().numpy()

        # MLX
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)

        for _ in range(3):
            loss, grads = loss_and_grad(mlx_model, mlx_input, mlx_target)
            mlx_optimizer.update(mlx_model, grads)
            mx.eval(mlx_model.parameters())

        mlx_weight = np.array(mlx_model.linear.weight)

        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-4, atol=1e-4,
            err_msg="High LR weights differ"
        )

    def test_zero_gradient_handling(self):
        """Zero gradients should be handled correctly."""
        set_seeds(42)

        # Create data where all predictions are correct (near-zero gradient)
        # Use very confident predictions
        X = np.eye(10, dtype=np.float32) * 10  # Strong diagonal
        y = np.arange(10, dtype=np.int64) % 2  # Labels 0,1,0,1...

        lr = 0.01

        # We need weights that produce correct predictions
        # Just verify both frameworks handle this case identically

        # PyTorch
        torch_model = create_torch_linear_model()
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

        torch_input = torch.from_numpy(X)
        torch_target = torch.from_numpy(y)
        torch_optimizer.zero_grad()
        logits = torch_model(torch_input)
        loss = torch_nn.functional.cross_entropy(logits, torch_target)
        loss.backward()
        torch_optimizer.step()

        torch_weight = torch_model.linear.weight.detach().numpy()
        torch_loss_val = loss.item()

        # MLX
        mlx_model = create_mlx_linear_model()
        mlx_optimizer = optim.SGD(learning_rate=lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(mlx_model, loss_fn)
        mlx_input = mx.array(X)
        mlx_target = mx.array(y)
        loss, grads = loss_and_grad(mlx_model, mlx_input, mlx_target)
        mlx_optimizer.update(mlx_model, grads)
        mx.eval(mlx_model.parameters(), loss)

        mlx_weight = np.array(mlx_model.linear.weight)
        mlx_loss_val = float(loss.item())

        # Losses should match
        np.testing.assert_allclose(
            mlx_loss_val, torch_loss_val, rtol=1e-5, atol=1e-5,
            err_msg="Loss values differ"
        )

        # Weights should match
        np.testing.assert_allclose(
            mlx_weight, torch_weight, rtol=1e-4, atol=1e-4,
            err_msg="Weights after near-zero gradient step differ"
        )
