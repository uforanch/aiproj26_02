import torch

from parity_relu_experiment import ParityMLP, generate_parity_data, train_model


def test_generate_parity_data_shapes_and_labels():
    x, y = generate_parity_data(n_samples=32, input_dim=8, device="cpu")

    assert x.shape == (32, 8)
    assert y.shape == (32, 1)
    assert torch.all((x == 1) | (x == -1))
    assert torch.allclose(y, torch.prod(x, dim=1, keepdim=True).float())


def test_model_forward_and_training_smoke_test():
    model = ParityMLP(input_dim=8, hidden_dim=16)
    x, y = generate_parity_data(n_samples=32, input_dim=8, device="cpu")

    logits = model(x)
    assert logits.shape == (32, 1)

    loss = train_model(model, x, y, epochs=1, batch_size=16, lr=1e-3)
    assert loss >= 0.0
