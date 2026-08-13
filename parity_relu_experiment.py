import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ParityMLP(nn.Module):
    """A two-layer ReLU network for learning the parity function on {±1}^d."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ParityMLP2(nn.Module):
    """A two-layer ReLU network for learning the parity function on {±1}^d."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ParityMLP3(nn.Module):
    """A two-layer ReLU network for learning the parity function on {±1}^d."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.SiLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.SiLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.SiLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.SiLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_parity_data(n_samples: int, input_dim: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniform points in {±1}^d and label them by the product of coordinates."""
    bits = torch.randint(0, 2, (n_samples, input_dim), device=device, dtype=torch.float32)
    x = bits * 2.0 - 1.0
    y = torch.prod(x, dim=1, keepdim=True).float()
    return x, y


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
) -> float:
    """Train a regression model with mean squared error on parity labels."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #c# riterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = float("inf")
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            #print(preds.shape)
            #print(batch_y.shape)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        last_loss = epoch_loss / x.size(0)

    return last_loss


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str | None = None) -> tuple[float, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).squeeze(-1)
        targets = y.to(device).squeeze(-1)
        mse = nn.functional.mse_loss(preds, targets).item()
        sign_accuracy = ((preds >= 0.0) == (targets >= 0.0)).float().mean().item()
    return mse, sign_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-layer ReLU network on the parity function")
    parser.add_argument("--input-dim", type=int, default=800)
    parser.add_argument("--hidden-dim", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    train_x, train_y = generate_parity_data(args.train_samples, args.input_dim, device=args.device)
    eval_x, eval_y = generate_parity_data(args.eval_samples, args.input_dim, device=args.device)

    model = ParityMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    train_loss = train_model(
        model,
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    eval_mse, sign_accuracy = evaluate_model(model, eval_x, eval_y, device=args.device)

    print(f"input_dim={args.input_dim} hidden_dim={args.hidden_dim}")
    print(f"train_samples={args.train_samples} eval_samples={args.eval_samples}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    print(f"device={args.device}")
    print(f"final_train_mse={train_loss:.6f}")
    print(f"eval_mse={eval_mse:.6f}")
    print(f"sign_accuracy={sign_accuracy:.4f}")


if __name__ == "__main__":
    main()
