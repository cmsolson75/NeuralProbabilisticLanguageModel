from .model import MLP
from .utils import get_device

import torch
from nplm.optimizer import Optimizer
from typing import Tuple
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: MLP,
        optim: Optimizer,
        loss_fn: torch.nn.Module,
        batch_size: int,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        lr_ramp: Tuple[float, float] = (0.1, 0.001),
    ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.data = train_data
        self.batch_size = batch_size
        self.device = get_device()
        self.lr_ramp = lr_ramp

    def fit(self, batches: int, log_interval: int = 1000) -> None:
        """Trains the model for a specified number of batches."""
        lr1, lr2 = self.lr_ramp
        lr_decay = torch.linspace(lr1, lr2, batches)
        Xs, Ys = self.data
        Xs, Ys = Xs.to(self.device), Ys.to(self.device)

        self.model.train()
        with tqdm(total=batches) as pbar:
            for batch in range(batches):
                batch_ix = torch.randint(0, Xs.shape[0], (self.batch_size,))
                logits = self.model(Xs[batch_ix])
                loss = self.loss_fn(logits, Ys[batch_ix])
                self.optim.zero_grad()
                loss.backward()
                lr = lr_decay[batch].item()
                self.optim.step(lr)

                if batch % log_interval == 0:
                    pbar.update(log_interval)
                    pbar.set_postfix({"loss": loss.item()})

    def eval(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Evaluates the model on the given data."""
        Xs, Ys = data
        Xs, Ys = Xs.to(self.device), Ys.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xs)
            loss = self.loss_fn(logits, Ys)
            perplexity = torch.exp(loss)
            print(
                f"Eval NLL {loss.item():.4f}, Eval Perplexity {perplexity.item():.4f}"
            )
        return perplexity.item()
