from typing import Callable, Tuple, Generator, Dict, Optional

from . import weight_initializers as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_device

Initializer = Callable[..., torch.Tensor]


def init_layer_weights(
    in_features: int,
    out_features: int,
    initializer: Initializer,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W = initializer(in_features, out_features, generator=generator)
    b = initializer(out_features, generator=generator)
    return W, b


def build_model_params(
    v: int,
    emb_size: int,
    context: int,
    H: int,
    initializer: Initializer,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    C = initializer(v, emb_size, generator=generator).to(device)
    W1, b1 = init_layer_weights(
        emb_size * context, H, initializer=initializer, generator=generator
    )
    W2, b2 = init_layer_weights(H, v, initializer=initializer, generator=generator)
    C, W1, b1, W2, b2 = (
        C.to(device),
        W1.to(device),
        b1.to(device),
        W2.to(device),
        b2.to(device),
    )
    return C, W1, b1, W2, b2


class Linear:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

    def access_layer_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weight, self.bias


class MLP:
    def __init__(
        self,
        v: int,
        emb_size: int,
        context: int,
        H: int,
        initializer: str = "normal",
        generator: Optional[torch.Generator] = None,
    ):
        # seed = torch.Generator().manual_seed()
        device = get_device()
        self.model_dims: Dict[str, int] = {
            "v": v,
            "emb_size": emb_size,
            "context": context,
            "H": H,
        }
        initializers: Dict[str, Initializer] = {
            "uniform": init.xavier_uniform,
            "normal": init.xavier_normal,
        }
        initializer = initializers[initializer]

        self.h_input: int = emb_size * context

        C, W1, b1, W2, b2 = build_model_params(
            v,
            emb_size,
            context,
            H,
            initializer=initializer,
            generator=generator,
            device=device,
        )

        self._init_layers(C, W1, b1, W2, b2)

    def _init_layers(
        self,
        C: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W2: torch.Tensor,
        b2: torch.Tensor,
    ) -> None:
        self.C = C
        self.hidden_layer = Linear(W1, b1)
        self.output_layer = Linear(W2, b2)

    def _access_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.C
        w1, b1 = self.hidden_layer.access_layer_params()
        w2, b2 = self.output_layer.access_layer_params()
        return c, w1, b1, w2, b2

    def parameters(self) -> Generator[torch.Tensor, None, None]:
        for param in self._access_params():
            yield param

    def train(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def eval(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None

    def access_param_dict(self) -> Dict[str, torch.Tensor]:
        C, W1, b1, W2, b2 = self._access_params()
        return {
            "C": C,
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "model_dims": self.model_dims,
        }

    @staticmethod
    def model_dict_to_params(
        model_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        accepted_keys = ["C", "W1", "b1", "W2", "b2", "model_dims"]
        for key in model_dict:
            if key not in accepted_keys:
                raise ValueError(f"{key} not allowed")

        C = model_dict["C"]
        W1 = model_dict["W1"]
        b1 = model_dict["b1"]
        W2 = model_dict["W2"]
        b2 = model_dict["b2"]
        return C, W1, b1, W2, b2

    def save(self, model_name: str = "model.pt") -> None:
        model = self.access_param_dict()
        torch.save(model, model_name)

    def load_state_dict(self, model_path: str) -> None:
        model = torch.load(model_path, map_location=torch.device(get_device()))
        C, W1, b1, W2, b2 = MLP.model_dict_to_params(model)
        self._init_layers(C, W1, b1, W2, b2)

    @classmethod
    def init_model_from_load(cls, model_path: str) -> "MLP":
        model_dict = torch.load(model_path, map_location=torch.device(get_device()))
        md = model_dict["model_dims"]
        out = cls(md["v"], md["emb_size"], md["context"], md["H"])
        out.load_state_dict(model_path)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.C[x].view(-1, self.h_input)
        z = F.tanh(self.hidden_layer(emb))
        logits = self.output_layer(z)
        return logits

    def __repr__(self) -> str:
        return f"MLP({self.model_dims})"
