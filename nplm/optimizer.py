from typing import Generator


class Optimizer:
    def __init__(self, params: Generator, weight_decay: float = 0, momentum: float = 0):
        self.params = list(params)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocitys = [None for p in self.params]

    def zero_grad(self) -> None:
        """Sets gradients of all parameters to None."""
        for param in self.params:
            param.grad = None

    def step(self, lr: float) -> None:
        """Performs a single optimization step."""
        for ix, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data

            if self.momentum > 0:
                if self.velocitys[ix] is None:
                    self.velocitys[ix] = grad
                else:
                    self.velocitys[ix] = self.momentum * self.velocitys[ix] + grad
                grad = self.velocitys[ix]

            param.data -= lr * grad
