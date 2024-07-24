from nplm.model import MLP
import torch
import json
import argparse
from nplm.data_preprocessing import load_datasets
from nplm.optimizer import Optimizer
from nplm.trainer import Trainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model trainer")
    parser.add_argument(
        "--config", type=str, help="config path", required=True
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as c:
        config = json.load(c)

    
    generator = torch.Generator().manual_seed(config['generatorSeed'])
    train, dev, test = load_datasets(config['context'], config['dataPath'])
    model = MLP(config['vocab'], config['embeddingSize'], config['context'], config['hiddenSize'], config['weightInitialization'], generator=generator)
    optim = Optimizer(model.parameters(), config['weightDecay'], config['momentum'])

    loss_fn = torch.nn.CrossEntropyLoss()
    lr_decay = config['learningRateDecay']
    trainer = Trainer(model, optim, loss_fn, config['batchSize'], train, lr_ramp = (lr_decay[0], lr_decay[1]))
    trainer.fit(config['epochs'])
    trainer.eval(dev)
    model.save(f"models/{config['runName']}.pt")



