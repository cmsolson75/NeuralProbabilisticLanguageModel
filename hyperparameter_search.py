import random
import torch
import json
from nplm.model import MLP
from nplm.data_preprocessing import load_datasets
from nplm.train_utils import Trainer, Optimizer


def log_params(best_params, file_name):
    with open(file_name, 'w') as bp:
        json.dump(best_params, bp, indent=4)

def random_hyperparameter_search(num_iterations):
    best_perplexity = float('inf')
    best_params = None
    generator = torch.Generator().manual_seed(42)
    
    for i in range(num_iterations):
        hyperparams = {
            'lr_start': random.uniform(0.01, 0.11),
            'lr_end': random.uniform(0.00001, 0.001),
            'h_size': random.randint(200, 1000),
            'context_size': random.randint(3, 12),
            'emb_dim': random.randint(5, 20),
            'momentum': random.uniform(0, 0.91),
            'weight_decay': random.uniform(0.00001, 0.001),
            'batch_size': random.choice([32, 64, 128])
        }
        
        train, dev, _ = load_datasets(hyperparams['context_size'], 'data/names.txt')
        
        # Initialize the model, optimizer, and trainer with the selected hyperparameters
        model = MLP(27, hyperparams['emb_dim'], hyperparams['context_size'], hyperparams['h_size'], generator=generator)
        optim = Optimizer(model.parameters(), weight_decay=hyperparams['weight_decay'], momentum=hyperparams['momentum'])
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(model, optim, loss_fn, batch_size=hyperparams['batch_size'], train_data=train, lr_ramp=(hyperparams['lr_start'], hyperparams['lr_end']))
        
        # Train the model
        trainer.fit(10000)
        
        # Evaluate the model
        current_perplexity = trainer.eval(dev)
        
        # Track the best hyperparameters
        if current_perplexity < best_perplexity:
            best_perplexity = current_perplexity
            best_params = hyperparams
            log_params(best_params, 'best_params.json')
        
        print(f"Iteration {i+1}/{num_iterations}: Perplexity = {current_perplexity}, Best Perplexity = {best_perplexity}")
    
    print(f"Best Hyperparameters: {best_params}")
    return best_params

if __name__ == "__main__":
    RUN_NAME = 'tuned_params.json'
    NUM_ITERATIONS = 2
    best_params = random_hyperparameter_search(num_iterations=NUM_ITERATIONS)
    log_params(best_params, RUN_NAME)