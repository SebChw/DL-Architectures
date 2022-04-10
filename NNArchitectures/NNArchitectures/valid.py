from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import torch
from sklearn.metrics import accuracy_score

def valid(dataloader, network, loss_fn, metric="accuracy", device = "cuda", leave_tqdm=True,):
    network.eval() #! Very important
    
    losses = []
    metrics = []

    loop = tqdm(dataloader, leave=leave_tqdm)
    
    with torch.no_grad():
        for input_, target in loop:
            input_, target = input_.to(device), target.to(device)
        
            output = network(input_)
            loss = loss_fn(output, target)

            predictions = torch.argmax(output, dim=1)

            losses.append(loss.detach().cpu().item())
            metrics.append(accuracy_score(target.detach().cpu().numpy(), predictions.detach().cpu().numpy()))

            loop.set_postfix(loss = losses[-1])
            loop.set_postfix(metric = metrics[-1])

        loss_avg = sum(losses) / len(losses)
        metric_avg = sum(metrics) / len(metrics)

        print(f"Validation  loss: {loss_avg}, Validation {metric}: {metric_avg}")

    return metric_avg