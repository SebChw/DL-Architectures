from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import torch
from .valid import valid

def train(epochs, dataloader, network, loss_fn, optimizer, device = "cuda", valid_each_epoch = False, valid_loader=None, leave_tqdm=True, save_checkpoint = True, save_folder = "models", model_name = None):
    scaler = GradScaler()

    if model_name is None:
        model_name = type(network).__name__ + ".pt"

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    epoch_losses = []
    best_loss = 100000
    for epoch in range(epochs):
        network.train()
        batch_losses = []
        loop = tqdm(dataloader, leave=leave_tqdm)
        loop.set_description(f"epoch: [{epoch}/{epochs}]")
        for input_, target in loop:
            input_, target = input_.to(device), target.to(device)
            optimizer.zero_grad()

            #Here we allow PyTorch to decide which operations should be done using float16 or float32
            with autocast():
                output = network(input_)
                loss = loss_fn(output, target)

            batch_losses.append(loss.detach().cpu().item())

            #Here we just scale loss by some constant and then run backward so that if we have float16 gradient does not vanish
            scaler.scale(loss).backward()

            scaler.step(optimizer) # This runs optimizer.step() for us with additionally some stuff

            scaler.update() # This updates a constant we multiply gradient with

            loop.set_postfix(loss = batch_losses[-1])

        loss_avg = sum(batch_losses) / len(batch_losses)
        loop.set_postfix(average_loss = loss_avg)
        epoch_losses.append(loss_avg)
       
        if valid_each_epoch and valid_loader is not None:
            valid(valid_loader, network, loss_fn, device=device, leave_tqdm=leave_tqdm)

        if save_checkpoint and epoch_losses[-1] < best_loss:
            best_loss = epoch_losses[-1]

            torch.save(network.state_dict(), os.path.join(save_folder, model_name))
            
    return epoch_losses

