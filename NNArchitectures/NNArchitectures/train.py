from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train(epochs, dataloader, network, loss_fn, optimizer, device = "cuda", leave_tqdm=True):
    scaler = GradScaler()

    epoch_losses = []
    for epoch in range(epochs):
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
            
    return epoch_losses

