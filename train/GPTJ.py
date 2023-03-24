import wandb 
import torch
from tqdm import tqdm

def single_epoch_train(
    model, 
    train_loader, 
    optimizer, 
    accelerator=False,
    scheduler=False, 
    accumulation_step=4,
    device="cuda"
    ):

    total_loss = 0.0
    for idx, batch in tqdm(enumerate(train_loader)):
        seq_input_ids, seq_attention_mask = (
            batch['seq_input_ids'].to(device),
            batch['seq_attention_mask'].to(device)
        )

        outputs = model(
                        input_ids = seq_input_ids,
                        labels=seq_input_ids,
                        attention_mask=seq_attention_mask)
        loss = outputs.loss / accumulation_step
        total_loss += loss.detach().float()

        accelerator.backward(loss)

        #gradient accumulation
        if (idx % accumulation_step == 0) or((idx + 1) == len(train_loader)): 
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

    train_epoch_loss = total_loss / len(train_loader)
    train_ppl = torch.exp(train_epoch_loss)
    wandb.log({"Train PPL": train_ppl})
