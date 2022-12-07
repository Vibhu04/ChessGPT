import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
import os
from tqdm import tqdm
from lib import utils



def train(
    epochs=10, 
    base_lr=0.001, 
    save_model_name='model/model.pth', 
    save_model_freq=500, 
    load_existing_model=True,
    val_freq=500, 
    write_logs=True, 
    delete_prev_logs=True
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = utils.load_dataloaders(train_split=0.99)
    print("Training:", len(train_loader))
    print("Validation:", len(val_loader))
    if write_logs:
        writer = SummaryWriter('runs')
    if delete_prev_logs:
        for root, dirs, files in os.walk('runs'):
            for file in files:
                os.remove(os.path.join(root, file))
        print("Logs deleted")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if load_existing_model:
        state_dict = torch.load(save_model_name, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("Existing model loaded")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), weight_decay=0.1, lr=base_lr) 
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5000, num_training_steps=-1
        )
    model.train()

    max_len = 1000
    inp_tens = None
    batch_size = 20
    batch_count = 0
    count = 0

    for epoch in range(epochs):

        print(f"EPOCH {epoch} started" + '=' * 30)
        for train_counter, train_batch in enumerate(train_loader, 0):

            game_tens = torch.tensor(tokenizer.encode(train_batch[0])).unsqueeze(0).to(device)
            if game_tens.size()[1] > max_len:
                continue
            if inp_tens == None:
                inp_tens = game_tens
                continue
            elif inp_tens.size()[1] + game_tens.size()[1] <= max_len:
                inp_tens = torch.cat((inp_tens, game_tens), 1)
                continue

            outputs = model(inp_tens, labels=inp_tens)
            loss, logits = outputs[:2]                        
            loss.backward()
            count += 1
            print('[%d, %5d] train loss: %.5f' % (epoch + 1, count, loss.detach().data))
            if write_logs:
                writer.add_scalar("train_loss", float(loss.detach().data), count)

            if count % batch_size == 0:
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()
                print("Stepping")

            if count % save_model_freq == 0:
                torch.save(model.state_dict(), save_model_name)
                print("Model saved")

            if count % val_freq == 0:
                validate(model, val_loader, tokenizer, device, count // val_freq, writer)
                model.train()

            inp_tens = game_tens


def validate(model, val_loader, tokenizer, device, batch_count, writer):

    model.eval()
    max_len = 1000
    counter = 0
    inp_tens = None
    with torch.no_grad():
        running_loss = 0
        for val_counter, val_batch in enumerate(tqdm(val_loader), 0):

            game_tens = torch.tensor(tokenizer.encode(val_batch[0])).unsqueeze(0).to(device)
            if game_tens.size()[1] > max_len:
                continue
            if inp_tens == None:
                inp_tens = game_tens
                continue
            elif inp_tens.size()[1] + game_tens.size()[1] <= max_len:
                inp_tens = torch.cat((inp_tens, game_tens), 1)
                continue

            outputs = model(inp_tens, labels=inp_tens)
            loss, logits = outputs[:2]                        
            counter += 1
            running_loss += loss

            inp_tens = game_tens

        print('[%d       ] validation loss: %.5f' % (batch_count,
                                                     running_loss / counter))
        writer.add_scalar('val loss', running_loss / counter,
                          batch_count)



if __name__ == '__main__':

    train()




