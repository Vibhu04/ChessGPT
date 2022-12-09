import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
import warnings
import os
from tqdm import tqdm
import argparse
import utils



def train():

    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = utils.load_dataloaders(dataset_path=args.dataset_path, train_split=args.train_split)
    print("Training:", len(train_loader))
    print("Validation:", len(val_loader))

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    if args.write_logs:
        writer = SummaryWriter('runs')
    if args.delete_prev_logs:
        for root, dirs, files in os.walk('runs'):
            for file in files:
                os.remove(os.path.join(root, file))
        print("Logs deleted")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    

    if args.load_existing_model:
        try:
            model = GPT2LMHeadModel.from_pretrained(args.save_model_path)
            print("Existing model loaded")
        except:
            warnings.warn('No model was found in the model directory that was provided. Using the pretrained model '
                          'provided by HF instead.')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    model = model.to(device)
    optimizer = AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.base_lr)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5000, num_training_steps=-1
        )
    model.train()

    max_len = 1000
    inp_tens = None
    batch_count = 0
    count = 0

    for epoch in range(args.epochs):

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
            loss = outputs[0]
            loss.backward()
            count += 1
            print('[%d, %5d] train loss: %.5f' % (epoch + 1, count, loss.detach().data))
            if args.write_logs:
                writer.add_scalar("train_loss", float(loss.detach().data), count)

            if count % args.batch_size == 0:
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if count % args.save_model_freq == 0:
                model.save_pretrained(args.save_model_path)
                tokenizer.save_pretrained(args.save_model_path)
                print("Model saved")

            if count % args.val_freq == 0:
                validate(model, val_loader, tokenizer, device, count // args.val_freq, writer)
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
            loss = outputs[0]
            counter += 1
            running_loss += loss

            inp_tens = game_tens

        print('[%d       ] validation loss: %.5f' % (batch_count,
                                                     running_loss / counter))
        writer.add_scalar('val loss', running_loss / counter,
                          batch_count)


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/dataset.txt', help="Path to the dataset file.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=20, help="Training batch size.")
    parser.add_argument('--base_lr', type=int, default=0.001, help="Base learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Value of weight decay for AdamW.")
    parser.add_argument('--save_model_path', type=str, default='model/', help="Path for saving model.")
    parser.add_argument('--save_model_freq', type=int, default=500, help="Frequency of saving the model.")
    parser.add_argument('--load_existing_model', type=bool, default=False, help="Load an existing model for training.")
    parser.add_argument('--val_freq', type=int, default=500, help="Frequency of performing validation.")
    parser.add_argument('--train_split', type=float, default=0.99, help="Ratio of total samples used for training.")
    parser.add_argument('--write_logs', type=bool, default=True, help="Use tensorboard to write logs.")
    parser.add_argument('--delete_prev_logs', type=bool, default=True, help="Delete the previously stored logs.")

    return parser




if __name__ == '__main__':

    train()




