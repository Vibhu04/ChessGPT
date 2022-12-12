
# ChessGPT

A chess engine made using OpenAI's GPT2.




## Installation

Install the requirements with:

```bash
pip install -r requirements.txt
```

    
## Training
### Download archive
Download the large KingBase2018 chess archive with:

```
$ python download.py
```
The following flags can be specified:
```
$ python download.py --help
usage: download.py [-h] [--download_location DOWNLOAD_LOCATION]
                   [--extract_location EXTRACT_LOCATION]

optional arguments:
  -h, --help            show this help message and exit
  --download_location DOWNLOAD_LOCATION
                        Location for downloading the chess archive.
  --extract_location EXTRACT_LOCATION
                        Location for extracting the downloaded files.

```
### Create dataset
Create a .txt dataset file from the .pgn files downloaded previously:

```
$ python create_dataset.py
```
The following flags can be specified:
```
$ python create_dataset.py --help
usage: create_dataset.py [-h] [--dataset_folder DATASET_FOLDER]
                         [--pgn_folder PGN_FOLDER] [--num_games NUM_GAMES]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_folder DATASET_FOLDER
                        Location of dataset folder.
  --pgn_folder PGN_FOLDER
                        Location of the pgn game files.
  --num_games NUM_GAMES
                        Number of chess games to include in the dataset.

```
The dataset will be in the [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface) format and will contain a single game on every line.
### Train the model
Start training the GPT2 model with:
```
$ python train.py --dataset_path='path/to/dataset'
```
The following flags can be specified:
```
$ python train.py --help
usage: train.py [-h] [--dataset_path DATASET_PATH] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--base_lr BASE_LR]
                [--weight_decay WEIGHT_DECAY]
                [--save_model_path SAVE_MODEL_PATH]
                [--save_model_freq SAVE_MODEL_FREQ]
                [--load_existing_model LOAD_EXISTING_MODEL]
                [--val_freq VAL_FREQ] [--train_split TRAIN_SPLIT]
                [--write_logs WRITE_LOGS]
                [--delete_prev_logs DELETE_PREV_LOGS]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the dataset file.
  --epochs EPOCHS       Number of training epochs.
  --batch_size BATCH_SIZE
                        Training batch size.
  --base_lr BASE_LR     Base learning rate.
  --weight_decay WEIGHT_DECAY
                        Value of weight decay for AdamW.
  --save_model_path SAVE_MODEL_PATH
                        Path for saving model.
  --save_model_freq SAVE_MODEL_FREQ
                        Frequency of saving the model.
  --load_existing_model LOAD_EXISTING_MODEL
                        Load an existing model for training.
  --val_freq VAL_FREQ   Frequency of performing validation.
  --train_split TRAIN_SPLIT
                        Ratio of total samples used for training.
  --write_logs WRITE_LOGS
                        Use tensorboard to write logs.
  --delete_prev_logs DELETE_PREV_LOGS
                        Delete the previously stored logs.

```
The optimizer uses a linear scheduler with warmup by default.

I specified the following hyperparameters:
```
epochs = 2
base_lr = 0.001
batch_size = 20
weight_decay = 0.1
```
and let the rest use their default values provided by their respective packages.



## Testing

After having trained the model, the project directory should have a structure similar to:
```
└── ChessGPT
        ├── assets
        ├── data
        ├── model
        │   ├── config.json
        │   ├── merges.txt
        │   ├── pytorch_model.bin
        │   ├── special_tokens_map.json
        │   ├── tokenizer_config.json
        │   └── vocab.json
        ├── runs
        ├── README.md
        ├── create_dataset.py
        ├── dataset.py
        ├── gpt2_engine.py
        ├── gui.py
        ├── play.py
        ├── requirements.txt
        ├── train.py
        └── utils.py
```

Finally, to test the model:
```
$ python play.py --model_path='model/'
```
This will open a chess GUI interface made with pygame and taken from [PasiduPerera](https://pererapm.medium.com/)'s Medium blog [post](https://levelup.gitconnected.com/chess-python-ca4532c7f5a4). I have integrated the GPT2 chess engine with this GUI.

*Note: there could be some issues with the GUI logic that need to be fixed.*
## Results

![](https://github.com/Vibhu04/ChessGPT/blob/main/assets/demo.gif)

My model currently manages to play a decent opening game, but struggles to play sensible moves thereafter. In the future it could be improved upon by a longer/different training method or by integrating it with heuristic approaches to avoid nonsensical moves. 

## Authors

- [Vibhu](https://github.com/Vibhu04)

