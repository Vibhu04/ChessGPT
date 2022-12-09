from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, dataset_file = 'data/dataset.txt'):
        super().__init__()
        self.eot = "<|endoftext|>"
        self.chess_file = open(dataset_file, 'r')
        self.chess_games = []
        for line in self.chess_file:
            if line.rstrip():
                self.chess_games.append(line.replace('\n', ''))
                
    def __len__(self):
        return len(self.chess_games)

    def __getitem__(self, item):
        return self.chess_games[item]


