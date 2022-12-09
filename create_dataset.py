import argparse
import os
import chess.pgn


def main():

    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    dataset_folder = args.dataset_folder
    if dataset_folder[-1] != '/':
        dataset_folder += '/'
    pgn_folder = args.pgn_folder
    if pgn_folder[-1] != '/':
        pgn_folder += '/'

    num_games = args.num_games
    print("Preparing to load " + str(num_games) + " games.")
        
    dataset_name = "dataset.txt"
    eot_token = "<|endoftext|>"

    valid_game_count = 0
    total_game_count = 0
    
    with open(dataset_folder + dataset_name, 'a') as f:
        for filename in os.listdir(pgn_folder):
            if not filename.endswith('.pgn'):
                continue
            print("Opening file", filename)
            pgn_file = open(pgn_folder + filename)
            while valid_game_count < num_games:
                try:
                    print("Loading game " + str(total_game_count + 1) + " from file " + filename)
                    next_game = chess.pgn.read_game(pgn_file)
                    if next_game == None:
                        break
                    game_txt = ''
                    for move in next_game.mainline_moves():
                        game_txt += str(move) + ' '
                    game_txt = game_txt[:-1] + eot_token + '\n'
                    f.write(game_txt)
                    valid_game_count += 1
                except:
                    print("Skipping game " + str(total_game_count + 1) + " due to errors")
                total_game_count += 1
            if valid_game_count >= num_games:
                break

    print("Dataset created at " + dataset_folder + dataset_name + ".")


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='data/', help="Location of dataset folder.")
    parser.add_argument('--pgn_folder', type=str, default='data/', help="Location of the pgn game files.")
    parser.add_argument('--num_games', type=int, default=100000, help="Number of chess games to include in the dataset.")

    return parser



if __name__ == "__main__":

    main()






