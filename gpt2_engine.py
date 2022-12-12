import chess
import numpy as np
import copy
from lm_scorer.models.gpt2 import GPT2LMScorer


class GPT2_Engine():

    def __init__(self, model_path='model/', beam_width=3, beam_length=3, verbose=False):

        print("Initialising GPT2 chess engine...")
        self.scorer = GPT2LMScorer(model_name=model_path)
        self.board = chess.Board()
        self.moves = ''
        self.beam_width = beam_width
        self.beam_length = beam_length
        self.verbose = verbose


    def find_best_move(self):

        boards = [copy.copy(self.board) for i in range(self.beam_width)]
        moves = [copy.copy(self.moves) for i in range(self.beam_width)]
        for i in range(self.beam_length):
            if i == 0:
                next_moves = [str(next_move) for next_move in list(self.board.legal_moves)]
                scores = self.scorer.sentence_score([self.moves + move for move in next_moves], reduce="prod")
                scores = np.array(scores)
                bw = min(self.beam_width, scores.shape[0])
                best_indices = np.argpartition(scores, -1 * bw)[-1 * bw:]
                best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]
                best_moves = [next_moves[ind] for ind in best_indices]
                for j in range(bw):
                    boards[j].push(chess.Move.from_uci(best_moves[j]))
                    moves[j] += best_moves[j] + ' '
            else:
                next_moves = [[str(next_move) for next_move in list(boards[j].legal_moves)] for j in range(self.beam_width)]
                scores = []
                for j in range(self.beam_width):
                    scores.append(self.scorer.sentence_score([moves[j] + next_move for next_move in next_moves[j]], reduce="prod"))
                max_possible = len(max(scores, key=len))
                scores = np.array([x + [-1] * (max_possible - len(x)) for x in scores])
                scores = scores.flatten()
                bw = min(self.beam_width, (scores >= 0).sum())
                best_indices = np.argpartition(scores, -1 * bw)[-1 * bw:]
                best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]
                temp_boards = []
                temp_moves = []
                for j in range(bw):
                    board = best_indices[j] // max_possible
                    move_index = best_indices[j] % max_possible
                    move = next_moves[board][move_index]
                    temp_moves.append(moves[board] + move + ' ')
                    temp_boards.append(copy.copy(boards[board]))
                    temp_boards[j].push(chess.Move.from_uci(move))
                boards = temp_boards
                moves = temp_moves

        best_move = moves[0].split(' ')[-1 * self.beam_length -1]

        return best_move



    def coords_to_uci(self, prev, next):

        alphs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        x_prev, y_prev = prev
        x_next, y_next = next
        x_prev = alphs[x_prev]
        y_prev = 8 - y_prev
        x_next = alphs[x_next]
        y_next = 8 - y_next

        uci = x_prev + str(y_prev) + x_next + str(y_next)

        return uci


    def uci_to_coords(self, uci):

        alph_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        x_prev = alph_to_num[uci[0]]
        y_prev = 8 - int(uci[1])
        x_next = alph_to_num[uci[2]]
        y_next = 8 - int(uci[3])

        return (x_prev, y_prev), (x_next, y_next)


    def get_next_move(self, prev_pos, next_pos):

        move_uci = self.coords_to_uci(prev_pos, next_pos)
        self.board.push(chess.Move.from_uci(move_uci))
        if self.board.is_game_over():
            print("Game Over.")
            return
        self.moves += move_uci + ' '
        next_move = self.find_best_move()
        self.board.push(chess.Move.from_uci(next_move))
        if self.board.is_game_over():
            print("Game Over.")
            return
        next_move_coords = self.uci_to_coords(next_move)

        return next_move_coords

