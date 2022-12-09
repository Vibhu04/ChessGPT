import chess
from lm_scorer.models.gpt2 import GPT2LMScorer

class GPT2_Engine():

    def __init__(self, model_path='model/'):

        print("Initialising GPT2 chess engine...")
        self.scorer = GPT2LMScorer(model_name=model_path)
        self.board = chess.Board()
        self.moves = ''


    def find_best_move(self):

        next_moves = []
        for next_move in list(self.board.legal_moves):
            next_moves.append(str(next_move))
        scores = self.scorer.sentence_score([self.moves + move for move in next_moves], reduce="prod")
        best_move = next_moves[scores.index(max(scores))]

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
        self.moves += move_uci + ' '
        next_move = self.find_best_move()
        self.board.push(chess.Move.from_uci(next_move))
        next_move_coords = self.uci_to_coords(next_move)

        return next_move_coords

