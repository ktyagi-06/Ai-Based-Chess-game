"""
HEURISTIC EVALUATION  +  NEURAL NET  +  FUZZY ACTIVITY  -> one score (centipawns, White's view)

   score = (1-w_nn) * H(board)  +  w_nn * NN(board)  +  w_mob * activity * mobility_diff
   H  = material + piece-square tables      (hand-crafted heuristic knowledge)
   NN = learned evaluation (neural.py)
   activity = fuzzy output (fuzzy.py)
   w_nn, w_mob, piece values = genes tuned by the Genetic Algorithm (genetic.py)
"""
import chess
from dataclasses import dataclass, field
from fuzzy import activity as fuzzy_activity, game_phase

DEFAULT_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}

# Piece-square tables (White's view, a8..h1 order, i.e. rank 8 first)
PST = {
 chess.PAWN:   [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5,
                0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
 chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30,
                -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
 chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10,
                -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
 chess.ROOK:   [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
                -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
 chess.QUEEN:  [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5,
                0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
 chess.KING:   [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
}
KING_END = [-50,-40,-30,-20,-20,-30,-40,-50, -30,-20,-10,0,0,-10,-20,-30, -30,-10,20,30,30,20,-10,-30, -30,-10,30,40,40,30,-10,-30,
            -30,-10,30,40,40,30,-10,-30, -30,-10,20,30,30,20,-10,-30, -30,-30,0,0,0,0,-30,-30, -50,-30,-30,-30,-30,-30,-30,-50]

def heuristic_eval(board, values=DEFAULT_VALUES):
    """Material + piece-square tables, centipawns, White's perspective."""
    phase = game_phase(board)
    score = 0
    for sq, p in board.piece_map().items():
        if p.color == chess.WHITE:
            idx = (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)
        else:                                            # mirror vertically for Black
            idx = chess.square_rank(sq) * 8 + chess.square_file(sq)
        if p.piece_type == chess.KING:
            pst = phase * PST[chess.KING][idx] + (1 - phase) * KING_END[idx]
            v = 0
        else:
            pst, v = PST[p.piece_type][idx], values[p.piece_type]
        score += (v + pst) if p.color == chess.WHITE else -(v + pst)
    return score

def mobility_diff(board):
    """(# pseudo-legal moves of White) - (# of Black)."""
    turn = board.turn
    board.turn = chess.WHITE; w = board.pseudo_legal_moves.count()
    board.turn = chess.BLACK; b = board.pseudo_legal_moves.count()
    board.turn = turn
    return w - b

@dataclass
class Params:
    """Genes evolved by the Genetic Algorithm."""
    knight: float = 320
    bishop: float = 330
    rook: float = 500
    queen: float = 900
    w_mob: float = 3.0
    w_nn: float = 0.0
    def values(self):
        return {chess.PAWN: 100, chess.KNIGHT: self.knight, chess.BISHOP: self.bishop, chess.ROOK: self.rook, chess.QUEEN: self.queen}
    def as_list(self):  return [self.knight, self.bishop, self.rook, self.queen, self.w_mob, self.w_nn]
    @staticmethod
    def from_list(g):   return Params(*g)

class Evaluator:
    def __init__(self, params=None, nn=None, use_fuzzy=True):
        self.p, self.nn, self.use_fuzzy = params or Params(), nn, use_fuzzy
        self.vals = self.p.values()

    def evaluate(self, board):
        h = heuristic_eval(board, self.vals)
        score = h
        if self.nn is not None and self.p.w_nn > 0:
            score = (1 - self.p.w_nn) * h + self.p.w_nn * self.nn.predict_cp(board)
        act = fuzzy_activity(h, game_phase(board)) if self.use_fuzzy else 0.5
        return score + self.p.w_mob * act * 2 * mobility_diff(board)

    def explain(self, board):
        h = heuristic_eval(board, self.vals)
        nn = self.nn.predict_cp(board) if self.nn is not None else None
        act = fuzzy_activity(h, game_phase(board))
        return {"heuristic": h, "neural_net": nn, "fuzzy_activity": act, "mobility_diff": mobility_diff(board),
                "total": self.evaluate(board)}
