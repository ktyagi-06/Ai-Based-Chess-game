"""
HEURISTIC SEARCH (move ordering, quiescence) + ADVERSARIAL SEARCH (Minimax with Alpha-Beta pruning)
White is the MAX player, Black is the MIN player; scores are from White's view.
"""
import chess, random
from evaluator import Evaluator

MATE = 100000
ORDER_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

def order_moves(board, moves):
    """Heuristic move ordering: MVV-LVA (Most Valuable Victim - Least Valuable Attacker), promotions, checks.
       Good ordering makes alpha-beta prune far more branches."""
    def key(m):
        s = 0
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or chess.PAWN      # en-passant -> pawn
            s += 10000 + 10 * ORDER_VAL[victim] - ORDER_VAL[board.piece_type_at(m.from_square)]
        if m.promotion: s += 9000 + m.promotion
        if board.gives_check(m): s += 50
        return s
    return sorted(moves, key=key, reverse=True)

class Searcher:
    def __init__(self, evaluator=None, depth=3, use_pruning=True, use_ordering=True, qdepth=4):
        self.ev = evaluator or Evaluator()
        self.depth, self.use_pruning, self.use_ordering, self.qdepth = depth, use_pruning, use_ordering, qdepth
        self.nodes = self.cutoffs = 0
        self.pvt = [[] for _ in range(80)]        # principal-variation table (the line the AI expects)
        self.pv, self.root_scores = [], []

    # ---- public API -------------------------------------------------------
    def best_move(self, board):
        self.nodes = self.cutoffs = 0
        self.pv, self.root_scores = [], []
        maximizing = board.turn == chess.WHITE
        moves = list(board.legal_moves)
        if self.use_ordering: moves = order_moves(board, moves)
        best, best_val = None, -float("inf") if maximizing else float("inf")
        alpha, beta = -float("inf"), float("inf")
        for m in moves:
            board.push(m)
            val = self.minimax(board, self.depth - 1, alpha, beta, not maximizing, 1)
            board.pop()
            self.root_scores.append((m, val))
            if (maximizing and val > best_val) or (not maximizing and val < best_val) or best is None:
                best, best_val = m, val
                self.pv = [m] + self.pvt[1]
            if self.use_pruning:
                if maximizing: alpha = max(alpha, best_val)
                else:          beta = min(beta, best_val)
        return best, best_val

    # ---- Minimax with Alpha-Beta -----------------------------------------
    def minimax(self, board, depth, alpha, beta, maximizing, ply):
        self.nodes += 1
        self.pvt[ply] = []
        if depth == 0:
            return self.quiesce(board, alpha, beta, maximizing, self.qdepth, ply)
        moves = list(board.legal_moves)
        if not moves:                                              # terminal node
            if board.is_check(): return (-MATE + ply) if maximizing else (MATE - ply)
            return 0
        if board.is_insufficient_material(): return 0
        if self.use_ordering: moves = order_moves(board, moves)
        if maximizing:
            value = -float("inf")
            for m in moves:
                board.push(m)
                v = self.minimax(board, depth - 1, alpha, beta, False, ply + 1)
                board.pop()
                if v > value: value = v; self.pvt[ply] = [m] + self.pvt[ply + 1]
                alpha = max(alpha, value)
                if self.use_pruning and alpha >= beta:
                    self.cutoffs += 1; break                         # beta cut-off
            return value
        value = float("inf")
        for m in moves:
            board.push(m)
            v = self.minimax(board, depth - 1, alpha, beta, True, ply + 1)
            board.pop()
            if v < value: value = v; self.pvt[ply] = [m] + self.pvt[ply + 1]
            beta = min(beta, value)
            if self.use_pruning and alpha >= beta:
                self.cutoffs += 1; break                             # alpha cut-off
        return value

    # ---- Quiescence search: keep searching captures so we don't evaluate mid-exchange (horizon effect)
    def quiesce(self, board, alpha, beta, maximizing, qdepth, ply):
        self.nodes += 1
        if board.is_check() and not any(board.legal_moves):
            return (-MATE + ply) if maximizing else (MATE - ply)
        stand = self.ev.evaluate(board)
        if qdepth == 0: return stand
        caps = order_moves(board, [m for m in board.legal_moves if board.is_capture(m)])
        if maximizing:
            if stand >= beta and self.use_pruning: return stand
            value = stand; alpha = max(alpha, stand)
            for m in caps:
                board.push(m); value = max(value, self.quiesce(board, alpha, beta, False, qdepth - 1, ply + 1)); board.pop()
                alpha = max(alpha, value)
                if self.use_pruning and alpha >= beta: break
            return value
        if stand <= alpha and self.use_pruning: return stand
        value = stand; beta = min(beta, stand)
        for m in caps:
            board.push(m); value = min(value, self.quiesce(board, alpha, beta, True, qdepth - 1, ply + 1)); board.pop()
            beta = min(beta, value)
            if self.use_pruning and alpha >= beta: break
        return value
