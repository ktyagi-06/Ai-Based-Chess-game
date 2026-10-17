"""Evidence for the report/viva: pruning gain, and strength of each AI component vs baselines."""
import random, chess, json, os, time
from evaluator import Evaluator, Params
from search import Searcher
from genetic import load_nn, play_game
from engine import load_ai

print("=== 1. Minimax vs Alpha-Beta (start position + a middlegame position) ===")
pos = {"start": chess.Board(), "middlegame": chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")}
for name, b in pos.items():
    for d in ((2, 3) if name == "start" else (2,)):   # plain minimax at depth 3 in a middlegame takes hours
        row = []
        for prune, order in ((False, False), (True, False), (True, True)):
            s = Searcher(depth=d, use_pruning=prune, use_ordering=order); t = time.time(); m, v = s.best_move(b.copy())
            row.append((s.nodes, round(time.time() - t, 2), b.san(m), round(v, 1)))
        print(f"{name:10s} depth {d}: minimax {row[0][0]:6d} nodes ({row[0][1]}s) | +alpha-beta {row[1][0]:6d} ({row[1][1]}s) | +move ordering {row[2][0]:6d} ({row[2][1]}s) | same value: {row[0][3]==row[1][3]==row[2][3]}")

class RandomPlayer:
    def best_move(self, b): return random.choice(list(b.legal_moves)), 0

def match(a, b, n, label, max_plies=80):
    res = [0, 0, 0]
    for i in range(n):
        w, bl = (a, b) if i % 2 == 0 else (b, a)
        r, _ = play_game(w, bl, 200 + i, max_plies)
        r = r if i % 2 == 0 else -r
        res[0 if r > 0 else 1 if r < 0 else 2] += 1
    print(f"{label}: wins {res[0]}  losses {res[1]}  draws {res[2]}")

print("\n=== 2. Playing strength (10-ply random opening, adjudicated on material after 80 plies) ===")
full, base = load_ai(2, True), load_ai(2, False)
match(base, RandomPlayer(), 4, "Baseline (heuristic, depth 2) vs Random")
match(full, base, 8, "Full AI (heuristic+NN+fuzzy+GA) vs Baseline")
