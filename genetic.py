"""
GENETIC ALGORITHM - evolves the evaluation-function parameters (the 'genes')
  genes      : [knight, bishop, rook, queen values, mobility weight, NN blend weight]
  fitness    : result of a short match (both colours) against a fixed baseline engine
  selection  : tournament (k=3)     crossover : blend (arithmetic) crossover
  mutation   : gaussian noise       elitism   : best 2 survive unchanged
"""
import random, json, os, sys, time, chess
from evaluator import Evaluator, Params
from search import Searcher

BOUNDS = [(290, 360), (290, 360), (450, 560), (820, 980), (0.0, 6.0), (0.1, 0.4)]   # plausible ranges (domain knowledge); NN blend kept in 10-40%
NAMES = ["knight", "bishop", "rook", "queen", "w_mob", "w_nn"]

def load_nn():
    try:
        from neural import NeuralEval
        return NeuralEval.load(os.path.join(os.path.dirname(__file__), "nn_weights.npz"))
    except Exception:
        return None

def material_margin(board):
    v = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum((v.get(p.piece_type, 0)) * (1 if p.color else -1) for p in board.piece_map().values())

def play_game(white, black, opening_seed, max_plies=50):
    """Returns +1 / 0 / -1 (White's view). Unfinished games are adjudicated on material."""
    b = chess.Board(); rng = random.Random(opening_seed)
    for _ in range(4): b.push(rng.choice(list(b.legal_moves)))          # random opening for variety
    for _ in range(max_plies):
        if b.is_game_over(claim_draw=True): break
        mv, _ = (white if b.turn else black).best_move(b); b.push(mv)
    if b.is_checkmate(): return -1 if b.turn else 1, 0
    m = material_margin(b)
    return (1 if m >= 3 else -1 if m <= -3 else 0), m

def fitness(genes, nn, seeds=(11, 12, 13), depth=2):
    me = Searcher(Evaluator(Params.from_list(genes), nn), depth=depth, qdepth=2)
    base = Searcher(Evaluator(Params(), None, use_fuzzy=False), depth=depth, qdepth=2)
    score = 0.0
    for sd in seeds:
        r1, m1 = play_game(me, base, sd);  r2, m2 = play_game(base, me, sd)
        score += (r1 - r2) + 0.02 * (m1 - m2)                           # +result, small material tie-breaker
    return score

def clip(g): return [min(hi, max(lo, x)) for x, (lo, hi) in zip(g, BOUNDS)]

def evolve(pop_size=8, generations=5, mut_rate=0.3, seed=42, out="best_params.json"):
    rnd = random.Random(seed); nn = load_nn()
    default = Params().as_list(); default[5] = 0.2          # seed individual: standard piece values + 20% neural net
    pop = [default] + [[rnd.uniform(lo, hi) for lo, hi in BOUNDS] for _ in range(pop_size - 1)]
    history, cache = [], {}
    def fit_of(g):                                           # deterministic search -> cache fitness (elites are not replayed)
        k = tuple(round(x, 6) for x in g)
        if k not in cache: cache[k] = fitness(g, nn)
        return cache[k]
    for gen in range(generations):
        t0 = time.time()
        fit = [fit_of(g) for g in pop]
        ranked = sorted(zip(fit, pop), key=lambda x: -x[0])
        history.append({"gen": gen, "best": ranked[0][0], "avg": sum(fit) / len(fit), "best_genes": ranked[0][1]})
        print(f"Gen {gen}: best={ranked[0][0]:+.2f} avg={sum(fit)/len(fit):+.2f}  ({time.time()-t0:.0f}s)  "
              + " ".join(f"{n}={v:.1f}" for n, v in zip(NAMES, ranked[0][1])), flush=True)
        if gen == generations - 1: break
        new = [g for _, g in ranked[:2]]                                 # elitism
        def pick():                                                      # tournament selection
            return max(rnd.sample(ranked, 3), key=lambda x: x[0])[1]
        while len(new) < pop_size:
            a, b = pick(), pick(); w = rnd.random()
            child = [w * x + (1 - w) * y for x, y in zip(a, b)]          # blend crossover
            child = [x + rnd.gauss(0, 0.1 * (hi - lo)) if rnd.random() < mut_rate else x
                     for x, (lo, hi) in zip(child, BOUNDS)]              # gaussian mutation
            new.append(clip(child))
        pop = new
    best = max(history, key=lambda h: h["best"])["best_genes"]
    # --- held-out validation: games with NEW openings, so a lucky fitness on the training games cannot win
    v_best, v_def = fitness(best, nn, seeds=(21, 22, 23)), fitness(default, nn, seeds=(21, 22, 23))
    chosen = best if v_best > v_def else default
    print(f"Validation on unseen openings: evolved={v_best:+.2f}  seed-default={v_def:+.2f}  -> using {'evolved' if chosen is best else 'seed-default'} genes", flush=True)
    json.dump({"genes": dict(zip(NAMES, chosen)), "validation": {"evolved": v_best, "default": v_def}, "history": history}, open(out, "w"), indent=2)
    print("Saved", out); return chosen

if __name__ == "__main__":
    evolve(*(int(a) for a in sys.argv[1:3]))
