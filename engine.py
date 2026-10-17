"""Builds the AI players (full = heuristic + neural net + fuzzy + GA-tuned; basic = heuristic only)."""
import json, os
from evaluator import Evaluator, Params
from search import Searcher
from genetic import load_nn

HERE = os.path.dirname(os.path.abspath(__file__))

def load_ai(depth, full=True):
    params, nn = Params(), None
    if full:
        p = os.path.join(HERE, "best_params.json")
        if os.path.exists(p): params = Params.from_list(list(json.load(open(p))["genes"].values()))
        nn = load_nn()
    return Searcher(Evaluator(params, nn, use_fuzzy=full), depth=depth)
