"""
FUZZY LOGIC  (Mamdani-style fuzzy inference system)
---------------------------------------------------
Inputs  : material balance (centipawns, White's view)  and  game phase (0 = endgame .. 1 = opening)
Output  : 'activity'  in [0,1]  -> how strongly the evaluation rewards piece mobility / initiative.
Steps   : 1) Fuzzification  2) Rule evaluation (AND = min)  3) Aggregation (OR = max)  4) Defuzzification (centroid)
"""
import chess

def tri(x, a, b, c):
    """Triangular membership function (a<=b<=c)."""
    if x <= a or x >= c:
        return 1.0 if (x == b) else 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)

def shoulder_left(x, b, c):   # 1 up to b, then falls to 0 at c
    return 1.0 if x <= b else (0.0 if x >= c else (c - x) / (c - b))

def shoulder_right(x, a, b):  # 0 up to a, rises to 1 at b
    return 0.0 if x <= a else (1.0 if x >= b else (x - a) / (b - a))

# ---- fuzzification -------------------------------------------------------
def material_sets(m):
    return {"losing":  shoulder_left(m, -300, -50),
            "equal":   tri(m, -200, 0, 200),
            "winning": shoulder_right(m, 50, 300)}

def phase_sets(p):
    return {"endgame":    shoulder_left(p, 0.2, 0.5),
            "middlegame": tri(p, 0.2, 0.5, 0.8),
            "opening":    shoulder_right(p, 0.5, 0.8)}

# output sets over universe [0,1]
def out_mu(name, x):
    return {"low": shoulder_left(x, 0.0, 0.5), "medium": tri(x, 0.0, 0.5, 1.0), "high": shoulder_right(x, 0.5, 1.0)}[name]

# ---- rule base: (material, phase) -> activity ----------------------------
RULES = [
    ("any",     "opening",    "high"),    # develop pieces, fight for the centre
    ("equal",   "middlegame", "medium"),
    ("winning", "middlegame", "low"),     # ahead: play safe, trade down
    ("losing",  "middlegame", "high"),    # behind: create complications
    ("winning", "endgame",    "medium"),  # ahead: activate king/rooks to convert
    ("equal",   "endgame",    "medium"),
    ("losing",  "endgame",    "high"),
]

def game_phase(board):
    """1.0 = full material (opening), 0.0 = bare kings, from non-pawn material."""
    w = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    total = sum(w[p.piece_type] for p in board.piece_map().values() if p.piece_type in w)
    return min(1.0, total / 24.0)

def activity(material_cp, phase, return_trace=False):
    mat, ph = material_sets(material_cp), phase_sets(phase)
    firing = {"low": 0.0, "medium": 0.0, "high": 0.0}
    for m_name, p_name, out in RULES:
        strength = ph[p_name] if m_name == "any" else min(mat[m_name], ph[p_name])  # AND = min
        firing[out] = max(firing[out], strength)                                    # aggregation = max
    num = den = 0.0
    for i in range(101):                                                            # centroid defuzzification
        x = i / 100
        mu = max(min(firing[k], out_mu(k, x)) for k in firing)                      # clip each set, then union
        num += x * mu; den += mu
    result = num / den if den else 0.5
    return (result, mat, ph, firing) if return_trace else result
