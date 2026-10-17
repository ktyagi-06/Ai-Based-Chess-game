"""
Turns an AI move into SIMPLE, MOVE-SPECIFIC reasoning (shown in the terminal and in the browser).
Everything stated here is checked against the actual position - nothing is canned.
"""
import chess, textwrap
from evaluator import heuristic_eval
from fuzzy import activity as fuzzy_activity, game_phase
from search import MATE

PN = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
PV = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5}

def _box(sq):                       # the 4x4 central area c3..f6
    return 2 <= chess.square_file(sq) <= 5 and 2 <= chess.square_rank(sq) <= 5

def move_reasons(board, move):
    """Why this move? Returns up to 3 short phrases, most important first."""
    me, opp = board.turn, not board.turn
    pt = board.piece_type_at(move.from_square)
    to, frm, name = move.to_square, move.from_square, chess.square_name(move.to_square)
    after = board.copy(stack=False); after.push(move)
    phase = game_phase(board)
    home = 0 if me == chess.WHITE else 7
    r = []
    if after.is_checkmate():
        return ["delivers checkmate - the game is won"]
    if board.is_castling(move):
        side = "kingside" if chess.square_file(to) > chess.square_file(frm) else "queenside"
        r.append(f"castles {side}: the king gets safe and the rook joins the game")
    if board.is_capture(move):
        ep = board.is_en_passant(move)
        vt = chess.PAWN if ep else board.piece_type_at(to)
        vname = f"{PN[vt]} on {chess.square_name(chess.square(chess.square_file(to), chess.square_rank(frm))) if ep else name}"
        if not after.is_attacked_by(opp, to):
            r.append(f"wins the {vname} for free - nothing can recapture")
        elif PV[vt] > PV[pt]:
            r.append(f"takes the {vname} with a cheaper piece, so it comes out ahead even if recaptured")
        elif PV[vt] == PV[pt]:
            r.append(f"trades its {PN[pt]} for the {vname}")
        else:
            r.append(f"takes the {vname}; the {PN[pt]} can be recaptured, but the search finds the exchange worthwhile")
    if move.promotion:
        r.append(f"promotes the pawn to a {PN[move.promotion]}")
    if after.is_check():
        r.append("gives check, forcing the opponent to respond")
    if (not board.is_capture(move) and not board.is_castling(move)
            and board.is_attacked_by(opp, frm) and not after.is_attacked_by(opp, to)):
        r.append("steps the king out of danger" if pt == chess.KING else f"moves the {PN[pt]} out of attack")
    # new threats created by the moved piece
    best_t = None
    for sq in after.attacks(to):
        p = after.piece_at(sq)
        if p and p.color == opp and p.piece_type != chess.KING and sq not in board.attacks(frm):
            undefended = not after.is_attacked_by(opp, sq)
            if undefended or PV[p.piece_type] > PV[pt]:
                key = (PV[p.piece_type], undefended)
                if best_t is None or key > best_t[0]: best_t = (key, sq, p.piece_type, undefended)
    if best_t:
        _, sq, t, und = best_t
        r.append(f"attacks the {'undefended ' if und else ''}{PN[t]} on {chess.square_name(sq)}")
    quiet = not board.is_capture(move)
    if quiet and pt in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(frm) == home and phase >= 0.6:
        r.append(f"develops the {PN[pt]} off the back rank" + (" to the centre" if _box(to) else ""))
    elif quiet and pt == chess.PAWN and to in CENTRE:
        r.append("stakes a claim in the centre")
    elif quiet and pt in (chess.KNIGHT, chess.BISHOP, chess.QUEEN) and _box(to) and not _box(frm):
        r.append(f"centralises the {PN[pt]}")
    elif quiet and pt == chess.ROOK and not any(board.piece_type_at(chess.square(chess.square_file(to), k)) == chess.PAWN for k in range(8)):
        r.append("puts the rook on an open file")
    elif quiet and pt == chess.PAWN and phase < 0.35:
        r.append("pushes the pawn toward promotion")
    elif quiet and pt == chess.KING and phase < 0.35 and not board.is_castling(move):
        r.append("activates the king, which is strong in the endgame")
    if not r:
        d = (heuristic_eval(after) - heuristic_eval(board)) * (1 if me == chess.WHITE else -1)
        if d >= 10: r.append(f"improves piece placement (+{d:.0f} centipawns on the piece-square tables)")
        else: r.append("a quiet move: nothing to win by force, so it picks the line whose positions score best after looking ahead")
    return r[:3]

def _score_words(val, mover):
    v = val if mover == chess.WHITE else -val
    if abs(val) > MATE - 1000:
        n = (MATE - abs(val) + 1) // 2
        return (f"sees a forced checkmate in {n}" if v > 0 else f"sees it can be checkmated in {n} if the opponent plays perfectly")
    a = abs(v)
    if a < 30: word = "the position is roughly equal"
    elif a < 100: word = "slightly better for the AI" if v > 0 else "slightly worse for the AI"
    elif a < 300: word = "clearly better for the AI" if v > 0 else "clearly worse for the AI"
    else: word = "winning for the AI" if v > 0 else "losing for the AI"
    return f"{word} ({val / 100:+.2f} pawns, White's view)"

def _fuzzy_words(h, phase):
    act, mat, ph, _ = fuzzy_activity(h, phase, True)
    m = max(mat, key=mat.get); p = max(ph, key=ph.get)
    mtxt = {"equal": "equal material", "winning": "White ahead", "losing": "Black ahead"}[m]
    lvl = "low" if act < 0.35 else "medium" if act < 0.65 else "high"
    return f"fuzzy logic: {p} + {mtxt} -> {lvl} piece activity ({act:.2f})"

def describe(board, move, val, searcher, depth=None):
    """board = position BEFORE the move.  Returns dict(who, san, lines=[(label,text)...])."""
    mover = board.turn; ev = searcher.ev
    who = "White" if mover == chess.WHITE else "Black"
    san = board.san(move)
    why = move_reasons(board, move)
    if len(searcher.root_scores) == 1: why = ["the only legal move"]
    lines = [("Why", "; ".join(why))]
    lines.append(("Think", f"looked {depth or searcher.depth} half-moves ahead (each side's turn counts as one, plus capture sequences) and searched {searcher.nodes:,} positions; "
                           f"alpha-beta made {searcher.cutoffs:,} cut-offs, ignoring whole branches that could not change the answer"))
    tmp, seq = board.copy(stack=False), []
    for m in searcher.pv[:4]:
        if m in tmp.legal_moves: seq.append(tmp.san(m)); tmp.push(m)
    if len(seq) > 1: lines.append(("Plan", "expects  " + "  ".join(seq)))
    after = board.copy(stack=False); after.push(move)
    e = ev.explain(after)
    parts = [f"heuristic (material + placement) {e['heuristic'] / 100:+.2f}"]
    if ev.nn is not None and ev.p.w_nn > 0: parts.append(f"neural net {e['neural_net'] / 100:+.2f} (trusted {ev.p.w_nn * 100:.0f}%)")
    else: parts.append("neural net off")
    if ev.use_fuzzy: parts.append(_fuzzy_words(e["heuristic"], game_phase(after)))
    lines.append(("Score", _score_words(val, mover)))
    lines.append(("Based on", "; ".join(parts)))
    sgn = 1 if mover == chess.WHITE else -1
    alts = sorted([(m, v) for m, v in searcher.root_scores if m != move], key=lambda x: -sgn * x[1])[:2]
    if alts and abs(val) < MATE - 1000:
        lines.append(("Others", "; ".join(f"{board.san(m)} ({'about equal, ' if abs(v - val) < 1 else 'no better than '}{sgn * v / 100:+.2f})" for m, v in alts) + "  [scores from the AI's own view]"))
    return {"who": who, "san": san, "number": board.fullmove_number, "lines": lines}

def format_console(ex, width=100):
    head = f"Move {ex['number']} | AI ({ex['who']}) plays {ex['san']}"
    out = ["-" * 70, head]
    for label, text in ex["lines"]:
        out.append(textwrap.fill(text, width=width, initial_indent=f"  {label:<9}: ", subsequent_indent=" " * 13))
    return "\n".join(out)
