"""
GAME CONTROLLER - one object owns the game; the terminal AND the browser both control it.
 * modes   : 'human' (you vs AI)  or  'ai' (AI vs AI)
 * a background thread plays the AI's moves and prints/stores the reasoning for each one
"""
import chess, math, threading, time
from engine import load_ai
from explain import describe, format_console
from search import MATE

START_COUNT = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
HELP = """Commands (type them here, or use the buttons in the browser):
  <move>            e4, Nf3, O-O, e8=Q  or  e2e4        (your moves, in 'Play vs AI' mode)
  mode ai | human   switch between AI-vs-AI and You-vs-AI at any time
  new               start a new game          undo      take back your last move
  play white|black  choose your colour        flip      turn the board around
  depth 1-4         AI thinking depth         speed S   seconds between AI-vs-AI moves
  pause | resume | step      control AI-vs-AI           black full|basic   Black's engine in AI-vs-AI
  board on|off      allow moving pieces with the mouse   help | quit"""

class Game:
    def __init__(self, depth=3, mode="human", human_color=chess.WHITE, delay=1.0, say=print):
        self.cv = threading.Condition(threading.RLock())
        self.say = say
        self.depth, self.mode, self.human_color, self.delay = depth, mode, human_color, delay
        self.paused = self.step_req = False
        self.flip = False; self.board_input = True; self.basic_black = False
        self.engines = {"full": load_ai(depth, True), "basic": load_ai(depth, False)}
        self.max_plies = 240
        self.version = 0; self.epoch = 0; self.thinking = False; self.think_who = ""
        self._reset_state()
        threading.Thread(target=self._engine_loop, daemon=True).start()

    # ------------------------------------------------------------------ state helpers
    def _reset_state(self):
        self.board = chess.Board(); self.expl = {}; self.eval_cp = 0

    def _bump(self): self.version += 1

    def _over(self):
        return self.board.is_game_over(claim_draw=True) or len(self.board.move_stack) >= self.max_plies

    def _ai_turn(self):
        if self._over(): return False
        return True if self.mode == "ai" else self.board.turn != self.human_color

    def _engine_for(self, color):
        return self.engines["basic" if (self.mode == "ai" and color == chess.BLACK and self.basic_black) else "full"]

    def _static_eval(self, board):
        if board.is_checkmate(): return -MATE if board.turn == chess.WHITE else MATE
        if board.is_game_over(claim_draw=True): return 0
        return self.engines["full"].ev.evaluate(board.copy(stack=False))

    def result_text(self):
        b = self.board
        if b.is_checkmate(): return "Checkmate - " + ("Black" if b.turn == chess.WHITE else "White") + " wins"
        if b.is_stalemate(): return "Draw - stalemate"
        if b.is_insufficient_material(): return "Draw - insufficient material"
        if b.can_claim_threefold_repetition(): return "Draw - threefold repetition"
        if b.can_claim_fifty_moves(): return "Draw - fifty-move rule"
        if len(b.move_stack) >= self.max_plies: return "Draw - move limit reached"
        return ""

    def _prompt(self):
        if self.mode == "human" and not self._over() and self.board.turn == self.human_color:
            self.say(f"\nYour move ({'White' if self.human_color else 'Black'}) - type e.g. e4, Nf3, e2e4  (or 'help'):")

    def _announce_end(self):
        if self._over(): self.say("\n*** GAME OVER: " + self.result_text() + " ***   (type 'new' for another game)")

    # ------------------------------------------------------------------ engine thread
    def _engine_loop(self):
        while True:
            with self.cv:
                while not (self._ai_turn() and (not self.paused or self.step_req)):
                    self.cv.wait(0.3)
                self.step_req = False
                before, epoch = self.board.copy(stack=False), self.epoch
                engine = self._engine_for(before.turn); engine.depth = self.depth
                self.thinking, self.think_who = True, ("White" if before.turn else "Black"); self._bump()
            try:
                move, val = engine.best_move(before.copy(stack=False))
                ex = describe(before, move, val, engine, self.depth)
            except Exception as e:                                     # never let the thread die silently
                self.say(f"[engine error] {e!r}"); time.sleep(1); 
                with self.cv: self.thinking = False; self.paused = True; self._bump()
                continue
            with self.cv:
                self.thinking = False
                if epoch != self.epoch:                                # game changed while thinking -> discard
                    self._bump(); continue
                self.board.push(move); ply = len(self.board.move_stack)
                self.expl[ply] = ex; self.eval_cp = max(-MATE, min(MATE, val))
                self._bump(); self.say(format_console(ex)); self._announce_end(); self._prompt()
                self.cv.notify_all()
                if self.mode == "ai":                                  # pause between AI-vs-AI moves (interruptible)
                    end = time.time() + self.delay
                    while time.time() < end and epoch == self.epoch and not self.step_req and self.mode == "ai":
                        self.cv.wait(0.1)

    # ------------------------------------------------------------------ actions (browser + terminal)
    def parse(self, text):
        b = self.board
        try: return b.parse_san(text)
        except ValueError: pass
        try:
            m = chess.Move.from_uci(text.lower())
            p = b.piece_at(m.from_square)
            if p and p.piece_type == chess.PAWN and chess.square_rank(m.to_square) in (0, 7) and not m.promotion:
                m = chess.Move(m.from_square, m.to_square, chess.QUEEN)
            if m in b.legal_moves: return m
        except ValueError: pass
        return None

    def human_move(self, text):
        with self.cv:
            if self._over(): return False, "The game is over - start a new game."
            if self.mode == "ai": return False, "AI vs AI mode is on - switch to 'Play vs AI' to move."
            if self.board.turn != self.human_color: return False, "It is the AI's turn."
            mv = self.parse(text.strip())
            if mv is None: return False, f"'{text}' is not a legal move here."
            san = self.board.san(mv); self.board.push(mv)
            self.eval_cp = self._static_eval(self.board); self._bump()
            self.say(f"You played {san}"); self._announce_end(); self.cv.notify_all()
            return True, ""

    def new_game(self):
        with self.cv:
            self.epoch += 1; self._reset_state(); self.paused = False; self.thinking = False; self._bump()
            self.say("\n=== New game ===  mode: " + ("AI vs AI" if self.mode == "ai" else "You vs AI")); self._prompt(); self.cv.notify_all()
        return True, ""

    def undo(self):
        with self.cv:
            if not self.board.move_stack: return False, "Nothing to undo."
            self.epoch += 1
            self._pop()
            if self.mode == "human":                       # also take back the AI reply, so it is your turn again
                while self.board.move_stack and self.board.turn != self.human_color: self._pop()
            else: self.paused = True
            self.thinking = False; self.eval_cp = self._static_eval(self.board); self._bump()
            self.say("Move taken back."); self._prompt(); self.cv.notify_all()
        return True, ""

    def _pop(self):
        self.expl.pop(len(self.board.move_stack), None); self.board.pop()

    def set_mode(self, mode):
        if mode not in ("human", "ai"): return False, "mode must be 'human' or 'ai'"
        with self.cv:
            if mode == self.mode: return True, ""
            self.epoch += 1; self.mode = mode; self.paused = False; self.thinking = False; self._bump()
            self.say("\n=== Mode: " + ("AI vs AI (both sides played by the engine)" if mode == "ai" else "You vs AI") + " ==="); self._prompt(); self.cv.notify_all()
        return True, ""

    def set_depth(self, d):
        d = max(1, min(4, int(d)))
        with self.cv: self.depth = d; self._bump(); self.say(f"AI depth set to {d}" + (" (slow!)" if d == 4 else ""))
        return True, ""

    def set_delay(self, s):
        with self.cv: self.delay = max(0.0, min(5.0, float(s))); self._bump(); self.cv.notify_all()
        return True, ""

    def set_color(self, color):
        with self.cv:
            self.epoch += 1; self.human_color = chess.WHITE if color == "white" else chess.BLACK
            self.thinking = False; self._bump(); self.say(f"You now play {color.capitalize()}."); self._prompt(); self.cv.notify_all()
        return True, ""

    def control(self, what):
        with self.cv:
            if what == "flip": self.flip = not self.flip
            elif what == "pause": self.paused = True; self.say("Paused.")
            elif what == "resume": self.paused = False; self.say("Resumed."); self.cv.notify_all()
            elif what == "step": self.step_req = True; self.cv.notify_all()
            self._bump()
        return True, ""

    def set_flag(self, name, value):
        with self.cv:
            if name == "board_input": self.board_input = bool(value)
            elif name == "basic_black": self.basic_black = bool(value); self.say("Black engine: " + ("basic (heuristic only)" if value else "full AI"))
            self._bump()
        return True, ""

    # ------------------------------------------------------------------ one entry point for typed text
    def command(self, line):
        t = line.strip()
        if not t: return True, ""
        w = t.lower().split()
        try:
            if w[0] in ("help", "?"): self.say(HELP); return True, ""
            if w[0] == "mode" and len(w) > 1: return self.set_mode({"ai": "ai", "aivsai": "ai", "human": "human", "you": "human", "vs": "human"}.get(w[1], w[1]))
            if w[0] == "new": return self.new_game()
            if w[0] == "undo": return self.undo()
            if w[0] == "flip": return self.control("flip")
            if w[0] in ("pause", "resume", "step"): return self.control(w[0])
            if w[0] == "depth" and len(w) > 1: return self.set_depth(w[1])
            if w[0] == "speed" and len(w) > 1: return self.set_delay(w[1])
            if w[0] == "play" and len(w) > 1 and w[1] in ("white", "black"): return self.set_color(w[1])
            if w[0] == "black" and len(w) > 1 and w[1] in ("full", "basic"): return self.set_flag("basic_black", w[1] == "basic")
            if w[0] == "board" and len(w) > 1 and w[1] in ("on", "off"): return self.set_flag("board_input", w[1] == "on")
        except ValueError: return False, "That command needs a number."
        return self.human_move(t)

    # ------------------------------------------------------------------ what the browser sees
    def snapshot(self):
        with self.cv:
            b = self.board; over = self._over(); human_turn = self.mode == "human" and not over and b.turn == self.human_color
            legal = [m.uci() for m in b.legal_moves] if (human_turn and self.board_input) else []
            tmp, sans = chess.Board(), []
            for m in b.move_stack: sans.append(tmp.san(m)); tmp.push(m)
            cap = {"white": [], "black": []}; bal = 0
            for col, key in ((chess.BLACK, "white"), (chess.WHITE, "black")):          # pieces captured BY key side
                for pt, n in START_COUNT.items():
                    miss = max(0, n - len(b.pieces(pt, col))); cap[key] += [chess.Piece(pt, col).symbol()] * miss
            for pt, v in VAL.items(): bal += v * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)))
            ev = self.eval_cp; bar = 100.0 if ev >= MATE - 1000 else 0.0 if ev <= -MATE + 1000 else 50 + 50 * math.tanh(ev / 600.0)
            lab = (f"M{(MATE - abs(ev) + 1) // 2}" if abs(ev) > MATE - 1000 else f"{ev / 100:+.1f}")
            if abs(ev) > MATE - 1000 and ev < 0: lab = "-" + lab
            if over: status = self.result_text()
            elif self.mode == "human": status = "Your move" + (" - CHECK" if b.is_check() else "") if human_turn else "AI is thinking..."
            else: status = ("Paused" if self.paused else ("White" if b.turn else "Black") + (" AI is thinking..." if self.thinking else " AI to move"))
            last = max(self.expl) if self.expl else None
            orient = "black" if (self.flip != (self.mode == "human" and self.human_color == chess.BLACK)) else "white"
            return {"version": self.version, "fen": b.fen(), "orientation": orient, "turn": "white" if b.turn else "black",
                    "legal": legal, "lastMove": ([chess.square_name(b.peek().from_square), chess.square_name(b.peek().to_square)] if b.move_stack else None),
                    "check": chess.square_name(b.king(b.turn)) if b.is_check() else None, "status": status, "over": over,
                    "result": self.result_text() if over else "", "mode": self.mode, "humanColor": "white" if self.human_color else "black",
                    "depth": self.depth, "delay": self.delay, "paused": self.paused, "thinking": self.thinking, "thinkWho": self.think_who,
                    "evalCp": ev, "bar": round(bar, 1), "evalLabel": lab, "moves": sans, "captured": cap, "balance": bal,
                    "allowBoard": self.board_input, "basicBlack": self.basic_black, "humanTurn": human_turn,
                    "explanation": ({"ply": last, **self.expl[last]} if last else None)}
