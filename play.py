"""
AI chess.  The board opens in your browser; you can type moves / commands here OR click and drag on the board.

   python play.py                 # You (White) vs AI
   python play.py --black         # you play Black
   python play.py --mode ai       # start in AI-vs-AI mode (switch any time with  'mode ai' / 'mode human')
   python play.py --depth 4       # stronger but slower AI (default 3)
   python play.py --no-browser    # don't auto-open the browser (open the printed address yourself)
   python play.py --no-gui        # text-only board in the terminal
"""
import argparse, sys, threading, time, chess
from engine import load_ai          # re-exported: benchmark.py uses play.load_ai
from game import Game, HELP

UNI = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔", "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

def show(board, flip=False):
    ranks = range(8) if flip else range(7, -1, -1); files = range(7, -1, -1) if flip else range(8)
    print()
    for r in ranks:
        print(f" {r + 1}  " + " ".join(UNI[board.piece_at(chess.square(f, r)).symbol()] if board.piece_at(chess.square(f, r)) else "·" for f in files))
    print("    " + " ".join("hgfedcba" if flip else "abcdefgh")); print()

def main():
    for stream in (sys.stdout, sys.stderr):                       # unicode-safe on Windows consoles
        try: stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser(description="AI chess with a live browser board")
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--black", action="store_true")
    ap.add_argument("--mode", choices=["human", "ai"], default="human"); ap.add_argument("--demo", action="store_true", help="same as --mode ai")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between moves in AI-vs-AI mode")
    ap.add_argument("--no-browser", action="store_true"); ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()
    mode = "ai" if a.demo else a.mode
    print("Loading AI (heuristic search + minimax/alpha-beta + neural net + fuzzy logic + GA-tuned weights) ...")
    game = Game(depth=max(1, min(4, a.depth)), mode=mode, human_color=chess.BLACK if a.black else chess.WHITE, delay=a.delay)
    if a.no_gui:
        def watch():
            seen = -1
            while True:
                if game.version != seen and not game.thinking:
                    seen = game.version
                    with game.cv: b = game.board.copy(); flip = game.mode == "human" and game.human_color == chess.BLACK
                    show(b, flip)
                time.sleep(0.2)
        threading.Thread(target=watch, daemon=True).start()
    else:
        from web import serve
        serve(game, open_browser=not a.no_browser)
    print("\n" + HELP + "\n")
    print("=== " + ("AI vs AI" if mode == "ai" else "You vs AI") + " ===")
    with game.cv: game._prompt(); game.cv.notify_all()
    while True:
        try: line = input()
        except (EOFError, KeyboardInterrupt): print("\nBye!"); return
        if line.strip().lower() in ("quit", "exit", "q"): print("Bye!"); return
        ok, msg = game.command(line)
        if msg: print("  >", msg)

if __name__ == "__main__": main()
