"""Tiny local web server (standard library only): serves ui.html, the piece drawings, game state, and actions."""
import json, os, re, threading, webbrowser, chess, chess.svg
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PIECES = {k: re.sub(r'\sid="[^"]*"', "", v) for k, v in chess.svg.PIECES.items()}   # offline piece artwork

def serve(game, port=0, open_browser=True):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path.startswith("/state"): self._send(json.dumps(game.snapshot()).encode(), "application/json")
            elif self.path.startswith("/pieces"): self._send(json.dumps(PIECES).encode(), "application/json")
            else: self._send(open(os.path.join(HERE, "ui.html"), "rb").read(), "text/html; charset=utf-8")
        def do_POST(self):
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                t = d.get("type"); ok, msg = False, "unknown action"
                if t == "move": ok, msg = game.human_move(d.get("uci", ""))
                elif t == "mode": ok, msg = game.set_mode(d.get("mode"))
                elif t == "new": ok, msg = game.new_game()
                elif t == "undo": ok, msg = game.undo()
                elif t in ("flip", "pause", "resume", "step"): ok, msg = game.control(t)
                elif t == "depth": ok, msg = game.set_depth(d.get("value", 3))
                elif t == "speed": ok, msg = game.set_delay(d.get("value", 1))
                elif t == "color": ok, msg = game.set_color(d.get("value", "white"))
                elif t == "flag": ok, msg = game.set_flag(d.get("name"), d.get("value"))
                out = {"ok": ok, "msg": msg, "state": game.snapshot()}
            except Exception as e:
                out = {"ok": False, "msg": f"error: {e}", "state": game.snapshot()}
            self._send(json.dumps(out).encode(), "application/json")
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Board is live at {url}" + ("   (opening your browser...)" if open_browser else ""))
    if open_browser:
        try: webbrowser.open(url)
        except Exception: pass
    return srv, url
