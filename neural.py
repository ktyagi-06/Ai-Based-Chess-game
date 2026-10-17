"""
NEURAL NETWORK (written from scratch with NumPy - no ML library)
Architecture : 773 inputs (768 one-hot piece-on-square + 5 material-difference inputs)  ->  Hidden(16, tanh)  ->  Output(1, tanh)
Learning     : supervised regression, forward pass + BACKPROPAGATION + mini-batch gradient descent (MSE loss)
Training data: random-game positions, labelled with a quiescence-search value (see train()).
"""
import numpy as np, chess, random, os, time

NIN = 773
SCALE_CNT = (8.0, 2.0, 2.0, 2.0, 1.0)             # max count of P,N,B,R,Q -> material inputs lie in [-1, 1]

def features(board):
    x = np.zeros(NIN, dtype=np.float32)
    for sq, p in board.piece_map().items():
        x[(p.piece_type - 1 + (0 if p.color == chess.WHITE else 6)) * 64 + sq] = 1.0
        if p.piece_type != chess.KING:             # explicit material balance: lets the net learn piece values easily
            x[768 + p.piece_type - 1] += (1.0 if p.color == chess.WHITE else -1.0) / SCALE_CNT[p.piece_type - 1]
    return x

class NeuralEval:
    weight_decay = 1e-3
    SCALE = 400.0                                   # target = tanh(score_cp / SCALE)
    def __init__(self, hidden=16, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.05, (NIN, hidden)).astype(np.float32); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = r.normal(0, 0.1, (hidden, 1)).astype(np.float32);    self.b2 = np.zeros(1, np.float32)

    def forward(self, X):
        self.X = X; self.h = np.tanh(X @ self.W1 + self.b1); self.y = np.tanh(self.h @ self.W2 + self.b2)
        return self.y

    def backward(self, t, lr):
        n = len(t)
        dy = 2 * (self.y - t) / n * (1 - self.y ** 2)             # dLoss/d(pre-activation of output)
        dW2 = self.h.T @ dy;  db2 = dy.sum(0)
        dh = (dy @ self.W2.T) * (1 - self.h ** 2)                 # propagate error to hidden layer
        dW1 = self.X.T @ dh;  db1 = dh.sum(0)
        wd = self.weight_decay                                    # L2 regularisation fights over-fitting
        self.W1 -= lr * (dW1 + wd * self.W1); self.b1 -= lr * db1; self.W2 -= lr * (dW2 + wd * self.W2); self.b2 -= lr * db2

    def fit(self, X, t, epochs=80, lr=0.3, batch=64, verbose=True):
        n = len(X)
        for ep in range(epochs):
            idx = np.random.permutation(n); loss = 0
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                loss += float(((self.forward(X[b]) - t[b]) ** 2).sum()); self.backward(t[b], lr)
            if verbose and (ep % 25 == 0 or ep == epochs - 1): print(f"  epoch {ep:3d}  MSE = {loss / n:.5f}")

    def predict_cp(self, board):
        y = float(np.tanh(np.tanh(features(board) @ self.W1 + self.b1) @ self.W2 + self.b2)[0])
        y = max(-0.999, min(0.999, y))
        return self.SCALE * np.arctanh(y)

    def save(self, path="nn_weights.npz"): np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    @staticmethod
    def load(path="nn_weights.npz"):
        d = np.load(path); nn = NeuralEval(d["W1"].shape[1]); nn.W1, nn.b1, nn.W2, nn.b2 = d["W1"], d["b1"], d["W2"], d["b2"]; return nn

def make_dataset(n_positions=3000, seed=1):
    """Random self-play positions labelled by quiescence search over the heuristic evaluator."""
    from search import Searcher
    from evaluator import Evaluator
    rng = random.Random(seed)
    s = Searcher(Evaluator(use_fuzzy=False), depth=1, qdepth=3)
    X, T = [], []
    while len(X) < 2 * n_positions:
        b = chess.Board()
        for _ in range(rng.randint(4, 70)):
            if b.is_game_over(): break
            b.push(rng.choice(list(b.legal_moves)))
        if b.is_game_over(): continue
        _, val = s.best_move(b)
        if abs(val) > 5000: continue
        t = np.tanh(val / NeuralEval.SCALE)
        X.append(features(b)); T.append([t])
        X.append(features(b.mirror())); T.append([-t])          # data augmentation: colour-flipped position, negated score
    return np.array(X), np.array(T, dtype=np.float32)

def train(n_positions=6000, epochs=250, path="nn_weights.npz"):
    t0 = time.time(); print("Generating training data ...")
    if os.path.exists("dataset.npz"): d = np.load("dataset.npz"); X, T = d["X"], d["T"]
    else: X, T = make_dataset(n_positions); np.savez_compressed("dataset.npz", X=X, T=T)
    p = np.random.default_rng(0).permutation(len(X)); X, T = X[p], T[p]      # shuffle (mirror pairs are adjacent)
    k = int(0.9 * len(X)); nn = NeuralEval()
    print(f"Training on {k} positions, validating on {len(X) - k} ...")
    nn.fit(X[:k], T[:k], epochs=epochs, lr=0.2)
    val = float(((nn.forward(X[k:]) - T[k:]) ** 2).mean())
    print(f"Validation MSE = {val:.5f}  (variance of targets = {float(T[k:].var()):.5f})")
    nn.save(path); print(f"Saved {path}  ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    train()
