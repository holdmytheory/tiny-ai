"""
Train your own tiny AI in 10 minutes.

This is the tiny AI from the Hold My Theory video "How ChatGPT Actually Works (We Built a Tiny One)".
It reads six old public-domain science books and learns to write by playing the same guessing game
as ChatGPT: guess what comes next. It guesses one letter at a time. Same size as the one in the
video: 4 layers, 819,328 dials.

    python tiny_ai.py                      train it for 10 minutes, then talk to it
    python tiny_ai.py --minutes 3          a quicker run
    python tiny_ai.py --books mybook.txt   learn from your own text instead
    python tiny_ai.py --bigger             a bigger AI (best with a GPU)
    python tiny_ai.py --ours               skip training and talk to the exact AI from the video
"""
import argparse, math, os, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

REPO = "https://raw.githubusercontent.com/holdmytheory/tiny-ai/main/"
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CHECKINS = [0, 5, 30, 60, 180, 600]   # seconds of training at which we stop the clock and ask it to write


# ---------------------------------------------------------------- the AI
class Block(nn.Module):
    """One layer. Every letter looks back at the letters before it (attention), then thinks it over."""
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2, self.heads = nn.LayerNorm(d), nn.LayerNorm(d), heads
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, 2)
        q, k, v = [z.view(B, T, self.heads, C // self.heads).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # look back, never ahead
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, context=128):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d)          # every letter gets a spot on a map: a list of d numbers
        self.pos = nn.Embedding(context, d)        # plus a sense of where it sits in the sentence
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        for m in self.modules():                   # all the dials start out random
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln(x))             # a score for every possible next letter
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def write(self, idx, n, temperature=0.8, generator=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.context:])
            p = F.softmax(logits[:, -1] / temperature, -1).float().cpu()    # its guess list for the next letter
            # Roll weighted dice to pick one. Every letter runs a race with a random finishing time, and
            # likelier letters run faster. Same as torch.multinomial, written out so the rolls match everywhere.
            race = -torch.log1p(-torch.rand(p.shape, generator=generator, dtype=torch.float64)).float()
            idx = torch.cat([idx, (p / race).argmax(-1, keepdim=True).to(idx.device)], 1)
        return idx


# ---------------------------------------------------------------- helpers
def fetch(path):
    """Use the file from this folder if it's here. Otherwise download it from GitHub, once."""
    local = os.path.join(HERE, path)
    if not os.path.exists(local):
        print(f"Downloading {path} ...")
        os.makedirs(os.path.dirname(local), exist_ok=True)
        urllib.request.urlretrieve(REPO + path, local + ".part")
        os.replace(local + ".part", local)
    return local


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def train(minutes=10.0, bigger=False, books=None, seed=1337):
    device = pick_device()
    torch.manual_seed(seed)
    text = open(books or fetch("data/corpus.txt"), encoding="utf-8", errors="replace").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cfg = dict(d=256, layers=6, heads=8, context=256) if bigger else dict(d=128, layers=4, heads=4, context=128)
    B, T, budget = 64, cfg["context"], minutes * 60
    if len(data) < 10 * T: raise SystemExit("That text is too short to learn from. Try a whole book.")
    model = TinyGPT(len(chars), **cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.05)
    dials = sum(p.numel() for p in model.parameters())
    print(f"{dials:,} dials · {len(text):,} letters to read · training on {device} for {minutes:g} minutes\n")

    def peek(elapsed, step, err):       # stop the clock and ask it to write something starting with "The "
        model.eval()
        g = torch.Generator().manual_seed(42)
        start = torch.tensor([[stoi.get(c, 0) for c in "The "]], device=device)
        out = "".join(chars[i] for i in model.write(start, 160, generator=g)[0].tolist()).replace("\n", " ")
        print(f"[{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}] step {step:>5} · error {err:.2f}\n   {out}\n")
        model.train()

    elapsed, step, err = 0.0, 0, math.log(len(chars))
    todo = [c for c in CHECKINS if c <= budget] + [budget]
    peek(0, 0, err); todo.pop(0)
    while elapsed < budget:
        t0 = time.perf_counter()
        ix = torch.randint(len(data) - T - 1, (B,))               # 64 random chunks of the books
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(device)
        warm, fade = min(1, (step + 1) / 60), 0.5 * (1 + math.cos(math.pi * min(elapsed / budget, 1)))
        for g in opt.param_groups: g["lr"] = 3e-4 + (3e-3 - 3e-4) * warm * fade
        _, loss = model(x, y)                                     # guess every next letter, check the answers
        opt.zero_grad(set_to_none=True); loss.backward()          # work out which way to turn each dial
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                                # and nudge every dial a hair
        l = loss.item()
        elapsed += time.perf_counter() - t0; step += 1
        err = l if step == 1 else 0.98 * err + 0.02 * l
        if todo and elapsed >= todo[0]:
            peek(elapsed, step, err); todo.pop(0)
    return model.eval(), chars, stoi, device


def load_ours():
    """The exact AI from the video, frozen after its 10 minutes of training. It runs on the CPU."""
    ck = torch.load(fetch("ours/minute-10.pt"), map_location="cpu", weights_only=True)
    chars = list(ck["chars"])
    model = TinyGPT(len(chars), **ck["config"]); model.load_state_dict(ck["model"])
    return model.eval(), chars, {c: i for i, c in enumerate(chars)}, "cpu"


def finish(model, chars, stoi, device, start, n=200, temperature=0.8, seed=None, tidy=True):
    """Give it the start of a sentence and let it write n more letters. The same seed gives the same dice rolls."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    start = "".join(c for c in start if c in stoi) or " "
    idx = torch.tensor([[stoi[c] for c in start]], device=device)
    text = "".join(chars[i] for i in model.eval().write(idx, n, temperature, g)[0].tolist())
    if tidy:                                         # end at its last full stop rather than mid-word
        end = max(text.rfind(p, len(start)) for p in ".!?")
        text = text[:end + 1] if end >= 0 else text + "..."
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train your own tiny AI in 10 minutes.")
    ap.add_argument("--minutes", type=float, default=10, help="how long to train (default: 10)")
    ap.add_argument("--books", help="a .txt file to learn from, instead of the six science books")
    ap.add_argument("--bigger", action="store_true", help="a bigger AI, about 4.8 million dials (best with a GPU)")
    ap.add_argument("--ours", action="store_true", help="skip training and talk to the exact AI from the video")
    a = ap.parse_args()
    if a.ours:
        ai = load_ours()
        print("The AI from the video, dice roll 7:\n  " + finish(*ai, "The light of the sun", 110, seed=7) + "\n")
    else:
        ai = train(a.minutes, a.bigger, a.books)
    print("Your turn. Type the start of a sentence, or just press Enter to quit.")
    try:
        while (s := input("> ")):
            print(finish(*ai, s) + "\n")
    except EOFError:
        pass
