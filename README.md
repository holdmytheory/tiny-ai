# tiny-ai

**Train your own tiny ChatGPT-style AI in 10 minutes.** Free, in your browser, nothing to install.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holdmytheory/tiny-ai/blob/main/tiny_ai.ipynb)

This is the tiny AI from the Hold My Theory video **How ChatGPT Actually Works (We Built a Tiny One)**.

It learns to write the way ChatGPT does: by guessing what comes next. Ours guesses one letter at a time, and everything it knows comes from six old science books. It has 819,328 dials, and we trained it for exactly ten minutes on an ordinary 4-core computer.

## What ours wrote

We stopped the clock six times during its ten minutes of training and asked it to write something starting with "The ". These are its real answers:

```text
 0:00.0   The l mxd(HQbbxpCQ6Jf;6.yab"QddmOrk4RrOHixmSzX9kt"PoO'WEzM?B00cRST)0r::FFz?
 0:05.5   The lemad s bsthe s fe iy bs the retheOHite finke pos are alerce Th0red plif
 0:29.7   The lamppis, be ow cous wablid. The or Hime fing thos ar cof ple sprrentulif
 1:01.6   The limpoduces. The fer cable difference of acck those in another marke planets
 3:01.2   The limps species are in absolutely or intermediate the planets which yould our
10:00.0   The lengths of the sun is become remote for accuracy, and all certain fixed or
```

At minute ten we asked it to finish "The light of the sun". It wrote:

> **The light of the sun is produced by the seeds of descent.**

Perfect grammar. Total confidence. Complete nonsense. It blended Darwin's book on evolution into the astronomy books. It doesn't know what's true, only what sounds like its books, and big chatbots make things up for the same basic reason. (That was the clearest of eight tries. You can see all eight below.)

## Run it

**In your browser:** click **Open in Colab** above, then press ▶ on each step.

**On your own computer,** with Python installed:

```bash
pip install torch
python tiny_ai.py
```

It uses your graphics card if you have one it can use (NVIDIA, or Apple Silicon on a Mac). Otherwise it runs on the CPU, like ours did.

| Command | What it does |
|---|---|
| `python tiny_ai.py` | Train it for 10 minutes, then type sentences for it to finish |
| `python tiny_ai.py --minutes 3` | A shorter run |
| `python tiny_ai.py --books mybook.txt` | Learn from your own text instead |
| `python tiny_ai.py --bigger` | A bigger AI: 6 layers and 4.8 million dials (best with a GPU) |
| `python tiny_ai.py --ours` | Skip training and talk to the exact AI from the video |

## Talk to the exact AI from the video

[`ours/minute-10.pt`](ours/minute-10.pt) is the AI from the video, frozen at minute ten. Run `python tiny_ai.py --ours`, or Step 4 in the notebook. With dice roll 7 it writes the same sentence, word for word. Rolls 0 to 6 are the other seven tries.

## Can you beat our sentence?

Train your own, then post the best (or worst) line it writes in the comments on the video.

## How it works

1. **A map for letters.** Every letter gets a spot on a map: a list of 128 numbers.
2. **Letters look at each other.** Each letter looks back at the letters before it and decides which ones matter. This is called attention.
3. **Stack it.** Four layers of that, one on top of another.
4. **Guess.** Out comes a guess list for the next letter. It rolls weighted dice to pick one, writes it down, and goes again.
5. **Train.** It reads 64 random chunks of the books, guesses every next letter, checks the answers, and nudges all 819,328 dials a hair toward better guesses. In ten minutes ours did that 3,621 times, reading its shelf about twelve times over.

ChatGPT works on the same idea, with pieces of words instead of letters, and much more of everything:

| | Our tiny AI | GPT-3 (OpenAI, 2020) | Llama 3.1 405B (Meta, 2024) |
|---|---|---|---|
| Dials | 819,328 | 175 billion | 405 billion |
| What it read | 6 books, 2.6 million letters | 300 billion word pieces | over 15 trillion word pieces |
| Computers | 4 CPU cores, 10 minutes | a Microsoft cluster of V100 GPUs | over 16,000 H100 GPUs |

OpenAI hasn't said how big today's ChatGPT models are. After all that reading, chatbots get a second round of training to act like a helpful assistant.

## What's in here

| File | What it is |
|---|---|
| [`tiny_ai.py`](tiny_ai.py) | The whole AI and its training, about 120 lines of code |
| [`tiny_ai.ipynb`](tiny_ai.ipynb) | The same thing as a Colab notebook |
| [`data/corpus.txt`](data/corpus.txt) | The six books, cleaned up: 2.6 million letters |
| [`data/BOOKS.md`](data/BOOKS.md) | Which books, and why they're free to use |
| [`ours/minute-10.pt`](ours/minute-10.pt) | The exact AI from the video, after its ten minutes |

## Credits

- **Books:** six public-domain science books from [Project Gutenberg](https://www.gutenberg.org). See [data/BOOKS.md](data/BOOKS.md).
- **The design:** a small GPT, the same family as ChatGPT, built on the transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). Inspired by Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) and [nanoGPT](https://github.com/karpathy/nanoGPT).
- **Numbers in the table:** GPT-3 from [Brown et al., 2020](https://arxiv.org/abs/2005.14165); Llama 3.1 from [Meta, 2024](https://ai.meta.com/blog/meta-llama-3-1/).
- **License:** MIT, for the code and the trained AI. See [LICENSE](LICENSE).

Made by **Hold My Theory**, a science channel.
