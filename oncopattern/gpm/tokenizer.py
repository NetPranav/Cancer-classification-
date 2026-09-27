"""Byte-level tokenizer with grounding tokens, so text, numbers and locations share one vocabulary.

* Text and numbers are UTF-8 bytes (256 ids): no external vocabulary, no
  out-of-vocabulary words, and numbers are spelled digit by digit, which is
  what makes measured values checkable.
* ``<img>`` marks where image-patch embeddings are inserted; ``<sum>`` closes each
  image with a pooled summary of its patches (max and mean), so the answer can see
  "is anything unusual anywhere" without first learning to search every patch.
* ``<c0>`` .. ``<c{n-1}>`` are quantised coordinates, so the model can *point*:
  a box is four tokens (y0 x0 y1 x1), as in Pix2Seq and Kosmos-2.
* ``<abstain>`` is a first-class output. The model can decline, and the
  verifiable-reward stage pays it a fixed amount (see ``verify.py``).
"""
from __future__ import annotations

import math
import re

SPECIALS = ("<pad>", "<bos>", "<eos>", "<img>", "<sep>", "<abstain>", "<sum>")


class ByteTokenizer:
    def __init__(self, n_coord: int = 64):
        self.n_coord = n_coord
        self.specials = list(SPECIALS) + [f"<c{i}>" for i in range(n_coord)]
        self.ids = {s: 256 + i for i, s in enumerate(self.specials)}
        self.vocab_size = 256 + len(self.specials)
        self.pad_id, self.bos_id, self.eos_id = self.ids["<pad>"], self.ids["<bos>"], self.ids["<eos>"]
        self.img_id, self.sep_id, self.abstain_id = self.ids["<img>"], self.ids["<sep>"], self.ids["<abstain>"]
        self.sum_id = self.ids["<sum>"]
        self.coord0 = self.ids["<c0>"]
        self._re = re.compile("(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in self._re.split(text):
            if not part:
                continue
            if part in self.ids:
                out.append(self.ids[part])
            else:
                out.extend(part.encode("utf-8"))
        return out

    def decode(self, ids, skip_specials: tuple[str, ...] = ("<pad>", "<bos>", "<eos>")) -> str:
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i < 256:
                buf.append(i)
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            s = self.specials[i - 256]
            if s not in skip_specials:
                out.append(s)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    # ---- grounding helpers -------------------------------------------------
    def coord(self, v: float) -> str:
        return f"<c{min(max(int(v * self.n_coord), 0), self.n_coord - 1)}>"

    def box_text(self, y0: float, x0: float, y1: float, x1: float) -> str:
        """Box in normalised [0, 1] coordinates -> 4 coordinate tokens.

        Min edges are floored and max edges ceiled to bin boundaries, so a box on
        an image of n_coord pixels round-trips exactly (a single cell is ~6 bins wide)."""
        n = self.n_coord
        lo = lambda v: min(max(int(math.floor(v * n + 1e-9)), 0), n - 1)
        hi = lambda v: min(max(int(math.ceil(v * n - 1e-9)) - 1, 0), n - 1)
        return f"<c{lo(y0)}><c{lo(x0)}><c{hi(y1)}><c{hi(x1)}>"

    def parse_box(self, text: str):
        vals = [int(m) for m in re.findall(r"<c(\d+)>", text)]
        if len(vals) < 4:
            return None
        a, b, c, d = vals[:4]
        n = self.n_coord
        return min(a, c) / n, min(b, d) / n, (max(a, c) + 1) / n, (max(b, d) + 1) / n
