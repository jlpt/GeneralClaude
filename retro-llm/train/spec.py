"""
Shared constants for the "retro-llm" integer transformer.

Everything the consoles do is pure integer math. This file is the single
source of truth for the hyper-parameters and lookup tables; the PyTorch
trainer, the numpy reference and the ROM exporters all import it.
"""
import math

# --- vocabulary: what the NES font can draw ---------------------------------
VOCAB = "\n !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
V = len(VOCAB)            # 36
NEWLINE = 0

# --- model shape ------------------------------------------------------------
D = 32                    # residual width
L = 2                     # layers
H = 4                     # attention heads
HD = D // H               # 8 dims per head
F = 64                    # MLP hidden width
T = 32                    # context window (ring-buffer KV cache)

# --- integer arithmetic knobs ----------------------------------------------
SS = 7                    # q.k dot product >> SS  -> attention score units
ATT_TAU = 8               # one attention-score unit = 1/8 nat
SLOPES = [0, 1, 2, 6]     # ALiBi penalty per head, score units per token of distance
LOGIT_TAU = 8             # one logit unit = 1/8 nat
P_BITS = 7                # attention probabilities are 0..127 (= /128)

# EXP_ATT[d] = 255 * exp(-d / 8), d = max_score - score clamped to 0..255
EXP_ATT = [int(math.floor(255.0 * math.exp(-d / ATT_TAU) + 0.5)) for d in range(256)]


def exp_sample_table(temp):
    """Sampling table for temperature `temp` (logit units are 1/8 nat)."""
    return [int(math.floor(255.0 * math.exp(-d / (LOGIT_TAU * temp)) + 0.5)) for d in range(256)]


TEMPS = [0.5, 0.7, 0.85, 1.0]   # selectable on the consoles, default index 2
DEFAULT_TEMP = 2


def encode(text):
    out = []
    for ch in text.upper():
        i = VOCAB.find(ch)
        if i >= 0:
            out.append(i)
    return out


def decode(ids):
    return "".join(VOCAB[i] for i in ids)
