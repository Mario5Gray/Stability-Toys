"""Which diffusers warnings block the Hunyuan CUDA acceptance (spec section 8).

Kept free of torch/diffusers imports so the policy is unit-testable on any host
while the acceptance itself stays GPU-gated.

The distinction that matters: diffusers uses "will be ignored" for two unrelated
things. A *config attribute* notice at load time is benign for the validated
Canny checkpoint and explicitly sanctioned. A dropped *runtime kwarg* means the
model silently lost an input it needed — that is how the transformer ended up
denoising without its rotary positional embeddings while every other assertion
in the acceptance still passed.
"""

from __future__ import annotations

import re

# Spec section 8: exactly these two extras are allowed for the validated Tencent
# Canny artifact. Anything else is a new, unexamined incompatibility.
ALLOWED_IGNORED_CONFIG_ATTRS = frozenset({"learn_sigma", "norm_type"})

_CONFIG_ATTR_KEYS = re.compile(r"'([^']+)'\s*:")


def _config_attr_keys(message: str) -> set[str]:
    """Dict keys named in a config-attributes warning.

    Only keys — the message quotes values too (`'norm_type': 'layer_norm'`), and
    counting `layer_norm` as an attribute would reject a sanctioned warning.
    """
    head = message.split("were passed", 1)[0]
    return set(_CONFIG_ATTR_KEYS.findall(head))


def blocking_ignored_warnings(messages: list[str]) -> list[str]:
    """Return the warnings that must fail the acceptance run."""
    blocking: list[str] = []
    for message in messages:
        if "cross_attention_kwargs" in message and "not expected by" in message:
            # A denoiser input was dropped by a substituted attention processor.
            blocking.append(message)
        elif "config attributes" in message and "will be ignored" in message:
            if not _config_attr_keys(message) <= ALLOWED_IGNORED_CONFIG_ATTRS:
                blocking.append(message)
        elif "will be ignored" in message:
            # Unrecognized shape: fail closed rather than assume it is benign.
            blocking.append(message)
    return blocking
