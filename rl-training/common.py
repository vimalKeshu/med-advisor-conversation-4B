from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

class JsonlWriter:

    def __init__(self, path: str | Path, *, durable: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = Path(path).open("a", encoding="utf-8")
        self.durable = durable

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()
        if self.durable:
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()

# ---------------------------------------------------------------------------
# scenario cards
# ---------------------------------------------------------------------------

SIM_SYSTEM = """You are role-playing a person talking to a medical AI assistant.
You are the one asking. You are never the assistant.

WHO YOU ARE
  A {role} with {literacy} medical knowledge, currently {affect}.
  What you want: {goal}
  How you talk: {style}

WHAT YOU KNOW AND WILL SAY FREELY
{facts}

WHAT YOU KNOW BUT WILL NOT VOLUNTEER
  Say these only if the assistant asks about that specific thing. Do not
  offer them, and do not hint that you are holding anything back.
{withheld}

RULES
- Write only your next message: 1-3 sentences in your own voice. No narration,
  no labels, no stage directions.
- Never invent clinical facts beyond the lists above. If asked something not
  listed, say you do not know.
- Stay in character even if the assistant is wrong, evasive, or refuses you.
- Never mention that you are role-playing.
- When everything you came to ask has been answered, reply with exactly [[DONE]].
"""

def _bullets(items: dict[str, str]) -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in items.items()) or "  - (nothing)"

@dataclass
class Scenario:
    id: str
    title: str
    domain: str
    max_turns: int
    simulator: dict[str, Any]
    thinking: bool = True
    oracle: dict[str, Any] = field(default_factory=dict)

    @property
    def opening(self) -> str:
        return self.simulator["opening"]

    def sim_prompt(self) -> str:
        return SIM_SYSTEM.format(
            **self.simulator["persona"],
            facts=_bullets(self.simulator["facts"]),
            withheld=_bullets(self.simulator.get("withheld") or {}),
        )

def load_scenarios(cfg: dict, root: Path) -> list[Scenario]:
    settings = cfg["scenarios"]
    cards = [card for f in settings["files"] for card in read_jsonl(root / f)]
    if settings.get("shuffle", True):
        random.Random(settings["shuffle_seed"]).shuffle(cards)
    for card in cards:
        if "thinking" not in card:
            if "thinking" not in settings:
                raise ValueError(
                    f"Scenario {card.get('id', '<unknown>')!r} has no thinking mode; "
                    "set its thinking field or scenarios.thinking in the config"
                )
            card["thinking"] = settings["thinking"]
    return [
        Scenario(
            id=c["id"],
            title=c["title"],
            domain=c["domain"],
            max_turns=int(c["max_turns"]),
            simulator=c["simulator"],
            thinking=bool(c["thinking"]),
            oracle=c.get("oracle") or {},
        )
        for c in cards
    ]

# ---------------------------------------------------------------------------
# rendering -- the token-id contract
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think>(.*?)</think>", re.S)

def split_think(text: str) -> tuple[str, str]:
    """(answer, reasoning). An unterminated <think> means generation was cut off
    before any answer, so the answer comes back empty and the caller ends the
    conversation."""
    if "<think>" in text and "</think>" not in text:
        return "", text.split("<think>", 1)[1].strip()
    reasoning = "\n".join(b.strip() for b in _THINK.findall(text))
    return _THINK.sub("", text).strip(), reasoning

@dataclass
class Renderer:
    tokenizer: Any
    system_message: str

    @staticmethod
    def suffix(thinking: bool) -> str:
        return (
            "<|im_start|>assistant\n"
            if thinking
            else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def prompt_ids(self, chat: list[dict[str, str]], thinking: bool) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_message}, *chat],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        if not text.endswith(self.suffix(thinking)):
            raise ValueError(f"generation suffix mismatch, got ...{text[-60:]!r}")
        return self.tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# logprobs -- used no-grad in rollout (reference, M3) and with-grad in the trainer
# ---------------------------------------------------------------------------

def _chunk_logprobs(head: Any, hidden: Any, targets: Any, temperature: float) -> Any:
    import torch
    logits = head(hidden).float()
    if temperature != 1.0:
        logits = logits / temperature
    target_logits = logits.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits - logits.logsumexp(dim=-1)

def completion_logprobs(
    model: Any,
    ids: list[int],
    prompt_len: int,
    temperature: float,
    chunk: int = 512,
    grad: bool = False,
) -> Any:
    """
    Return one log-probability per completion token:

        log pi(ids[t] | ids[:t])

    for t = prompt_len ... len(ids)-1.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    device = next(model.parameters()).device
    # Valid sequence but no completion.
    if prompt_len == len(ids):
        return torch.empty(0,dtype=torch.float32,device=device,)
    x = torch.tensor(ids, dtype=torch.long, device=device,).unsqueeze(0)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hidden = model.get_decoder()(input_ids=x, use_cache=False,).last_hidden_state[0]
        # hidden[k] predicts ids[k + 1].
        # First completion token is ids[prompt_len],
        # so its predictor is hidden[prompt_len - 1].
        hidden = hidden[prompt_len - 1 : -1]
        targets = x[0, prompt_len:]
        assert hidden.shape[0] == targets.shape[0]
        head = model.get_output_embeddings()
        pieces = []
        for i in range(0, hidden.shape[0], chunk):
            h = hidden[i : i + chunk]
            y = targets[i : i + chunk]
            if grad:
                logp = checkpoint(_chunk_logprobs, head, h, y, temperature, use_reentrant=False,)
            else:
                logp = _chunk_logprobs(head, h, y, temperature,)
            pieces.append(logp)
        return torch.cat(pieces)




