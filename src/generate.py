# -*- coding: utf-8 -*-
"""Generation utilities for the poetry RNN experiments.

The dataset used by this project places the optional genre token after
``<SOP>``.  The generator follows that concrete interface so evaluation uses
the same prefix layout as training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch


FORM_ID_TO_TOKEN = {
    0: "<5JUE>",
    1: "<7JUE>",
    2: "<5LV>",
    3: "<7LV>",
}

FORM_SPECS = {
    "<5JUE>": (4, 5),
    "<7JUE>": (4, 7),
    "<5LV>": (8, 5),
    "<7LV>": (8, 7),
}

SPECIAL_TOKEN_NAMES = {
    "<PAD>", "<UNK>", "<SOP>", "<EOP>",
    "PAD", "UNK", "SOP", "EOP",
    "<5JUE>", "<7JUE>", "<5LV>", "<7LV>",
}

PUNCTUATION = set("，。！？；、,.!?;:：")


def _is_chinese_char(text: str) -> bool:
    return len(text) == 1 and "\u4e00" <= text <= "\u9fff"


@dataclass
class GenerationRecord:
    strategy: str
    form_token: Optional[str]
    prompt: str
    poem: str


class PoetryGenerator:
    """Autoregressive generator with greedy, top-k, top-p and constrained modes."""

    def __init__(
        self,
        model: torch.nn.Module,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        device: Optional[torch.device | str] = None,
        max_gen_len: int = 200,
        use_form_token: bool = False,
    ) -> None:
        self.model = model
        self.word2idx = word2idx
        self.idx2word = {int(k): v for k, v in idx2word.items()}
        self.device = torch.device(device) if device is not None else self._model_device()
        self.max_gen_len = max_gen_len
        self.use_form_token = use_form_token

        self.pad_id = self._token_id("<PAD>", "PAD", default=0)
        self.unk_id = self._token_id("<UNK>", "UNK", default=1)
        self.sop_id = self._token_id("<SOP>", "SOP", default=2)
        self.eop_id = self._token_id("<EOP>", "EOP", default=3)

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _token_id(self, *names: str, default: Optional[int] = None) -> Optional[int]:
        for name in names:
            if name in self.word2idx:
                return self.word2idx[name]
        return default

    def _form_token(self, form_id: Optional[int] = None, form_token: Optional[str] = None) -> Optional[str]:
        if form_token is not None:
            return form_token
        if form_id is None:
            return None
        return FORM_ID_TO_TOKEN.get(int(form_id))

    def _prefix_ids(self, prompt: str = "", form_token: Optional[str] = None) -> List[int]:
        ids = [self.sop_id]
        if self.use_form_token and form_token in self.word2idx:
            ids.append(self.word2idx[form_token])
        ids.extend(self.word2idx.get(ch, self.unk_id) for ch in prompt if ch not in PUNCTUATION)
        return ids

    def _ids_to_text(self, ids: Iterable[int]) -> str:
        chars: List[str] = []
        for token_id in ids:
            token = self.idx2word.get(int(token_id), "")
            if token in SPECIAL_TOKEN_NAMES:
                continue
            chars.append(token)
        return "".join(chars)

    def _valid_char_mask(self, logits: torch.Tensor, allow_eop: bool = True) -> torch.Tensor:
        masked = logits.clone()
        for token in SPECIAL_TOKEN_NAMES:
            token_id = self.word2idx.get(token)
            if token_id is not None:
                masked[token_id] = float("-inf")
        if allow_eop and self.eop_id is not None:
            masked[self.eop_id] = logits[self.eop_id]
        return masked

    def _valid_chinese_mask(self, logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        for token_id, token in self.idx2word.items():
            if not _is_chinese_char(token):
                masked[int(token_id)] = float("-inf")
        return masked

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: Sequence[int],
        no_repeat_ngram_size: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty and repetition_penalty != 1.0:
            for token_id in set(generated):
                logits[token_id] = logits[token_id] / repetition_penalty

        if no_repeat_ngram_size <= 1 or len(generated) + 1 < no_repeat_ngram_size:
            return logits

        prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
        banned = set()
        for i in range(len(generated) - no_repeat_ngram_size + 1):
            ngram = tuple(generated[i:i + no_repeat_ngram_size])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for token_id in banned:
            logits[token_id] = float("-inf")
        return logits

    def _sample(
        self,
        logits: torch.Tensor,
        strategy: str,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        temperature = max(float(temperature), 1e-6)
        logits = logits / temperature

        finite = torch.isfinite(logits)
        if not finite.any():
            return int(torch.argmax(torch.nan_to_num(logits, nan=-1e9)).item())

        strategy = strategy.lower()
        if strategy == "greedy":
            return int(torch.argmax(logits).item())

        filtered = logits.clone()
        if strategy == "topk":
            k = max(1, min(int(top_k), filtered.numel()))
            threshold = torch.topk(filtered, k).values[-1]
            filtered[filtered < threshold] = float("-inf")
        elif strategy == "topp":
            top_p = min(max(float(top_p), 1e-6), 1.0)
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = float("-inf")

        probs = torch.softmax(filtered, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
            return int(torch.argmax(logits).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        form_id: Optional[int] = None,
        form_token: Optional[str] = None,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.90,
        max_gen_len: Optional[int] = None,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate one poem.

        ``strategy="constrained"`` delegates to the fixed-format decoder.
        Other strategies stop when ``<EOP>`` is produced or ``max_gen_len`` is
        reached.
        """
        resolved_form = self._form_token(form_id=form_id, form_token=form_token)
        if strategy.lower() == "constrained":
            return self.generate_constrained(
                prompt=prompt,
                form_id=form_id,
                form_token=resolved_form,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=max(no_repeat_ngram_size, 3),
            )

        max_len = max_gen_len or self.max_gen_len
        self.model.eval()

        prefix = self._prefix_ids(prompt, resolved_form)
        sequence = list(prefix)
        generated: List[int] = []

        for _ in range(max_len):
            input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
            logits, _hidden = self.model(input_ids)
            next_logits = logits[0, -1, :]
            next_logits = self._valid_char_mask(next_logits, allow_eop=True)
            next_logits = self._apply_repetition_penalty(
                next_logits,
                generated,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
            next_id = self._sample(next_logits, strategy, temperature, top_k, top_p)
            if next_id == self.eop_id:
                break
            generated.append(next_id)
            sequence.append(next_id)

        return self._ids_to_text(prefix + generated)

    @torch.no_grad()
    def generate_constrained(
        self,
        prompt: str = "",
        form_id: Optional[int] = None,
        form_token: Optional[str] = None,
        temperature: float = 0.9,
        top_k: int = 20,
        top_p: float = 0.90,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Generate a poem with fixed line count, line length and punctuation."""
        resolved_form = self._form_token(form_id=form_id, form_token=form_token) or "<5JUE>"
        line_count, line_len = FORM_SPECS.get(resolved_form, FORM_SPECS["<5JUE>"])
        target_chars = line_count * line_len

        self.model.eval()
        sequence = self._prefix_ids("", resolved_form)
        chars = [ch for ch in prompt if _is_chinese_char(ch)]
        char_ids = [self.word2idx.get(ch, self.unk_id) for ch in chars[:target_chars]]
        sequence.extend(char_ids)
        generated = list(char_ids)

        while len(generated) < target_chars:
            input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
            logits, _hidden = self.model(input_ids)
            next_logits = self._valid_chinese_mask(logits[0, -1, :])
            next_logits = self._apply_repetition_penalty(
                next_logits,
                generated,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=1.1,
            )
            next_id = self._sample(next_logits, "topk", temperature, top_k, top_p)
            generated.append(next_id)
            sequence.append(next_id)

        poem_chars = self._ids_to_text(generated)[:target_chars]
        return self._format_fixed_lines(poem_chars, line_count, line_len)

    @torch.no_grad()
    def generate_acrostic(
        self,
        heads: str,
        form_id: Optional[int] = None,
        form_token: Optional[str] = None,
        temperature: float = 0.9,
        top_k: int = 20,
    ) -> str:
        """Generate an acrostic poem with each line starting from ``heads``."""
        resolved_form = self._form_token(form_id=form_id, form_token=form_token)
        if resolved_form is None:
            resolved_form = "<5JUE>" if len(heads) <= 4 else "<5LV>"
        line_count, line_len = FORM_SPECS.get(resolved_form, (len(heads), 5))
        line_count = max(line_count, len(heads))

        self.model.eval()
        sequence = self._prefix_ids("", resolved_form)
        generated: List[int] = []

        for line_idx in range(line_count):
            if line_idx < len(heads) and heads[line_idx] in self.word2idx:
                forced = self.word2idx[heads[line_idx]]
            else:
                forced = self.word2idx.get("春", self.unk_id)
            generated.append(forced)
            sequence.append(forced)

            for _ in range(line_len - 1):
                input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
                logits, _hidden = self.model(input_ids)
                next_logits = self._valid_chinese_mask(logits[0, -1, :])
                next_id = self._sample(next_logits, "topk", temperature, top_k, 0.9)
                generated.append(next_id)
                sequence.append(next_id)

            punct = "，" if line_idx % 2 == 0 else "。"
            punct_id = self.word2idx.get(punct)
            if punct_id is not None:
                sequence.append(punct_id)

        poem_chars = self._ids_to_text(generated)
        return self._format_fixed_lines(poem_chars, line_count, line_len)

    def generate_many(
        self,
        prompts: Sequence[str],
        strategies: Sequence[str],
        form_ids: Sequence[int] = (0, 1, 2, 3),
        temperature: float = 0.9,
        top_k: int = 20,
        top_p: float = 0.90,
    ) -> List[GenerationRecord]:
        records: List[GenerationRecord] = []
        for strategy in strategies:
            for idx, prompt in enumerate(prompts):
                form_id = form_ids[idx % len(form_ids)]
                form_token = FORM_ID_TO_TOKEN.get(form_id)
                poem = self.generate(
                    prompt=prompt,
                    form_id=form_id,
                    strategy=strategy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    no_repeat_ngram_size=3 if strategy != "greedy" else 0,
                    repetition_penalty=1.05 if strategy != "greedy" else 1.0,
                )
                records.append(GenerationRecord(strategy, form_token, prompt, poem))
        return records

    @staticmethod
    def _format_fixed_lines(text: str, line_count: int, line_len: int) -> str:
        lines = []
        for i in range(line_count):
            chunk = text[i * line_len:(i + 1) * line_len]
            punct = "，" if i % 2 == 0 else "。"
            lines.append(chunk + punct)
        return "".join(lines)

