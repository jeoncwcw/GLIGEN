"""Utilities for sharing self-attention key/value pairs between UNet passes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import torch.nn as nn

from .attention import BasicTransformerBlock, SelfAttention

KVCache = Optional[Tuple[object, object]]


@dataclass
class _AttnRecord:
    attn: SelfAttention
    cache: KVCache = None


class SelfAttentionSharingController:
    """Coordinates self-attention key/value recording and reuse."""

    def __init__(self, module: nn.Module, detach_shared: bool = True):
        self.module = module
        self.detach_shared = detach_shared
        self._records: List[_AttnRecord] = self._collect_attn_layers()

    def _collect_attn_layers(self) -> List[_AttnRecord]:
        records: List[_AttnRecord] = []
        for block in self.module.modules():
            if isinstance(block, BasicTransformerBlock):
                records.append(_AttnRecord(attn=block.attn1))
        return records

    def _iter_attn(self) -> Iterable[SelfAttention]:
        return (record.attn for record in self._records)

    def reset(self) -> None:
        for record in self._records:
            record.cache = None
            record.attn.clear_kv_override()
            record.attn.clear_kv_cache()
            record.attn.disable_kv_recording()

    def enable_recording(self, detach: Optional[bool] = None) -> None:
        detach_flag = self.detach_shared if detach is None else detach
        for record in self._records:
            record.cache = None
            record.attn.enable_kv_recording(detach=detach_flag)

    def collect_caches(self) -> List[KVCache]:
        caches: List[KVCache] = []
        for record in self._records:
            caches.append(record.attn.get_kv_cache())
            record.cache = caches[-1]
            record.attn.disable_kv_recording()
        return caches

    def apply_caches(self, caches: Sequence[KVCache], detach: Optional[bool] = None) -> None:
        detach_flag = self.detach_shared if detach is None else detach
        for record, cache in zip(self._records, caches):
            record.cache = cache
            record.attn.set_kv_override(cache, detach=detach_flag)

    def clear_overrides(self) -> None:
        for record in self._records:
            record.attn.clear_kv_override()
            record.cache = None

    @contextmanager
    def recording_scope(self, detach: Optional[bool] = None) -> Iterator[None]:
        self.enable_recording(detach)
        try:
            yield
        finally:
            self.collect_caches()

    def cached_caches(self) -> List[KVCache]:
        return [record.cache for record in self._records]

    def num_layers(self) -> int:
        return len(self._records)
