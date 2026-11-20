# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM request and transfer bookkeeping for native BTC cache plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _TransferMetricName,
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    SchedulerOffloadConfig,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec, ReqContext
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.metrics.cache_hit_source import CacheHitSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import OffloadKey
    from vllm.v1.kv_offload.btc_joint import (
        BtcTreeOffloadingManager,
        BtcTreeOffloadingSpec,
        DeviceProjection,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(kw_only=True)
class BtcRequestContext(ReqContext):
    # Lower-tier adapters use the same context while a finished request's
    # writeback is still awaiting receipts.
    _request: Request

    @property
    def num_computed_tokens(self) -> int:
        return self._request.num_computed_tokens

    @property
    def num_in_flight_tokens(self) -> int:
        return self._request.num_in_flight_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return self._request.num_prompt_tokens

    @property
    def num_tokens(self) -> int:
        return self._request.num_tokens


@dataclass
class BtcRequestState:
    req: Request
    req_context: BtcRequestContext
    max_load_tokens: int | None
    max_offload_tokens: int | None
    num_locally_computed_tokens: int = 0
    load_source_ranges: list[tuple[int, int, CacheHitSource]] = field(default_factory=list)
    load_failed: bool = False
    transfer_jobs: set[int] = field(default_factory=set)
    writebacks: dict[OffloadKey, tuple[DeviceProjection, int]] = field(default_factory=dict)
    observed_writebacks: set[OffloadKey] = field(default_factory=set)
    finished: bool = False


@dataclass(frozen=True)
class _StorePins:
    ticket: int
    blocks: tuple[KVCacheBlock, ...]


class BtcConnectorScheduler:
    """Deliver native cache operations and retire them after all rank receipts."""

    def __init__(
        self,
        spec: BtcTreeOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        manager: BtcTreeOffloadingManager,
    ) -> None:
        self.config = SchedulerOffloadConfig.from_spec(spec, vllm_config, kv_cache_config)
        self.manager = manager
        self._host_capacity = spec.host_capacity
        self._offloading_enabled = not spec.async_init
        self._ready_ranks: set[int] = set()
        self._ids = count()
        self._req_status: dict[str, BtcRequestState] = {}
        self._job_statuses: dict[int, TransferJobStatus] = {}
        self._load_jobs: dict[int, TransferJob] = {}
        self._store_pins: dict[int, _StorePins] = {}
        self._stats = OffloadingConnectorStats()
        self._info_sent = False
        self._events = OffloadingEventsTracker(
            spec.kv_events_config,
            {
                group.group_idx: group.kv_event_group_spec
                for group in self.config.kv_group_configs
            },
            num_hash_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
        )

    def _generate_job_id(self) -> int:
        return next(self._ids)

    def on_new_request(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        limits = {}
        for name in ("max_load_tokens", "max_offload_tokens"):
            value = params.get(name)
            if value is not None and (type(value) is not int or value < 0):
                logger.warning("%s must be a non-negative int, got %r", name, value)
                value = None
            limits[name] = value
        context = BtcRequestContext(
            req_id=request.request_id,
            kv_transfer_params=request.kv_transfer_params,
            kv_hints=request.kv_hints,
            _request=request,
        )
        self._events.on_new_request(context, request, self.config.kv_group_configs)
        self._req_status[request.request_id] = BtcRequestState(
            request, context, **limits
        )

    def request_finished(self, request: Request) -> None:
        state = self._req_status.get(request.request_id)
        if state is not None:
            state.finished = True
            self._finish_request(state)

    def _finish_request(self, state: BtcRequestState) -> None:
        if state.finished and not state.transfer_jobs and not state.writebacks:
            del self._req_status[state.req.request_id]

    def offer_writeback(
        self, request: Request, projection: DeviceProjection, end: int
    ) -> None:
        state = self._req_status[request.request_id]
        if self.config.offload_prompt_only and end > request.num_prompt_tokens:
            return
        if state.max_offload_tokens is not None and end > state.max_offload_tokens:
            return
        # A later request can offer a resident page again after a failed
        # write or Host eviction. Avoid reconsidering it on every decode step.
        key = projection.key
        if key not in state.observed_writebacks:
            state.observed_writebacks.add(key)
            state.writebacks[key] = (projection, end)

    def enqueue_load(
        self, job_id: int, job: TransferJob, keys: tuple[OffloadKey, ...]
    ) -> None:
        self._track_job(job_id, job.req_id, keys, is_store=False)
        self._load_jobs[job_id] = job

    def _track_job(
        self,
        job_id: int,
        req_id: str,
        keys: tuple[OffloadKey, ...],
        *,
        is_store: bool,
    ) -> None:
        self._req_status[req_id].transfer_jobs.add(job_id)
        self._job_statuses[job_id] = TransferJobStatus(
            req_id=req_id,
            pending_count=self.config.num_workers,
            keys=keys,
            is_store=is_store,
        )

    def _take_store_jobs(self, pool: BlockPool) -> dict[int, TransferJob]:
        stores = {}
        for state in tuple(self._req_status.values()):
            if not state.writebacks:
                continue
            offered = {}
            for projection, end in state.writebacks.values():
                block = projection.block
                slot = block.block_id
                identity = (projection.group, slot, projection.generation or 0)
                if (
                    self.manager._device_slots.get(slot) is projection
                    and pool.blocks[slot] is block
                    and block.block_hash == projection.key
                    and block.block_hash_num_tokens == end
                ):
                    offered[identity] = (projection, end)
            selected = (
                self.manager.tree.select_device_writebacks(offered, len(offered))
                if offered
                else ()
            )
            ready = {
                offered[identity][0].key: offered[identity]
                for identity in map(tuple, selected)
            }
            state.writebacks = ready
            candidates = sorted(
                tuple(ready)[: self._host_capacity],
                key=lambda key: (ready[key][0].group, ready[key][1]),
            )
            if not candidates:
                self._finish_request(state)
                continue
            for key in candidates:
                state.req_context.set_offload_key_position(key, ready[key][1])
            reserve = self.manager.prepare_store(
                state.req, [ready[key] for key in candidates]
            )
            if reserve is None:
                continue
            ticket, slots = reserve
            projections = {key: ready.pop(key) for key in candidates}
            keys = tuple(key for key in candidates if key in slots)
            if not keys:
                assert not ticket
                self._finish_request(state)
                continue

            # Keep the GPU pointer order aligned with the native Host slots by
            # group, and pin each physical source through all rank receipts.
            groups = self.config.kv_group_configs
            blocks = tuple(projections[key][0].block for key in keys)
            group_sizes = [0] * len(groups)
            block_indices = [0] * len(groups)
            for key in keys:
                projection, end = projections[key]
                group = groups[projection.group]
                position = end // group.tokens_per_block - 1
                if group_sizes[projection.group] == 0:
                    block_indices[projection.group] = position
                group_sizes[projection.group] += 1
                self._events.record_store(state.req, group, position, key)
            source = GPULoadStoreSpec(
                [block.block_id for block in blocks], group_sizes, block_indices
            )
            destination = CPULoadStoreSpec([slots[key] for key in keys])
            job_id = self._generate_job_id()
            pool.touch(list(blocks))
            self._store_pins[job_id] = _StorePins(ticket, blocks)
            self._track_job(job_id, state.req.request_id, keys, is_store=True)
            stores[job_id] = TransferJob(state.req.request_id, source, destination)
        return stores

    def build_connector_meta(
        self, output: SchedulerOutput, pool: BlockPool
    ) -> OffloadingConnectorMetadata:
        # Pins prevent reuse. A recurrent CoW can still move away a cached
        # hash and keep the request writing the original block this step.
        # Drain older D2H readers before that step's model writes.
        overwritten = {
            copy.src_block_id for copy in (output.kv_cache_block_copies or ())
        }
        flushes = {
            job_id
            for job_id, pins in self._store_pins.items()
            if any(
                block.block_id in overwritten or block.block_hash != key
                for block, key in zip(
                    pins.blocks, self._job_statuses[job_id].keys, strict=True
                )
            )
        }
        loads, self._load_jobs = self._load_jobs, {}
        stores = self._take_store_jobs(pool) if self._offloading_enabled else {}
        return OffloadingConnectorMetadata(
            load_jobs=loads,
            store_jobs=stores,
            jobs_to_flush=flushes or None,
            offloading_enabled=self._offloading_enabled,
        )

    def update_connector_output(
        self, metadata: OffloadingWorkerMetadata, pool: BlockPool
    ) -> None:
        if not self._offloading_enabled:
            self._ready_ranks.update(metadata.ready_ranks)
            if len(self._ready_ranks) >= self.config.num_workers:
                self._offloading_enabled = True
        for stats, byte_metric, time_metric, size_metric in (
            (
                metadata.transfer_stats.load,
                _TransferMetricName.LOAD_BYTES,
                _TransferMetricName.LOAD_TIME,
                _TransferMetricName.LOAD_SIZE,
            ),
            (
                metadata.transfer_stats.store,
                _TransferMetricName.STORE_BYTES,
                _TransferMetricName.STORE_TIME,
                _TransferMetricName.STORE_SIZE,
            ),
        ):
            if not stats.is_empty():
                self._stats.increase_counter(byte_metric, stats.bytes)
                self._stats.increase_counter(time_metric, stats.time)
                for chunk_bytes in stats.sizes:
                    self._stats.observe_histogram(size_metric, chunk_bytes)
        for job_id, acknowledgements in metadata.completed_jobs.items():
            job = self._job_statuses[job_id]
            assert 0 < acknowledgements <= job.pending_count
            job.pending_count -= acknowledgements
            job.failed |= job_id in metadata.failed_jobs
            if job.pending_count:
                continue
            state = self._req_status[job.req_id]
            if job.is_store:
                pins = self._store_pins.pop(job_id)
                try:
                    self.manager.complete_store(
                        pins.ticket,
                        job.keys,
                        state.req_context,
                        success=not job.failed,
                    )
                finally:
                    pool.free_blocks(pins.blocks)
            else:
                state.load_failed |= job.failed
                self.manager.complete_load(job.req_id, success=not job.failed)
            state.transfer_jobs.remove(job_id)
            del self._job_statuses[job_id]
            self._finish_request(state)

    def get_external_cache_hit_sources(
        self, request: Request, external_tokens: int
    ) -> dict[CacheHitSource, int]:
        if not external_tokens:
            return {}
        state = self._req_status[request.request_id]
        start = state.num_locally_computed_tokens
        end = start + external_tokens
        events = [(end, 0, CacheHitSource.EXTERNAL_UNSPECIFIED)]
        for lo, hi, tier in state.load_source_ranges:
            lo, hi = max(start, lo), min(end, hi)
            if lo < hi:
                events.extend(((lo, 1, tier), (hi, -1, tier)))
        result: Counter[CacheHitSource] = Counter()
        active: Counter[CacheHitSource] = Counter()
        for boundary, delta, tier in sorted(events):
            if boundary > start:
                tiers = [kind for kind, n in active.items() if n > 0]
                source = (
                    CacheHitSource.outermost(tiers)
                    if tiers
                    else CacheHitSource.EXTERNAL_UNSPECIFIED
                )
                result[source] += boundary - start
            active[tier] += delta
            start = boundary
        return dict(result)

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self._events.take_events(list(self.manager.take_events()))

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._stats
        self._stats = OffloadingConnectorStats()
        if not self._info_sent:
            stats.set_info(self.manager.config_info())
            self._info_sent = True
        return None if stats.is_empty() else stats

    def has_pending_push_work(self) -> bool:
        return bool(
            self._job_statuses
            or any(state.writebacks for state in self._req_status.values())
            or self.manager.has_pending_work()
        )

    def pending_gpu_slots(self) -> set[int]:
        return {
            block.block_id
            for pins in self._store_pins.values()
            for block in pins.blocks
        }

    def reset_cache(self) -> bool:
        if self._job_statuses:
            return False
        for state in tuple(self._req_status.values()):
            state.writebacks.clear()
            state.observed_writebacks.clear()
            state.load_source_ranges.clear()
            self._finish_request(state)
        self.manager.reset_cache()
        self._events.reset()
        return True

    def shutdown(self) -> None:
        self.manager.shutdown()
