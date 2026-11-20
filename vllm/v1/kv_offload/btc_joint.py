# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BTC joint matching with vLLM-owned device pages and transfer workers."""

import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from itertools import count
from math import lcm
from typing import TYPE_CHECKING, Any

from vllm.distributed.device_communicators.shm_broadcast import check_shm_free_space
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    ConfigInfoMapping,
    GPULoadStoreSpec,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.btc import BtcOffloadingSpec
from vllm.v1.kv_offload.btc_connector_scheduler import BtcConnectorScheduler
from vllm.v1.kv_offload.btc_connector_worker import BtcConnectorWorker
from vllm.v1.kv_offload.btc_transfer_spec import (
    BtcConnectorMetadata,
    BtcDiskOperation,
    BtcLoadSpec,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import _all_workers_barrier
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo
    from vllm.v1.request import Request

logger = init_logger(__name__)


class Tier(IntEnum):
    DEVICE = 0
    HOST = 1
    DISK = 2


class PageKind(IntEnum):
    READY = 0
    JOIN = 1
    LOAD = 2


class BoundLoadReadError(RuntimeError):
    """Recoverable source I/O failure before a bound load is submitted."""


@dataclass(frozen=True)
class PageBinding:
    group: int
    position: int
    hash_row: int
    tier: Tier
    source_id: int
    destination_id: int | None = None
    kind: PageKind = PageKind.LOAD
    generation: int = 0
    operation_id: int = 0

    def trace(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier.name
        data["kind"] = self.kind.name
        return data


@dataclass(frozen=True)
class AdmissionPlan:
    device_tokens: int
    matched_tokens: int
    pages: tuple[PageBinding, ...]
    plan_id: int | None = None

    @property
    def external_tokens(self) -> int:
        return self.matched_tokens - self.device_tokens


def _json(data) -> str:
    return json.dumps(data, separators=(",", ":"))


DeviceIdentity = tuple[int, int, int]


@dataclass(frozen=True)
class DeviceProjection:
    group: int
    key: OffloadKey
    block: "KVCacheBlock"
    generation: int | None

    def identity(self) -> DeviceIdentity:
        assert self.generation is not None
        return self.group, self.block.block_id, self.generation


@dataclass
class SharedLoad:
    operation_id: int
    pages: tuple[PageBinding, ...]
    targets: tuple[DeviceProjection, ...]
    waiters: set[str] = field(default_factory=set)


@dataclass
class SharedLoadWaiter:
    operations: tuple[int, ...]
    pending: set[int]
    failed: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class BoundTierLoad:
    pages: tuple[PageBinding, ...]
    host_ticket: int
    disk_ticket: int


def _trace_device_identity(identity: DeviceIdentity) -> dict[str, int]:
    group, slot, generation = identity
    return {"group": group, "slot": slot, "generation": generation}


class BtcTreeOffloadingManager:
    """Bridge native BTC operations to vLLM transport jobs and GPU references."""

    def __init__(
        self,
        num_chunks: int,
        info: "CPUOffloadingInfo",
        config: "OffloadingConfig",
        trace_transfers: bool,
        disk_slots: int = 0,
    ) -> None:
        if socket_path := os.environ.get("RTP_BTC_TREE_SOCKET"):
            from btc_tree_rpc import TreeCacheClient

            self.tree = TreeCacheClient(len(config.groups), socket_path)
        else:
            from btc_tree import TreeCache

            self.tree = TreeCache(
                len(config.groups), config.extra_config.get("btc_tree_library")
            )
        self.tree.configure_host(num_chunks)
        if disk_slots:
            self.tree.configure_disk(
                disk_slots, config.parallel.world_size, trace_transfers
            )
        self._has_disk = bool(disk_slots)
        self._trace_transfers = trace_transfers
        self._config_info = info.as_config_info()
        self._events: list[OffloadingEvent] | None = (
            [] if config.enable_kv_cache_events else None
        )
        self._hash_block_size = config.cache.tokens_per_hash
        self._group_block_sizes = {
            group.group_id: group.tokens_per_block for group in config.groups
        }
        self.share_device_loads = bool(config.extra_config.get("btc_join", False))
        self.protect_device_pages = self.share_device_loads or bool(
            config.extra_config.get("btc_gpu_reclaim", False)
        )
        self._gpu_pool: BlockPool | None = None
        self._device_slots: dict[int, DeviceProjection] = {}
        self._device_key_to_slot: dict[OffloadKey, int] = {}
        self._generations = count(1)
        self._reclaiming: set[DeviceIdentity] = set()
        self._pending_device_pages: dict[OffloadKey, PageBinding] = {}
        self._load_leases: dict[str, BoundTierLoad] = {}
        self._disk_transport_jobs: dict[int, int] = {}

    def bind_device_pool(self, pool: "BlockPool") -> None:
        self._gpu_pool = pool

    def prepare_store(
        self, request: "Request", candidates: Sequence[tuple[DeviceProjection, int]]
    ) -> tuple[int, dict[OffloadKey, int]] | None:
        rows = [
            (projection.group, end // self._hash_block_size - 1)
            for projection, end in candidates
        ]
        result = self.tree.prepare_host_store(
            request.block_hashes, rows, request.request_id
        )
        if result is None:
            return None
        ticket, bindings, evicted = result
        slots = {candidates[index][0].key: slot for index, slot in bindings}
        if self._events is not None and slots:
            # A new reservation supersedes queued metadata-only failure events
            # for the same keys, even before the replacement is readable.
            events = []
            for event in self._events:
                if event.metadata_only:
                    keys = [key for key in event.keys if key not in slots]
                    if not keys:
                        continue
                    event = replace(event, keys=keys)
                events.append(event)
            self._events = events
        if self._events is not None and evicted:
            self._events.append(
                OffloadingEvent(
                    keys=[make_offload_key(key, group) for group, key, _ in evicted],
                    medium=Medium.CPU,
                    removed=True,
                )
            )
        return ticket, slots

    def complete_store(
        self,
        ticket: int,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        success: bool,
    ) -> None:
        if not self.tree.settle_host_store(ticket, success):
            raise RuntimeError(f"BTC refused to settle Host store ticket {ticket}")
        if self._events is not None:
            self._events.append(
                OffloadingEvent(
                    keys=list(keys),
                    medium=Medium.CPU,
                    removed=not success,
                    metadata_only=not success,
                    removal_expected=success,
                    req_context=req_context if success else None,
                )
            )
        if success:
            self._log_transfer(
                "store", map(get_offload_group_idx, keys), req_context.req_id
            )

    def publish_device(
        self, hashes: Sequence[bytes], group: int, row: int, block: "KVCacheBlock"
    ) -> DeviceProjection | None:
        key = make_offload_key(hashes[row], group)
        slot = block.block_id
        previous = self._device_slots.get(slot)
        if previous is not None and previous.key == key:
            return previous
        # Native BTC exposes one device copy per logical key. Never retain a
        # second physical reference for a duplicate vLLM hash entry.
        if self.protect_device_pages and (
            key in self._device_key_to_slot or key in self._pending_device_pages
        ):
            return None
        self.invalidate_device(slot)
        generation = None
        if self.protect_device_pages:
            pool = self._gpu_pool
            assert pool is not None and pool.blocks[slot] is block
            generation = next(self._generations)
            pool.touch([block])
            try:
                self.tree.publish_device(hashes, [(group, row, slot, generation)])
            except Exception:
                pool.free_blocks([block])
                raise
        else:
            self.tree.publish(hashes, [(group, row, Tier.DEVICE, slot)])
        entry = DeviceProjection(group, key, block, generation)
        self._device_slots[slot] = entry
        if generation is not None:
            self._device_key_to_slot[key] = slot
        return entry

    def invalidate_device(self, slot: int) -> None:
        previous = self._device_slots.get(slot)
        if previous is None:
            return
        if previous.generation is None:
            self.tree.unpublish(previous.group, Tier.DEVICE, slot)
        else:
            identity = previous.identity()
            if identity not in self._reclaiming:
                self.tree.unpublish_device(*identity)
            pool = self._gpu_pool
            assert pool is not None and previous.block.ref_cnt > 0
            pool.free_blocks([previous.block], evict_first=True)
            del self._device_key_to_slot[previous.key]
        del self._device_slots[slot]

    def begin_shared_load(
        self,
        hashes: Sequence[bytes],
        pages: tuple[PageBinding, ...],
        operation_id: int,
    ) -> SharedLoad:
        pool = self._gpu_pool
        assert pool is not None
        targets = []
        owned_pages = []
        for page in pages:
            assert page.destination_id is not None and page.destination_id > 0
            block = pool.blocks[page.destination_id]
            key = make_offload_key(hashes[page.hash_row], page.group)
            assert key not in self._pending_device_pages
            assert key not in self._device_key_to_slot and block.block_hash is None
            generation = next(self._generations)
            targets.append(DeviceProjection(page.group, key, block, generation))
            owned_pages.append(
                replace(page, generation=generation, operation_id=operation_id)
            )
        blocks = [target.block for target in targets]
        pool.touch(blocks)
        try:
            self.tree.begin_device_load(
                hashes,
                [
                    (
                        page.group,
                        page.hash_row,
                        page.tier,
                        page.source_id,
                        page.destination_id,
                        page.generation,
                    )
                    for page in owned_pages
                ],
                operation_id,
            )
        except Exception:
            pool.free_blocks(blocks)
            raise
        self._pending_device_pages.update(
            (target.key, page) for target, page in zip(targets, owned_pages)
        )
        return SharedLoad(operation_id, tuple(owned_pages), tuple(targets))

    def settle_shared_load(self, transfer: SharedLoad, publish: bool) -> None:
        pool = self._gpu_pool
        assert pool is not None
        for record in transfer.targets:
            assert record.block.ref_cnt > 0
            assert pool.blocks[record.block.block_id] is record.block
            assert record.block.block_hash is None
        if not self.tree.settle_device_load(transfer.operation_id, publish):
            raise RuntimeError("BTC refused to settle the active shared load")
        for record, page in zip(transfer.targets, transfer.pages):
            del self._pending_device_pages[record.key]
            if publish:
                # All workers are done. Make the exact GPU hash visible before
                # any waiter resumes. The operation pin becomes the BTC pin.
                pool._insert_block_hash(
                    BlockHashWithGroupId(record.key),
                    record.block,
                    (page.hash_row + 1) * self._hash_block_size,
                )
                slot = record.block.block_id
                self._device_slots[slot] = record
                self._device_key_to_slot[record.key] = slot
        if not publish:
            pool.free_blocks([record.block for record in transfer.targets])

    def _is_idle(self, projection: DeviceProjection) -> bool:
        pool = self._gpu_pool
        assert pool is not None
        block = projection.block
        return (
            projection.generation is not None
            and pool.blocks[block.block_id] is block
            and block.block_hash == projection.key
            and block.ref_cnt == 1
        )

    def reclaim_device_slots(
        self, shortage: int, protected: set[int]
    ) -> tuple[int, list[DeviceIdentity], int]:
        pool = self._gpu_pool
        assert pool is not None
        eligible = [
            entry.identity()
            for slot, entry in self._device_slots.items()
            if slot not in protected
            and slot not in pool._reuse_watchers
            and self._is_idle(entry)
        ]
        if not eligible:
            return 0, [], 0
        transaction, selected = self.tree.plan_device_reclaim(eligible, shortage)
        released: list[DeviceIdentity] = []
        if transaction:
            try:
                eligible_set = set(eligible)
                for identity in map(tuple, selected):
                    if identity not in eligible_set:
                        continue
                    _, slot, _ = identity
                    entry = self._device_slots.get(slot)
                    if (
                        entry is None
                        or not self._is_idle(entry)
                        or entry.identity() != identity
                        or slot in pool._reuse_watchers
                    ):
                        continue
                    before = pool.get_num_free_blocks()
                    self._reclaiming.add(identity)
                    try:
                        pool.evict_blocks({slot})
                    finally:
                        self._reclaiming.remove(identity)
                    if pool.get_num_free_blocks() > before:
                        released.append(identity)
            finally:
                self.tree.settle_device_reclaim(transaction, released)
        return transaction, released, len(eligible)

    def release_device_refs_for_reset(self, pending_transfers: set[int]) -> None:
        pool = self._gpu_pool
        assert pool is not None
        before = pool.get_num_free_blocks()
        pins_before = len(self._device_slots)
        # Determine whether the normal reset can succeed *before* changing
        # anything. Apart from its null block, only BTC-only guards may be used.
        can_reset = (
            pool.num_gpu_blocks - before - 1 == pins_before
            and not pending_transfers.intersection(self._device_slots)
            and all(self._is_idle(entry) for entry in self._device_slots.values())
        )
        if can_reset:
            for slot in tuple(self._device_slots):
                self.invalidate_device(slot)
        if self._trace_transfers:
            logger.info(
                "BTC_GPU_RESET info=%s",
                _json(
                    {
                        "allowed": can_reset,
                        "free_before": before,
                        "free_after": pool.get_num_free_blocks(),
                        "pins_before": pins_before,
                        "pins_after": len(self._device_slots),
                    }
                ),
            )

    def prepare_bound_load(
        self, pages: tuple[PageBinding, ...], request_id: str
    ) -> CPULoadStoreSpec:
        assert request_id not in self._load_leases
        host = [(page.group, page.source_id) for page in pages if page.tier is Tier.HOST]
        disk = [(page.group, page.source_id) for page in pages if page.tier is Tier.DISK]
        assert len(host) + len(disk) == len(pages)
        host_ticket = 0
        try:
            if host:
                host_ticket = self.tree.pin_host_load(host)
            if disk:
                spec = BtcLoadSpec(
                    [page.source_id if page.tier is Tier.HOST else -1 for page in pages],
                    [page.source_id if page.tier is Tier.DISK else -1 for page in pages],
                )
                disk_ticket = self.tree.pin_disk_load(disk)
            else:
                spec = CPULoadStoreSpec([slot for _, slot in host])
                disk_ticket = 0
        except Exception:
            if host_ticket:
                self._release_host_load(host_ticket)
            raise
        self._load_leases[request_id] = BoundTierLoad(pages, host_ticket, disk_ticket)
        return spec

    def _release_host_load(self, ticket: int) -> None:
        if not self.tree.release_host_load(ticket):
            raise RuntimeError(f"BTC refused to release Host load ticket {ticket}")

    def complete_load(self, request_id: str, success: bool) -> None:
        load = self._load_leases[request_id]
        host = [
            (page.group, page.source_id) for page in load.pages if page.tier is Tier.HOST
        ]
        try:
            if load.disk_ticket:
                self.tree.complete_disk_load(load.disk_ticket, success)
            if self._has_disk and host and success:
                self.tree.retry_disk_writes(host, request_id)
        finally:
            if load.host_ticket:
                self._release_host_load(load.host_ticket)
            del self._load_leases[request_id]
        if success:
            self._log_transfer("load", (group for group, _ in host), request_id)
        if self._trace_transfers and success:
            logger.info(
                "BTC_JOINT_LOAD request=%s pages=%s",
                request_id,
                _json([page.trace() for page in load.pages]),
            )

    def pending_load_gpu_slots(self) -> set[int]:
        return {
            page.destination_id
            for load in self._load_leases.values()
            for page in load.pages
            if page.destination_id is not None
        }

    def _disk_page_trace(self, row: dict) -> dict:
        group = row["group"]
        hash_row = row["hash_row"]
        tier = row["tier"]
        end = (hash_row + 1) * self._hash_block_size
        binding = PageBinding(
            group,
            end // self._group_block_sizes[group] - 1,
            hash_row,
            Tier[tier] if isinstance(tier, str) else Tier(tier),
            row["source_id"],
        )
        return row | binding.trace()

    def take_disk_jobs(
        self, next_job_id: Callable[[], int]
    ) -> dict[int, BtcDiskOperation]:
        if not self._has_disk:
            return {}
        jobs, reclaims = self.tree.take_disk_jobs()
        if self._trace_transfers:
            for notice in reclaims:
                reclaim = {key: value for key, value in notice.items() if key != "request"}
                reclaim["incoming"] = self._disk_page_trace(notice["incoming"])
                reclaim["lru"] = [self._disk_page_trace(row) for row in notice["lru"]]
                if notice["victim"] is not None:
                    reclaim["victim"] = self._disk_page_trace(notice["victim"])
                logger.info(
                    "BTC_DISK_RECLAIM request=%s info=%s",
                    notice["request"],
                    _json(reclaim),
                )
        result = {}
        for job in jobs:
            job_id = next_job_id()
            self._disk_transport_jobs[job_id] = job["id"]
            result[job_id] = BtcDiskOperation(
                tuple(job["disk_ids"]), tuple(job["host_ids"])
            )
        return result

    def collect_disk_receipts(
        self, metadata: OffloadingWorkerMetadata
    ) -> OffloadingWorkerMetadata:
        disk_jobs = self._disk_transport_jobs.keys() & metadata.completed_jobs.keys()
        if not disk_jobs:
            return metadata
        for job_id in sorted(disk_jobs):
            done, events = self.tree.settle_disk_job(
                self._disk_transport_jobs[job_id],
                metadata.completed_jobs[job_id],
                job_id in metadata.failed_jobs,
            )
            if done:
                del self._disk_transport_jobs[job_id]
            for event in events:
                self._log_disk_result(job_id, event)
        return replace(
            metadata,
            completed_jobs={
                job_id: count
                for job_id, count in metadata.completed_jobs.items()
                if job_id not in disk_jobs
            },
            failed_jobs=metadata.failed_jobs - disk_jobs,
        )

    def _log_disk_result(self, job_id: int, event: dict) -> None:
        kind = event["kind"]
        if kind == "failed":
            logger.warning(
                "BTC_DISK_WRITE_FAILED job=%d logical_slots=%s",
                job_id,
                tuple(event["disk_ids"]),
            )
        elif self._trace_transfers:
            request_id = event["request"]
            pages = _json([self._disk_page_trace(page) for page in event["pages"]])
            if kind == "commit":
                logger.info(
                    "BTC_DISK_WRITE_COMMIT request=%s job=%d pages=%s",
                    request_id,
                    job_id,
                    pages,
                )
            elif kind == "retry":
                logger.info(
                    "BTC_DISK_WRITE_RETRY request=%s pages=%s", request_id, pages
                )

    def _log_transfer(
        self, operation: str, groups: Iterable[int], request_id: str
    ) -> None:
        if self._trace_transfers:
            for group, count in sorted(Counter(groups).items()):
                logger.info(
                    "BTC_TRANSFER operation=%s group=%d blocks=%d request=%s",
                    operation,
                    group,
                    count,
                    request_id,
                )

    def take_events(self) -> list[OffloadingEvent]:
        if self._events is None:
            return []
        events, self._events = self._events, []
        return events

    def config_info(self) -> Sequence[ConfigInfoMapping]:
        return [self._config_info]

    def has_pending_work(self) -> bool:
        return self._has_disk and self.tree.disk_has_pending_work()

    def reset_blocked(self) -> bool:
        return bool(self._load_leases) or (
            self._has_disk and self.tree.disk_reset_blocked()
        )

    def reset_cache(self) -> None:
        assert not self._load_leases and not self._disk_transport_jobs
        if self._has_disk:
            self.tree.reset_disk()
        self.tree.reset_host()
        if self._events is not None:
            self._events.clear()

    def shutdown(self) -> None:
        for slot in tuple(self._device_slots):
            self.invalidate_device(slot)
        if close := getattr(self.tree, "close", None):
            close()


class BtcTreeOffloadingSpec(BtcOffloadingSpec):
    def __init__(self, config: "OffloadingConfig") -> None:
        super().__init__(config)
        self.host_capacity = self.num_chunks
        self.disk_slots = 0

    def create_tree_manager(self) -> BtcTreeOffloadingManager:
        if self.host_capacity <= 0:
            raise ValueError("BTC offload needs space for at least one Host row")
        return BtcTreeOffloadingManager(
            self.host_capacity,
            self.cpu_info,
            self.config,
            bool(self.extra_config.get("btc_trace_transfers", False)),
            self.disk_slots,
        )


class BtcDiskOffloadingSpec(BtcTreeOffloadingSpec):
    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)
        parallel = config.parallel
        if parallel.world_size > 1 and (
            parallel.world_size != parallel.local_world_size
            or parallel.world_size != parallel.tp_size
            or parallel.pp_size != 1
            or parallel.pcp_size != 1
            or parallel.dcp_size != 1
            or parallel.data_parallel_size != 1
            or self.replicated_layout
            or config.canonical_layout
        ):
            raise ValueError(
                "BTC Disk multi-worker requires nonreplicated, noncanonical "
                "single-node tensor parallelism"
            )
        self.disk_slots = int(self.extra_config["btc_disk_slots"])
        self.disk_staging_chunks = int(self.extra_config["btc_disk_staging_chunks"])
        self.host_capacity = self.num_chunks - self.disk_staging_chunks
        if (
            self.disk_slots <= 0
            or not 0 < self.disk_staging_chunks < self.num_chunks
            or not self._uses_shared_region()
        ):
            raise ValueError("BTC Disk requires space for Disk, staging and Host cache")
        self.rank_row_bytes = self.kv_bytes_per_chunk // parallel.world_size

    def _create_layout_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        parallel = self.config.parallel
        if parallel.world_size > 1:
            try:
                check_shm_free_space(
                    self.host_capacity * self.kv_bytes_per_chunk,
                    allocation_name="BTC rank-local Host cache regions in /dev/shm",
                )
            finally:
                _all_workers_barrier()
        region = SharedOffloadRegion(
            engine_id=f"{self.config.engine_id}_{parallel.rank}",
            num_chunks=self.host_capacity,
            rank=0,
            kv_bytes_per_chunk=self.rank_row_bytes,
            cpu_page_size=self.cpu_page_size_per_worker,
            private=True,
            populate=not self.async_init,
        )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.host_capacity,
                mmap_region=region,
                defer_host_setup=self.async_init,
                pin_pace=self.async_init_pin_pace,
            )
        except Exception:
            region.cleanup()
            raise


class BtcJointConnector(KVConnectorBase_V1, SupportsHMA):
    """Bind one BTC match to vLLM allocations and reuse its transfer transport."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        self._canonical_layout = offloading_config.canonical_layout
        spec = OffloadingSpecFactory.create_spec(offloading_config)
        if not isinstance(spec, BtcTreeOffloadingSpec):
            raise ValueError("BtcJointConnector requires BtcTreeOffloadingSpec")
        self.connector_scheduler: BtcConnectorScheduler | None = None
        self.connector_worker: BtcConnectorWorker | None = None
        self._plans: dict[str, AdmissionPlan] = {}
        self._plan_ids = count(1)
        self._shared_loads: dict[int, SharedLoad] = {}
        self._shared_waiters: dict[str, SharedLoadWaiter] = {}
        self._failed_bound_load_requests: dict[str, Request] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = BtcConnectorScheduler(
                spec, vllm_config, kv_cache_config, spec.create_tree_manager()
            )
            config = self.scheduler.config
            if (
                config.blocks_per_chunk != 1
                or len(config.kv_group_configs) != len(kv_cache_config.kv_cache_groups)
                or vllm_config.num_hash_lookahead_tokens
                or vllm_config.speculative_config is not None
                or self._kv_transfer_config.kv_load_failure_policy != "fail"
                or vllm_config.parallel_config.pipeline_parallel_size != 1
                or config.dcp_world_size != 1
                or kv_cache_config.hisparse_host_num_blocks is not None
            ):
                raise ValueError(
                    "BTC joint matching requires whole pages, prefix-cacheable "
                    "groups, no HiSparse or speculation, PP/DCP=1 and fail-on-load-error"
                )
            self._alignment = lcm(
                *(group.tokens_per_block for group in config.kv_group_configs)
            )
            for group in config.kv_group_configs:
                logger.info(
                    "BTC_HOST_GROUP group=%d tokens_per_block=%d layers=%s",
                    group.group_idx,
                    group.tokens_per_block,
                    ",".join(
                        kv_cache_config.kv_cache_groups[group.group_idx].layer_names
                    ),
                )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = BtcConnectorWorker(
                spec, vllm_config, kv_cache_config
            )

    @property
    def scheduler(self) -> BtcConnectorScheduler:
        assert self.connector_scheduler is not None
        return self.connector_scheduler

    @property
    def _worker_transport(self) -> BtcConnectorWorker:
        assert self.connector_worker is not None
        return self.connector_worker

    @property
    def requires_kv_delivery(self) -> bool:
        # BTC is a best-effort cache.
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        if vllm_config.attention_config.hisparse_config is not None:
            return "BLHNC"
        return "LBHNC"

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    @property
    def btc_manager(self) -> BtcTreeOffloadingManager:
        return self.scheduler.manager

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        self._worker_transport.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)
        self._worker_transport.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, OffloadingConnectorMetadata)
        self._worker_transport.start_kv_transfers(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, OffloadingConnectorMetadata)
        self._worker_transport.prepare_store_kv(metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert isinstance(self._get_connector_metadata(), OffloadingConnectorMetadata)
        return self._worker_transport.get_finished(finished_req_ids)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        results = super().get_transfer_results(finished_req_ids)
        results.failed_recving = self._worker_transport.take_failed_recving()
        return results

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            return
        meta = self.btc_manager.collect_disk_receipts(meta)
        kv = self._kv_cache_manager
        assert kv is not None
        self.scheduler.update_connector_output(meta, kv.block_pool)

    def on_new_request(self, request: "Request") -> None:
        self.scheduler.on_new_request(request)

    def get_external_cache_hit_sources(
        self,
        request: "Request",
        num_external_tokens: int,
    ) -> dict[CacheHitSource, int]:
        return self.scheduler.get_external_cache_hit_sources(
            request, num_external_tokens
        )

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        for manager in kv_cache_manager.coordinator.single_type_managers:
            if type(manager) not in (
                FullAttentionManager,
                MambaManager,
                SlidingWindowManager,
            ):
                raise ValueError("Unsupported cache group for BTC joint matching")
            if (
                isinstance(manager, MambaManager)
                and manager.mamba_cache_mode != "align"
            ):
                raise ValueError("BTC joint matching needs aligned Mamba checkpoints")
        kv_cache_manager.block_pool.on_cached_block_invalidated = (
            self.btc_manager.invalidate_device
        )
        self.btc_manager.bind_device_pool(kv_cache_manager.block_pool)
        if self.supports_device_reclaim:
            kv_cache_manager.block_pool.prepare_prefix_cache_reset = (
                self._release_device_refs_for_reset
            )

    @property
    def supports_device_reclaim(self) -> bool:
        return self.btc_manager.protect_device_pages

    def _pending_gpu_transfers(self) -> set[int]:
        return (
            self.scheduler.pending_gpu_slots()
            | self.btc_manager.pending_load_gpu_slots()
        )

    def _release_device_refs_for_reset(self) -> None:
        self.btc_manager.release_device_refs_for_reset(self._pending_gpu_transfers())

    def reclaim_gpu_cache(
        self,
        request: "Request",
        required_free: int,
        protected_block_ids: set[int],
        *,
        phase: str,
        attempt: int,
    ) -> Callable[[bool], None] | None:
        kv = self._kv_cache_manager
        assert kv is not None
        pool = kv.block_pool
        manager = self.btc_manager
        before = pool.get_num_free_blocks()
        pins_before = len(manager._device_slots)
        plan = self._plans.get(request.request_id)
        pending = self._pending_gpu_transfers()
        guarded = []
        if manager._trace_transfers:
            for slot, projection in sorted(manager._device_slots.items()):
                ref_cnt = projection.block.ref_cnt
                reasons = [
                    reason
                    for needed, reason in (
                        (slot in protected_block_ids, "plan_ready"),
                        (ref_cnt > 1, "live_ref"),
                        (slot in pending, "pending_transfer"),
                        (slot in pool._reuse_watchers, "reuse_watcher"),
                    )
                    if needed
                ]
                if reasons:
                    guarded.append(
                        dict(
                            _trace_device_identity(projection.identity()),
                            reason=reasons[0],
                            reasons=reasons,
                            ref_cnt=ref_cnt,
                        )
                    )
        transaction, victims, eligible = manager.reclaim_device_slots(
            required_free - before,
            protected_block_ids | pending,
        )
        if not manager._trace_transfers:
            return None
        after = pool.get_num_free_blocks()
        pins_after = len(manager._device_slots)

        def report(retry_succeeded: bool) -> None:
            logger.info(
                "BTC_RECLAIM request=%s info=%s",
                request.request_id,
                _json(
                    {
                        "phase": phase,
                        "attempt": attempt,
                        "plan_id": plan.plan_id if plan is not None else None,
                        "requested": required_free - before,
                        "required_free": required_free,
                        "free_before": before,
                        "free_after": after,
                        "native_txn_id": transaction,
                        "victims": [_trace_device_identity(victim) for victim in victims],
                        "protected": guarded,
                        "eligible": eligible,
                        "pins_before": pins_before,
                        "pins_after": pins_after,
                        "retried": after > before,
                        "retry_succeeded": retry_succeeded,
                    }
                ),
            )

        return report

    def publish_computed_gpu_blocks(
        self, request: "Request", num_committed_tokens: int
    ) -> None:
        kv = self._kv_cache_manager
        assert kv is not None
        if num_committed_tokens <= 0 or not kv.enable_caching:
            return
        hashes = request.block_hashes
        step = self.scheduler.config.tokens_per_hash
        groups = self.scheduler.config.kv_group_configs
        rows = kv.get_blocks(request.request_id).blocks
        for group, blocks in zip(groups, rows):
            num_blocks = num_committed_tokens // group.tokens_per_block
            for position, block in enumerate(blocks[:num_blocks]):
                end = (position + 1) * group.tokens_per_block
                row = end // step - 1
                if row >= len(hashes) or block.block_hash_num_tokens != end:
                    continue
                key = make_offload_key(hashes[row], group.group_idx)
                if block.block_hash == key:
                    projection = self.btc_manager.publish_device(
                        hashes, group.group_idx, row, block
                    )
                    if projection is not None:
                        self.scheduler.offer_writeback(request, projection, end)

    def try_joint_prefix_match(
        self, request: "Request", max_cache_hit_length: int | None
    ) -> tuple[KVCacheBlocks, int]:
        kv = self._kv_cache_manager
        assert kv is not None
        groups = self.scheduler.config.kv_group_configs
        managers = kv.coordinator.single_type_managers
        step = self.scheduler.config.tokens_per_hash
        hashes = request.block_hashes
        status = self.scheduler._req_status[request.request_id]
        if not kv.prefix_cache_lookup_enabled(request) or status.transfer_jobs:
            self._plans[request.request_id] = AdmissionPlan(0, 0, ())
            return kv.empty_kv_cache_blocks, 0
        maximum = min(
            request.get_max_cache_hit_length(max_cache_hit_length), len(hashes) * step
        )

        endpoints = []
        for end in range(self._alignment, maximum + 1, self._alignment):
            required_groups = []
            for manager, group in zip(managers, groups):
                size = group.tokens_per_block
                start = max(0, manager.get_num_skipped_tokens(end) // size)
                required_groups.append(
                    tuple(
                        (pos + 1) * size // step - 1
                        for pos in range(start, end // size)
                    )
                )
            endpoints.append((end, tuple(required_groups)))
        manager = self.btc_manager
        match = (
            manager.tree.match_with_join
            if manager.share_device_loads
            else manager.tree.match
        )
        device, matched, device_bindings, bindings = match(hashes, endpoints)
        pages = self._make_page_bindings(bindings)
        if (
            request.skip_reading_external_kv
            or status.load_failed
            or not self.scheduler._offloading_enabled
            or (
                status.max_load_tokens is not None
                and matched - device > status.max_load_tokens
            )
        ):
            matched, pages = device, self._make_page_bindings(device_bindings)
        plan = AdmissionPlan(device, matched, pages, next(self._plan_ids))
        self._plans[request.request_id] = plan
        if self.btc_manager._trace_transfers:
            logger.info(
                "BTC_JOINT_MATCH request=%s plan=%s",
                request.request_id,
                _json(
                    {
                        "device_tokens": device,
                        "matched_tokens": matched,
                        "native_calls": 1,
                        "plan_id": plan.plan_id,
                        "bindings": [page.trace() for page in pages],
                    }
                ),
            )

        if not matched:
            return kv.empty_kv_cache_blocks, 0
        pool = kv.block_pool
        rows = [
            [pool.null_block] * (matched // group.tokens_per_block) for group in groups
        ]
        loads: list[list[int]] = [[] for _ in groups]
        joins: list[list[int]] = [[] for _ in groups]
        for page in pages:
            if page.kind is PageKind.LOAD:
                loads[page.group].append(page.position)
                continue
            block = pool.blocks[page.source_id]
            key = make_offload_key(hashes[page.hash_row], page.group)
            if page.kind is PageKind.JOIN:
                self._shared_target(page, key)
                joins[page.group].append(page.position)
            else:
                assert block.block_hash == key
                if manager.share_device_loads:
                    projection = manager._device_slots[page.source_id]
                    assert projection.identity() == (
                        page.group,
                        page.source_id,
                        page.generation,
                    )
            rows[page.group][page.position] = block
        load_positions = tuple(map(tuple, loads)) if any(loads) else None
        join_positions = tuple(map(tuple, joins)) if any(joins) else None
        return KVCacheBlocks(tuple(rows), load_positions, join_positions), device

    def _make_page_bindings(
        self, bindings: Sequence[tuple[int, ...]]
    ) -> tuple[PageBinding, ...]:
        config = self.scheduler.config
        pages = []
        for binding in bindings:
            if len(binding) == 7:
                group, row, kind, tier, source_id, generation, operation = binding
            else:
                group, row, tier, source_id = binding
                kind = PageKind.READY if tier == Tier.DEVICE else PageKind.LOAD
                generation, operation = 0, 0
            size = config.kv_group_configs[group].tokens_per_block
            pages.append(
                PageBinding(
                    group,
                    (row + 1) * config.tokens_per_hash // size - 1,
                    row,
                    Tier(tier),
                    source_id,
                    kind=PageKind(kind),
                    generation=generation,
                    operation_id=operation,
                )
            )
        return tuple(pages)

    def _shared_target(self, page: PageBinding, key: OffloadKey) -> PageBinding:
        source = self.btc_manager._pending_device_pages.get(key)
        if (
            source is not None
            and source.operation_id == page.operation_id
            and source.group == page.group
            and source.destination_id == page.source_id
            and source.generation == page.generation
        ):
            return source
        raise RuntimeError("BTC JOIN page has no live matching GPU destination")

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        status = self.scheduler._req_status[request.request_id]
        if status.transfer_jobs:
            return None, False
        status.num_locally_computed_tokens = num_computed_tokens
        plan = self._plans[request.request_id]
        return plan.external_tokens, bool(plan.external_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        plan = self._plans.pop(request.request_id, None)
        if plan is None:
            assert num_external_tokens == 0
            return
        assert plan.external_tokens == num_external_tokens
        pages = tuple(
            replace(
                page, destination_id=blocks.blocks[page.group][page.position].block_id
            )
            for page in plan.pages
        )
        if num_external_tokens:
            pages = self._schedule_bound_load(request, plan, pages)
        if self.btc_manager._trace_transfers and pages:
            logger.info(
                "BTC_JOINT_ALLOC request=%s pages=%s",
                request.request_id,
                _json([dict(page.trace(), plan_id=plan.plan_id) for page in pages]),
            )

    def _schedule_bound_load(
        self, request: "Request", plan: AdmissionPlan, pages: tuple[PageBinding, ...]
    ) -> tuple[PageBinding, ...]:
        scheduler = self.scheduler
        status = scheduler._req_status[request.request_id]
        status.load_source_ranges.clear()
        loads = []
        joins = []
        # GDN state and any holes below D are read dependencies for the entire
        # external interval, even though their physical spans may be shorter.
        for page in pages:
            if page.kind is PageKind.READY:
                continue
            key = make_offload_key(request.block_hashes[page.hash_row], page.group)
            if page.kind is PageKind.JOIN:
                assert page.source_id == page.destination_id
                source = self._shared_target(page, key)
                joins.append(page)
            else:
                source = page
                loads.append((page, key))
            group = scheduler.config.kv_group_configs[page.group]
            lo = page.position * group.tokens_per_block
            hi = lo + group.tokens_per_block
            if (
                group.sliding_window_size_in_chunks is not None
                or hi <= plan.device_tokens
            ):
                lo, hi = plan.device_tokens, plan.matched_tokens
            source_tier = (
                CacheHitSource.DISK if source.tier is Tier.DISK else CacheHitSource.HOST
            )
            status.load_source_ranges.append((lo, hi, source_tier))
        transfer = None
        if loads:
            loads.sort(key=lambda entry: (entry[0].group, entry[0].position))
            load_pages, load_keys = zip(*loads)
            try:
                transfer = self._submit_new_load(request, load_pages, list(load_keys))
            except BoundLoadReadError as error:
                # The scheduler parks this request before it can consume a
                # failure receipt. Existing JOIN operations keep their pins.
                status.load_failed = True
                self._failed_bound_load_requests[request.request_id] = request
                logger.warning(
                    "BTC bound load for %s failed: %s", request.request_id, error
                )
                return pages

        if self.btc_manager.share_device_loads:
            operations = {page.operation_id for page in joins}
            if transfer is not None:
                operations.add(transfer.operation_id)
                bound = {(page.group, page.position): page for page in transfer.pages}
                pages = tuple(
                    bound.get((page.group, page.position), page) for page in pages
                )
            self._attach_waiter(request.request_id, operations, pages)
        return pages

    def _submit_new_load(
        self, request: "Request", loads: tuple[PageBinding, ...], keys: list[OffloadKey]
    ) -> SharedLoad | None:
        scheduler = self.scheduler
        status = scheduler._req_status[request.request_id]
        manager = self.btc_manager
        num_groups = len(scheduler.config.kv_group_configs)
        group_sizes = [0] * num_groups
        block_indices = [0] * num_groups
        target_ids = []
        for page in loads:
            assert page.destination_id is not None and page.destination_id != 0
            if group_sizes[page.group] == 0:
                block_indices[page.group] = page.position
            group_sizes[page.group] += 1
            target_ids.append(page.destination_id)
        target = GPULoadStoreSpec(
            target_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        job_id = scheduler._generate_job_id()
        transfer = None
        if manager.share_device_loads:
            transfer = manager.begin_shared_load(request.block_hashes, loads, job_id + 1)
            loads = transfer.pages
        try:
            source = manager.prepare_bound_load(loads, request.request_id)
        except Exception:
            if transfer is not None:
                manager.settle_shared_load(transfer, False)
            raise

        assert not status.transfer_jobs
        scheduler.enqueue_load(
            job_id,
            TransferJob(req_id=request.request_id, src_spec=source, dst_spec=target),
            tuple(dict.fromkeys(keys)),
        )
        if transfer is not None:
            self._shared_loads[transfer.operation_id] = transfer
            if manager._trace_transfers:
                logger.info(
                    "BTC_JOIN_SUBMIT operation=%d info=%s",
                    transfer.operation_id,
                    _json(
                        {
                            "owner": request.request_id,
                            "pages": [page.trace() for page in transfer.pages],
                        }
                    ),
                )
        return transfer

    def _attach_waiter(
        self, request_id: str, operations: set[int], pages: tuple[PageBinding, ...]
    ) -> None:
        assert operations and request_id not in self._shared_waiters
        operation_ids = tuple(sorted(operations))
        self._shared_waiters[request_id] = SharedLoadWaiter(operation_ids, operations)
        for operation_id in operation_ids:
            self._shared_loads[operation_id].waiters.add(request_id)
        if self.btc_manager._trace_transfers:
            logger.info(
                "BTC_JOIN_ATTACH request=%s info=%s",
                request_id,
                _json(
                    {
                        "operations": operation_ids,
                        "pages": [
                            page.trace()
                            for page in pages
                            if page.kind is not PageKind.READY
                        ],
                    }
                ),
            )

    def prepare_scheduler_output(
        self, output: KVConnectorOutput | None
    ) -> KVConnectorOutput | None:
        failures = self._failed_bound_load_requests
        if output is None:
            if not failures:
                return None
            output = KVConnectorOutput()
        self._resolve_load_output(output)
        if failures:
            if output.finished_recving is None:
                output.finished_recving = set()
            output.finished_recving.update(failures)
            output.failed_recving.update(failures)
            for request in failures.values():
                request.num_computed_tokens = 0
            failures.clear()
        return output

    def _resolve_load_output(self, output: KVConnectorOutput) -> None:
        # Raw worker receives are for individual jobs and name their creators.
        # Only the composite per-request completion can wake a shared waiter.
        output.finished_recving = set()
        output.failed_recving.clear()
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, OffloadingWorkerMetadata):
            return
        for job_id, count in metadata.completed_jobs.items():
            status = self.scheduler._job_statuses.get(job_id)
            if status is None or status.is_store:
                continue
            if count < status.pending_count:
                continue
            assert count == status.pending_count
            success = not status.failed and job_id not in metadata.failed_jobs
            transfer = self._shared_loads.get(job_id + 1)
            if transfer is not None:
                self._finish_shared_load(transfer, success, output)
            else:
                output.finished_recving.add(status.req_id)
                if not success:
                    output.failed_recving.add(status.req_id)

    def _finish_shared_load(
        self, transfer: SharedLoad, success: bool, output: "KVConnectorOutput"
    ) -> None:
        manager = self.btc_manager
        waiters = {req_id: self._shared_waiters[req_id] for req_id in transfer.waiters}
        active = sorted(
            req_id for req_id, waiter in waiters.items() if not waiter.cancelled
        )
        publish = bool(active) and success
        manager.settle_shared_load(transfer, publish)
        del self._shared_loads[transfer.operation_id]
        if manager._trace_transfers:
            logger.info(
                "BTC_JOIN_COMPLETE operation=%d info=%s",
                transfer.operation_id,
                _json(
                    {
                        "success": success,
                        "published": publish,
                        "active_waiters": active,
                        "cancelled_waiters": sorted(set(waiters) - set(active)),
                        "pages": [page.trace() for page in transfer.pages],
                        "target_refs_after": {
                            target.block.block_id: target.block.ref_cnt
                            for target in transfer.targets
                        },
                    }
                ),
            )
        assert output.finished_recving is not None
        for req_id, waiter in waiters.items():
            waiter.pending.remove(transfer.operation_id)
            waiter.failed |= not success
            if waiter.pending:
                continue
            output.finished_recving.add(req_id)
            if waiter.failed and not waiter.cancelled:
                output.failed_recving.add(req_id)
            del self._shared_waiters[req_id]
            if manager._trace_transfers:
                logger.info(
                    "BTC_JOIN_WAKE request=%s info=%s",
                    req_id,
                    _json(
                        {
                            "success": not waiter.failed,
                            "cancelled": waiter.cancelled,
                            "operations": waiter.operations,
                        }
                    ),
                )

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self.scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._plans.pop(request.request_id, None)
        if request.request_id in self._failed_bound_load_requests:
            # An early cancellation must not offer the unwritten blocks to
            # a finished-request store trigger before the failure is drained.
            request.num_computed_tokens = 0
        waiter = self._shared_waiters.get(request.request_id)
        if waiter is not None:
            waiter.cancelled = True
            if self.btc_manager._trace_transfers:
                target_refs = {
                    target.block.block_id: target.block.ref_cnt
                    for operation_id in waiter.pending
                    for target in self._shared_loads[operation_id].targets
                }
                logger.info(
                    "BTC_JOIN_DETACH request=%s info=%s",
                    request.request_id,
                    _json(
                        {
                            "operations": sorted(waiter.pending),
                            "held_target_refs": target_refs,
                        }
                    ),
                )
        self.scheduler.request_finished(request)
        return False, None

    def reset_cache(self) -> bool | None:
        if self._shared_loads or self._failed_bound_load_requests:
            return False
        if self.btc_manager.reset_blocked():
            return False
        return self.scheduler.reset_cache()

    def has_pending_push_work(self) -> bool:
        return bool(
            self._failed_bound_load_requests
        ) or self.scheduler.has_pending_push_work()

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        # Any leftover proposed plans were skipped or rejected. There are no
        # retained references for an unallocated attempt. The sole native
        # deferred-bind path is HiSparse, excluded by this connector.
        self._plans.clear()
        kv = self._kv_cache_manager
        assert kv is not None
        # Pin admitted Host-to-Disk sources before reserving Host slots for
        # new GPU writebacks, whose reservation may evict idle Host pages.
        disk_jobs = self.btc_manager.take_disk_jobs(self.scheduler._generate_job_id)
        metadata = self.scheduler.build_connector_meta(scheduler_output, kv.block_pool)
        if not disk_jobs:
            return metadata
        return BtcConnectorMetadata(
            load_jobs=metadata.load_jobs,
            store_jobs=metadata.store_jobs,
            jobs_to_flush=metadata.jobs_to_flush,
            offloading_enabled=metadata.offloading_enabled,
            disk_jobs=disk_jobs,
        )

    def take_events(self) -> Iterable["KVCacheEvent"]:
        return self.scheduler.take_events()

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_stats()
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats":
        if data is None:
            return OffloadingConnectorStats()
        return OffloadingConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics":
        return OffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
