# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side vLLM job dispatch and rank receipts for native BTC transfers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    build_canonical_kv_caches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.btc_transfer_spec import BtcConnectorMetadata
from vllm.v1.kv_offload.pointer_utils import resolve_device_pointers

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class _ActiveJob:
    is_load: bool = False
    request_id: str | None = None


class BtcConnectorWorker:
    """Borrow vLLM KV pages while this rank's BTC worker owns their transfers."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        self._spec = spec
        self._vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        self._rank = spec.config.parallel.rank
        self._is_store_writer = (
            not spec.replicated_layout or spec.config.canonical_layout or self._rank == 0
        )
        self._report_ready = spec.async_init
        self._worker: OffloadingWorker | None = None
        self._kv_caches: CanonicalKVCaches | None = None
        self._pending_stores: dict[int, TransferJob] = {}
        self._active_jobs: dict[int, _ActiveJob] = {}
        self._failed_recving: set[str] = set()
        self._receipts = OffloadingWorkerMetadata()

    @property
    def worker(self) -> OffloadingWorker:
        assert self._worker is not None, "BTC worker requires registered KV caches"
        return self._worker

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        assert self._worker is None, "BTC worker already has registered KV caches"
        canonical = build_canonical_kv_caches(
            self._vllm_config,
            self._kv_cache_config,
            kv_caches,
            get_offloading_group_ids(self._kv_cache_config),
        )
        # Use the BTC spec factory so deferred setup and test worker wrappers
        # apply to the same native worker that owns the transfer tickets.
        worker = self._spec.get_worker(canonical)
        try:
            worker.start_deferred_setup()
        except Exception:
            worker.shutdown()
            raise
        self._kv_caches = canonical
        self._worker = worker

    def _check_new_job(self, job_id: int) -> None:
        if job_id in self._active_jobs or job_id in self._receipts.completed_jobs:
            raise RuntimeError(f"BTC received duplicate rank job {job_id}")

    def _queue_store(self, job_id: int, job: TransferJob) -> None:
        self._check_new_job(job_id)
        if not self._is_store_writer:
            # Even a non-writer contributes this rank's ACK to the store quorum.
            self._receipts.mark_completed(job_id)
            return
        if job_id in self._pending_stores:
            raise RuntimeError(f"BTC received duplicate deferred store {job_id}")
        self._pending_stores[job_id] = job

    def _submit_stores(self) -> None:
        assert self._kv_caches is not None
        for job_id, job in self._pending_stores.items():
            self._check_new_job(job_id)
            assert isinstance(job.src_spec, GPULoadStoreSpec)
            pointers = resolve_device_pointers(job.src_spec, self._kv_caches)
            if not self.worker.submit_store(job_id, pointers, job.dst_spec):
                raise RuntimeError(f"BTC worker refused D2H job {job_id}")
            self._active_jobs[job_id] = _ActiveJob()
        self._pending_stores.clear()

    def handle_preemptions(self, metadata: OffloadingConnectorMetadata) -> None:
        for job_id in metadata.jobs_to_flush or ():
            job = metadata.store_jobs.pop(job_id, None)
            if job is not None:
                self._queue_store(job_id, job)
        self._submit_stores()
        if metadata.jobs_to_flush:
            self.worker.wait(metadata.jobs_to_flush)

    def _submit_disk_jobs(self, metadata: BtcConnectorMetadata) -> None:
        submit_store = getattr(self.worker, "submit_disk_store", None)
        submit_release = getattr(self.worker, "submit_disk_release", None)
        if not callable(submit_store) or not callable(submit_release):
            raise RuntimeError("BTC worker lacks rank-local Disk transfer support")
        for job_id, job in metadata.disk_jobs.items():
            self._check_new_job(job_id)
            disk_ids = list(job.disk_ids)
            if job.host_ids:
                submitted = submit_store(job_id, list(job.host_ids), disk_ids)
            else:
                submitted = submit_release(job_id, disk_ids)
            if not submitted:
                raise RuntimeError(f"BTC worker refused Disk job {job_id}")
            self._active_jobs[job_id] = _ActiveJob()

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata) -> None:
        assert self._kv_caches is not None
        if not metadata.offloading_enabled:
            assert not metadata.store_jobs and not metadata.load_jobs
        if isinstance(metadata, BtcConnectorMetadata) and metadata.disk_jobs:
            self._submit_disk_jobs(metadata)
        self._submit_stores()
        synchronous = set()
        for job_id, job in metadata.load_jobs.items():
            self._check_new_job(job_id)
            assert isinstance(job.dst_spec, GPULoadStoreSpec)
            pointers = resolve_device_pointers(job.dst_spec, self._kv_caches)
            if not self.worker.submit_load(
                job_id, job.src_spec, pointers, expedited=job.sync
            ):
                raise RuntimeError(f"BTC worker refused H2D job {job_id}")
            if job.sync:
                synchronous.add(job_id)
            self._active_jobs[job_id] = _ActiveJob(
                is_load=True, request_id=None if job.sync else job.req_id
            )
        if synchronous:
            self.worker.wait(synchronous)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata) -> None:
        # Defer ordinary stores until the next engine step so they do not hold
        # up sampling. Preemption can force a store before reusing its GPU pages.
        for job_id, job in metadata.store_jobs.items():
            self._queue_store(job_id, job)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_recving = set()
        for result in self.worker.get_finished():
            job_id = result.job_id
            job = self._active_jobs.pop(job_id, None)
            if job is None:
                raise RuntimeError(f"BTC worker returned an unknown rank job {job_id}")
            self._receipts.mark_completed(job_id, success=result.success)
            if result.success:
                if result.transfer_size is not None and result.transfer_time is not None:
                    stats = self._receipts.transfer_stats
                    direction = stats.load if job.is_load else stats.store
                    direction.record(result.transfer_size, result.transfer_time)
            elif job.is_load and job.request_id is None:
                raise RuntimeError(f"BTC synchronous load job {job_id} failed")
            if job.request_id is not None:
                finished_recving.add(job.request_id)
                if not result.success:
                    self._failed_recving.add(job.request_id)
        # Stores, Disk operations, and synchronous loads are job receipts only.
        return set(), finished_recving

    def take_failed_recving(self) -> set[str]:
        failed_recving, self._failed_recving = self._failed_recving, set()
        return failed_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if self._report_ready and self._worker is not None and self._worker.is_ready():
            self._receipts.ready_ranks.add(self._rank)
            self._report_ready = False
        if not self._receipts.completed_jobs and not self._receipts.ready_ranks:
            return None
        metadata = self._receipts
        self._receipts = OffloadingWorkerMetadata()
        return metadata

    def shutdown(self) -> None:
        # The runtime drains outstanding transfers before GPU views may be freed.
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
        self._kv_caches = None
        self._pending_stores.clear()
        self._active_jobs.clear()
        self._failed_recving.clear()
        self._receipts = OffloadingWorkerMetadata()
