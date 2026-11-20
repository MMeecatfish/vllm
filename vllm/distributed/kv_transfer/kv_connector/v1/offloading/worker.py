# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    build_canonical_kv_caches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.pointer_utils import resolve_device_pointers

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        # Non-writers still ack: pending_count waits for world_size per job.
        self._is_store_writer = (
            not self.spec.replicated_layout
            or self.spec.config.canonical_layout
            or self.spec.config.parallel.rank == 0
        )

        # job_id -> req_id for in-flight loads (None for sync loads).
        self._load_jobs: dict[int, ReqId | None] = {}
        self._failed_recving: set[ReqId] = set()
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()
        # Under async_init, report readiness once.
        self._report_ready = self.spec.async_init

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self._kv_caches = kv_caches
        self.worker = self.spec.get_worker(kv_caches)
        self.worker.start_deferred_setup()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._init_worker(
            build_canonical_kv_caches(
                self.vllm_config,
                self.kv_cache_config,
                kv_caches,
                get_offloading_group_ids(self.kv_cache_config),
            )
        )

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            device_ptrs = resolve_device_pointers(src_spec, self._kv_caches)
            success = self.worker.submit_store(job_id, device_ptrs, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        if not metadata.offloading_enabled:
            assert not metadata.store_jobs and not metadata.load_jobs
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            device_ptrs = resolve_device_pointers(src_spec, self._kv_caches)
            success = self.worker.submit_store(job_id, device_ptrs, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        sync_job_ids: set[int] = set()
        for job_id, entry in metadata.load_jobs.items():
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            device_ptrs = resolve_device_pointers(entry.dst_spec, self._kv_caches)
            if entry.sync:
                # Its request already runs this step: no finished_recving.
                sync_job_ids.add(job_id)
                self._load_jobs[job_id] = None
                success = self.worker.submit_load(
                    job_id, entry.src_spec, device_ptrs, expedited=True
                )
            else:
                self._load_jobs[job_id] = entry.req_id
                success = self.worker.submit_load(job_id, entry.src_spec, device_ptrs)
            assert success
        if sync_job_ids:
            self.worker.wait(sync_job_ids)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Returns:
        tuple of (finished_sending, finished_recving). Stores never
        emit finished_sending — the scheduler tracks store completion
        via kv_connector_worker_meta.completed_jobs and fences any
        block reuse via jobs_to_flush. Loads still emit
        finished_recving so the base scheduler can resume requests
        blocked on remote KV (and free aborted-during-load reqs).

        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            is_load = job_id in self._load_jobs
            if not transfer_result.success:
                self._connector_worker_meta.mark_completed(job_id, success=False)
                req_id = self._load_jobs.pop(job_id, None)
                if is_load and req_id is None:
                    # Its request already ran this step on the unloaded KV,
                    # which failed_recving cannot undo.
                    raise RuntimeError(f"KV offload sync load job {job_id} failed")
                if req_id is not None:
                    self._failed_recving.add(req_id)
                    finished_recving.add(req_id)
                continue
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def take_failed_recving(self) -> set[ReqId]:
        """Drain the requests whose load failed since the last call."""
        failed_recving, self._failed_recving = self._failed_recving, set()
        return failed_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return job completions, and readiness under async_init, since the
        last call."""
        if self._report_ready and self.worker is not None and self.worker.is_ready():
            self._connector_worker_meta.ready_ranks = {self.spec.config.parallel.rank}
            self._report_ready = False
        if (
            not self._connector_worker_meta.completed_jobs
            and not self._connector_worker_meta.ready_ranks
        ):
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._failed_recving.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
