# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt vLLM KV layouts and connector jobs to a rank-local BTC runtime."""

from __future__ import annotations

import ctypes
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    DevicePointers,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.btc_transfer_spec import BtcLoadSpec
from vllm.v1.kv_offload.pointer_utils import iter_pointer_groups

if TYPE_CHECKING:
    from vllm.v1.kv_offload.cpu.gpu_worker import (
        CPUOffloadingWorker,
        SingleDirectionOffloadingHandler,
    )
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

_Handle = ctypes.c_void_p
_Size = ctypes.c_size_t
_SizePtr = ctypes.POINTER(_Size)
_DiskPtr = ctypes.POINTER(ctypes.c_int64)
_OutHandle = ctypes.POINTER(_Handle)


@dataclass(frozen=True)
class BtcWorkerDiskConfig:
    directory: str
    slots: int
    staging_slots: int


@dataclass(frozen=True)
class _TransferSpans:
    host_rows: np.ndarray
    disk_ids: np.ndarray
    span_pages: np.ndarray
    span_offsets: np.ndarray
    gpu_ptrs: np.ndarray
    sizes: np.ndarray
    total_bytes: int


def _uint_array(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uintp)


def _disk_array(values: list[int] | np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


class NativeTransferLibrary:
    """Bind the C ABI for a worker-local BTC instance and transfer tickets."""

    def __init__(self, path: str):
        self._lib = ctypes.CDLL(path)
        self._create = self._bind(
            "create", ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, _Size, _Size, _Size,
            _OutHandle,
        )
        copy_args = (
            _Handle, _Size, _Size, _SizePtr, _DiskPtr,
            _Size, _SizePtr, _SizePtr, _SizePtr, _SizePtr, ctypes.c_uint64,
            _OutHandle,
        )
        self._load = self._bind("submit_load", ctypes.c_int, *copy_args)
        self._store = self._bind("submit_store", ctypes.c_int, *copy_args)
        self._disk_store = self._bind(
            "submit_disk_store", ctypes.c_int,
            _Handle, _Size, _SizePtr, _DiskPtr, ctypes.c_uint64, _OutHandle,
        )
        self._disk_release = self._bind(
            "release_disk", ctypes.c_int,
            _Handle, _Size, _DiskPtr,
        )
        self.query = self._bind("query", ctypes.c_int, _Handle)
        self.wait = self._bind("wait", ctypes.c_int, _Handle)
        self.destroy_ticket = self._bind("destroy_ticket", None, _Handle)
        self.destroy_rank = self._bind("destroy", None, _Handle)
        self._last_error = self._bind("last_error", ctypes.c_char_p)

    def _bind(self, name: str, result, *args):
        function = getattr(self._lib, f"btc_rank_{name}")
        function.restype = result
        function.argtypes = list(args)
        return function

    def last_error(self) -> str:
        return (self._last_error() or b"unknown BTC transfer error").decode(
            errors="replace"
        )

    def create(
        self,
        device: int,
        rank: int,
        disk: BtcWorkerDiskConfig | None,
        row_bytes: int,
    ) -> _Handle:
        handle = _Handle()
        status = self._create(
            device,
            rank,
            os.fsencode(disk.directory) if disk else None,
            disk.slots if disk else 0,
            row_bytes,
            disk.staging_slots if disk else 0,
            ctypes.byref(handle),
        )
        if status != 0 or not handle:
            if handle:
                self.destroy_rank(handle)
            raise RuntimeError(f"Cannot create BTC runtime for rank {rank}: {self.last_error()}")
        return handle

    def _check_ticket(self, status: int, ticket: _Handle) -> _Handle | None:
        if status == 0 and ticket:
            return ticket
        if ticket:
            self.destroy_ticket(ticket)
        if status == -1:
            return None
        raise RuntimeError(f"Unexpected BTC submission status {status}")

    def submit_copy(
        self,
        handle: _Handle,
        job: int,
        stream: int,
        spans: _TransferSpans,
        *,
        store: bool,
    ) -> _Handle | None:
        ticket = _Handle()
        submit = self._store if store else self._load
        status = submit(
            handle, stream,
            spans.host_rows.size,
            spans.host_rows.ctypes.data_as(_SizePtr),
            spans.disk_ids.ctypes.data_as(_DiskPtr),
            spans.sizes.size,
            spans.span_pages.ctypes.data_as(_SizePtr),
            spans.span_offsets.ctypes.data_as(_SizePtr),
            spans.gpu_ptrs.ctypes.data_as(_SizePtr),
            spans.sizes.ctypes.data_as(_SizePtr),
            job,
            ctypes.byref(ticket),
        )
        return self._check_ticket(status, ticket)

    def submit_disk_store(
        self, handle: _Handle, job: int, host_rows: np.ndarray, disk_ids: np.ndarray
    ) -> _Handle | None:
        ticket = _Handle()
        status = self._disk_store(
            handle,
            host_rows.size,
            host_rows.ctypes.data_as(_SizePtr),
            disk_ids.ctypes.data_as(_DiskPtr),
            job,
            ctypes.byref(ticket),
        )
        return self._check_ticket(status, ticket)

    def release_disk(self, handle: _Handle, disk_ids: np.ndarray) -> bool:
        status = self._disk_release(
            handle, disk_ids.size, disk_ids.ctypes.data_as(_DiskPtr)
        )
        if status in (0, -1):
            return status == 0
        raise RuntimeError(f"Unexpected BTC disk release status {status}")


@dataclass
class _PendingTransfer:
    ticket: _Handle | None
    num_bytes: int
    submitted: float
    status: int = 0


class BtcNativeOffloadingWorker(OffloadingWorker):
    """The Python worker supplies borrowed KV addresses and returns BTC receipts."""

    def __init__(
        self,
        layout: CPUOffloadingWorker,
        library: NativeTransferLibrary,
        *,
        rank: int,
        trace_transfers: bool,
        disk: BtcWorkerDiskConfig | None = None,
    ):
        self._layout = layout
        self._library = library
        self._rank = rank
        self._device = torch.cuda.current_device()
        self._trace_transfers = trace_transfers
        self._pending: dict[int, _PendingTransfer] = {}
        self._disk_row_bytes = 0
        self._host_row_base = 0
        self._disk_directory: tempfile.TemporaryDirectory | None = None
        if disk:
            region = self._mmap_region
            if region is None or region.rank != 0:
                raise ValueError("BTC Disk requires rank-local contiguous KV rows")
            self._disk_row_bytes, remainder = divmod(
                region.total_size_bytes, region.num_chunks
            )
            if remainder or self._disk_row_bytes < region.cpu_page_size:
                raise ValueError("Invalid BTC Disk Host row stride")
            self._host_row_base = region.base_tensor.data_ptr()
            self._host_row_count = region.num_chunks
            os.makedirs(disk.directory, exist_ok=True)
            self._disk_directory = tempfile.TemporaryDirectory(
                prefix=f"btc-vllm-rank-{rank}-", dir=disk.directory
            )
            disk = BtcWorkerDiskConfig(
                self._disk_directory.name, disk.slots, disk.staging_slots
            )
        try:
            self._handle = library.create(self._device, rank, disk, self._disk_row_bytes)
        except Exception:
            if self._disk_directory is not None:
                self._disk_directory.cleanup()
            raise

    @property
    def _mmap_region(self) -> SharedOffloadRegion | None:
        return self._layout._mmap_region

    def _host_rows(self, host_ids: np.ndarray) -> np.ndarray:
        if (host_ids < 0).any() or (host_ids >= self._host_row_count).any():
            raise ValueError("Invalid BTC Host slot")
        return self._host_row_base + _uint_array(host_ids) * self._disk_row_bytes

    def _pack(
        self,
        layout: SingleDirectionOffloadingHandler,
        device_ptrs: DevicePointers,
        spec: BlockIDsLoadStoreSpec,
    ) -> _TransferSpans:
        host_ids = spec.block_ids
        disk_ids = (
            _disk_array(spec.disk_ids)
            if isinstance(spec, BtcLoadSpec)
            else np.full(host_ids.size, -1, dtype=np.int64)
        )
        if disk_ids.size != host_ids.size:
            raise ValueError("BTC transfer source arrays differ in length")
        has_disk = (disk_ids >= 0).any()
        if has_disk and (not self._disk_row_bytes or layout.gpu_to_cpu):
            raise ValueError("Disk sources require the BTC rank-local load path")
        # The direct layout gives each group the same within-row offsets in every
        # Host slot. Use row zero as the geometry template for Disk sources.
        layout_slots = np.where(disk_ids >= 0, 0, host_ids) if has_disk else host_ids
        capacity = layout._estimate_max_copy_ops(device_ptrs.group_block_counts)
        src = np.empty(capacity, dtype=np.uintp)
        dst = np.empty_like(src)
        sizes = np.empty_like(src)
        span_pages = np.empty_like(src) if self._disk_row_bytes else None
        host_offset = span_offset = total_bytes = 0
        for group in iter_pointer_groups(device_ptrs, layout.blocks_per_chunk):
            host_end = host_offset + group.n_chunks
            if host_end > host_ids.size:
                raise ValueError("BTC job is missing Host or Disk source slots")
            span_end, group_bytes = layout._fill_group_ops(
                group.group_idx,
                device_ptrs=device_ptrs,
                dev_ptr_offset=group.dev_ptr_offset,
                cpu_block_ids=layout_slots[host_offset:host_end],
                group_size=group.group_size,
                cpu_skip_count=group.skip,
                all_src=src,
                all_dst=dst,
                all_sizes=sizes,
                op_idx=span_offset,
            )
            if span_pages is not None:
                if span_end - span_offset != group.group_size * group.n_data_refs:
                    raise ValueError("BTC Disk requires the direct rank-local layout")
                offsets = np.arange(group.group_size, dtype=np.uintp) + group.skip
                page_indices = host_offset + offsets // layout.blocks_per_chunk
                span_pages[span_offset:span_end] = np.tile(
                    page_indices, group.n_data_refs
                )
            host_offset, span_offset = host_end, span_end
            total_bytes += group_bytes
        if host_offset != host_ids.size:
            raise ValueError("BTC job has unused Host or Disk source slots")

        host_ptrs, gpu_ptrs = (dst, src) if layout.gpu_to_cpu else (src, dst)
        host_ptrs = host_ptrs[:span_offset]
        gpu_ptrs = gpu_ptrs[:span_offset]
        sizes = sizes[:span_offset]
        if span_pages is None:
            # Host-only descriptors already contain exact addresses, including
            # layouts backed by separate tensors or shared canonical slices.
            return _TransferSpans(
                host_ptrs,
                np.full(span_offset, -1, dtype=np.int64),
                np.arange(span_offset, dtype=np.uintp),
                np.zeros(span_offset, dtype=np.uintp),
                gpu_ptrs, sizes, total_bytes,
            )

        span_pages = span_pages[:span_offset]
        layout_rows = self._host_rows(layout_slots)
        span_offsets = host_ptrs - layout_rows[span_pages]
        host_rows = np.where(disk_ids >= 0, np.uintp(0), layout_rows)
        return _TransferSpans(
            host_rows, disk_ids, span_pages, span_offsets, gpu_ptrs, sizes, total_bytes,
        )

    def _record(
        self, job_id: int, ticket: _Handle | None, num_bytes: int, submitted: float
    ) -> bool:
        if ticket is None:
            logger.error(
                "BTC submission failed rank=%d job=%d: %s",
                self._rank, job_id, self._library.last_error(),
            )
        self._pending[job_id] = _PendingTransfer(
            ticket, num_bytes, submitted, status=-1 if ticket is None else 0
        )
        # The connector worker observes terminal failures in get_finished().
        return True

    def _submit_copy(
        self,
        job_id: int,
        layout: SingleDirectionOffloadingHandler,
        device_ptrs: DevicePointers,
        spec: LoadStoreSpec,
    ) -> bool:
        assert job_id not in self._pending
        assert isinstance(spec, BlockIDsLoadStoreSpec)
        spans = self._pack(layout, device_ptrs, spec)
        submitted = time.perf_counter()
        ticket = self._library.submit_copy(
            self._handle,
            job_id,
            torch.cuda.current_stream(self._device).cuda_stream,
            spans,
            store=layout.gpu_to_cpu,
        )
        if self._trace_transfers and ticket:
            logger.info(
                "BTC_RANK_WORKER rank=%d pid=%d job=%d action=%s disk_pages=%d "
                "spans=%d bytes=%d",
                self._rank, os.getpid(), job_id,
                "device_to_host" if layout.gpu_to_cpu else "load",
                np.count_nonzero(spans.disk_ids >= 0), spans.sizes.size,
                spans.total_bytes,
            )
        return self._record(job_id, ticket, spans.total_bytes, submitted)

    @override
    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit_copy(job_id, self._layout._store_handler, device_ptrs, dst_spec)

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        device_ptrs: DevicePointers,
        expedited: bool = False,
    ) -> bool:
        handler = (
            self._layout._expedited_load_handler if expedited else self._layout._load_handler
        )
        return self._submit_copy(job_id, handler, device_ptrs, src_spec)

    def submit_disk_store(
        self, job_id: int, host_ids: list[int], disk_ids: list[int]
    ) -> bool:
        assert job_id not in self._pending
        assert self._disk_row_bytes and len(host_ids) == len(disk_ids)
        rows = self._host_rows(np.asarray(host_ids, dtype=np.int64))
        submitted = time.perf_counter()
        ticket = self._library.submit_disk_store(
            self._handle, job_id, rows, _disk_array(disk_ids)
        )
        return self._record(job_id, ticket, 0, submitted)

    def submit_disk_release(self, job_id: int, disk_ids: list[int]) -> bool:
        assert job_id not in self._pending
        assert self._disk_row_bytes
        submitted = time.perf_counter()
        success = self._library.release_disk(self._handle, _disk_array(disk_ids))
        if not success:
            logger.error(
                "BTC disk release failed rank=%d job=%d: %s",
                self._rank, job_id, self._library.last_error(),
            )
        self._pending[job_id] = _PendingTransfer(
            None, 0, submitted, status=1 if success else -1
        )
        return True

    def _update(self, job_id: int, transfer: _PendingTransfer, *, wait: bool) -> None:
        if transfer.status != 0:
            return
        assert transfer.ticket is not None
        operation = self._library.wait if wait else self._library.query
        transfer.status = operation(transfer.ticket)
        if transfer.status == -1:
            logger.error(
                "BTC transfer failed rank=%d job=%d: %s",
                self._rank, job_id, self._library.last_error(),
            )
        elif transfer.status != 1 and (wait or transfer.status != 0):
            raise RuntimeError(f"Unexpected BTC transfer status {transfer.status}")

    @override
    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, transfer in tuple(self._pending.items()):
            self._update(job_id, transfer, wait=False)
            if transfer.status == 0:
                continue
            if transfer.ticket is not None:
                self._library.destroy_ticket(transfer.ticket)
            del self._pending[job_id]
            if transfer.status == 1 and transfer.num_bytes:
                results.append(
                    TransferResult(
                        job_id, True, transfer.num_bytes,
                        time.perf_counter() - transfer.submitted,
                    )
                )
            else:
                results.append(TransferResult(job_id, transfer.status == 1))
        return results

    @override
    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            transfer = self._pending.get(job_id)
            if transfer is not None:
                self._update(job_id, transfer, wait=True)

    @override
    def start_deferred_setup(self) -> None:
        self._layout.start_deferred_setup()

    @override
    def is_ready(self) -> bool:
        return self._layout.is_ready()

    @override
    def shutdown(self) -> None:
        if not self._handle:
            return
        # Drain before releasing the vLLM memory the runtime borrowed.
        for transfer in self._pending.values():
            if transfer.ticket is not None:
                self._library.destroy_ticket(transfer.ticket)
        self._pending.clear()
        self._library.destroy_rank(self._handle)
        self._handle = _Handle()
        if self._disk_directory is not None:
            self._disk_directory.cleanup()
        self._layout.shutdown()
