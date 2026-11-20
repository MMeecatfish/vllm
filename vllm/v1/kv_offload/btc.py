# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BTC worker spec and its native CUDA/WorkLoop backend."""

import os

from typing_extensions import override

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


class BtcOffloadingSpec(CPUOffloadingSpec):
    """Construct each rank's native BTC transfer worker."""

    def _create_layout_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        worker = super().create_worker(kv_caches)
        assert isinstance(worker, CPUOffloadingWorker)
        return worker

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        from vllm.v1.kv_offload.btc_native_worker import (
            BtcNativeOffloadingWorker,
            BtcWorkerDiskConfig,
            NativeTransferLibrary,
        )

        if not current_platform.is_cuda():
            raise ValueError("BTC GPU/host transfers require NVIDIA CUDA")
        library_path = self.extra_config.get("btc_transfer_library") or os.getenv(
            "RTP_BTC_TRANSFER_LIBRARY"
        )
        if not library_path:
            raise ValueError("Set RTP_BTC_TRANSFER_LIBRARY or btc_transfer_library")
        library = NativeTransferLibrary(library_path)
        layout = self._create_layout_worker(kv_caches)
        try:
            disk_dir = self.extra_config.get("btc_disk_dir")
            disk = (
                BtcWorkerDiskConfig(
                    disk_dir,
                    int(self.extra_config["btc_disk_slots"]),
                    int(self.extra_config["btc_disk_staging_chunks"]),
                )
                if disk_dir
                else None
            )
            return BtcNativeOffloadingWorker(
                layout,
                library,
                rank=self.config.parallel.rank,
                trace_transfers=bool(self.extra_config.get("btc_trace_transfers")),
                disk=disk,
            )
        except Exception:
            layout.shutdown()
            raise

    @override
    def get_manager(self) -> OffloadingManager:
        raise ValueError("BTC requires BtcJointConnector for scheduler integration")
