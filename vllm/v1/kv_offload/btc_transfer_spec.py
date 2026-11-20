# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BTC transfer metadata carried by the vLLM offloading connector."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec


class BtcLoadSpec(CPULoadStoreSpec):
    """One Host row or logical Disk page per destination, in GPU spec order."""

    def __init__(self, block_ids: Sequence[int], disk_ids: Sequence[int]) -> None:
        super().__init__(list(block_ids))
        self.disk_ids = np.asarray(disk_ids, dtype=np.int64)


@dataclass(frozen=True)
class BtcDiskOperation:
    disk_ids: tuple[int, ...]
    # A release has no Host sources.
    host_ids: tuple[int, ...] = ()


@dataclass
class BtcConnectorMetadata(OffloadingConnectorMetadata):
    disk_jobs: dict[int, BtcDiskOperation] = field(default_factory=dict)
