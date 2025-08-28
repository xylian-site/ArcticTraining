# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from arctic_training.config.base import BaseConfig
from arctic_training.config.utils import HumanInt


class ProfilerConfig(BaseConfig):
    """Profiler configuration for training."""

    enable: bool = Field(default=False)
    """ Enable PyTorch profiler for detailed performance analysis. """

    start_step: int = Field(default=4)
    """ Step to start profiling. """

    end_step: int = Field(default=5)
    """ Step to end profiling. """

    output_dir: Path = Field(default=Path("./profiler_traces"))
    """ Directory to save profiler traces. """

    schedule_wait: int = Field(default=1)
    """ Number of steps to wait before starting profiling. """

    schedule_warmup: int = Field(default=1)
    """ Number of warmup steps for profiler. """

    schedule_active: int = Field(default=2)
    """ Number of active profiling steps. """

    record_shapes: bool = Field(default=True)
    """ Whether to record tensor shapes in profiler. """

    profile_memory: bool = Field(default=True)
    """ Whether to profile memory usage. """

    with_stack: bool = Field(default=True)
    """ Whether to record stack traces. """

    with_flops: bool = Field(default=True)
    """ Whether to record FLOP counts. """


class MemoryProfilerConfig(BaseConfig):
    """Memory profiler configuration for training."""

    enable: Literal[None, "step", "e2e"] = None
    """ Enable memory profiling. Options: None (disabled), "step" (per-step), "e2e" (end-to-end). """

    output_dir: Path = Field(default=Path("./mem_snapshots"))
    """ Directory to save memory profiling snapshots. """

    max_entries: HumanInt = Field(default=100_000, ge=1)
    """ Maximum number of entries to store in the memory profiler. """
