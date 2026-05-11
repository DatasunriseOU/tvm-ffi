# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file to
# you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression tests for DLPack conversion error paths."""

from __future__ import annotations

import numpy as np
import pytest
import tvm_ffi


def test_from_dlpack_invalid_object_raises_type_error() -> None:
    with pytest.raises(TypeError, match="compatible tensor or PyCapsule"):
        tvm_ffi.from_dlpack(object())


def test_from_dlpack_consumed_legacy_capsule_raises_type_error() -> None:
    torch = pytest.importorskip("torch")

    owner = torch.arange(4, dtype=torch.float32)
    capsule = torch.utils.dlpack.to_dlpack(owner)
    tvm_ffi.from_dlpack(capsule)

    with pytest.raises(TypeError, match="compatible tensor or PyCapsule"):
        tvm_ffi.from_dlpack(capsule)


def test_from_dlpack_consumed_versioned_capsule_raises_type_error() -> None:
    base = tvm_ffi.from_dlpack(np.arange(4, dtype=np.float32))
    capsule = base.__dlpack__(max_version=tvm_ffi.core.__dlpack_version__)
    tvm_ffi.from_dlpack(capsule)

    with pytest.raises(TypeError, match="compatible tensor or PyCapsule"):
        tvm_ffi.from_dlpack(capsule)


def test_from_dlpack_exchange_api_failure_raises_python_exception() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        pytest.skip("torch.Tensor.__dlpack_c_exchange_api__ not available")

    class BadTorchExchange:
        __dlpack_c_exchange_api__ = torch.Tensor.__dlpack_c_exchange_api__

    with pytest.raises(RuntimeError):
        tvm_ffi.from_dlpack(BadTorchExchange())
