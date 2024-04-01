from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)

class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out

def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _map(out: Storage, out_shape: Shape, out_strides: Strides, in_storage: Storage, in_shape: Shape, in_strides: Strides) -> None:
        for i in prange(len(out)):
            idx = np.zeros(MAX_DIMS, dtype=np.int32)
            broadcast_index(i, out_shape, idx)
            in_pos = index_to_position(idx, in_shape, in_strides)
            out_pos = index_to_position(idx, out_shape, out_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)

def tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _zip(out: Storage, out_shape: Shape, out_strides: Strides, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None:
        for i in prange(len(out)):
            idx = np.zeros(MAX_DIMS, dtype=np.int32)
            broadcast_index(i, out_shape, idx)
            a_pos = index_to_position(idx, a_shape, a_strides)
            b_pos = index_to_position(idx, b_shape, b_strides)
            out_pos = index_to_position(idx, out_shape, out_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)

def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    def _reduce(out: Storage, out_shape: Shape, out_strides: Strides, a_storage: Storage, a_shape: Shape, a_strides: Strides, reduce_dim: int) -> None:
        for i in prange(len(out)):
            idx = np.zeros(MAX_DIMS, dtype=np.int32)
            broadcast_index(i, out_shape, idx)
            sum_val = 0.0
            for j in range(a_shape[reduce_dim]):
                idx[reduce_dim] = j
                a_pos = index_to_position(idx, a_shape, a_strides)
                sum_val = fn(sum_val, a_storage[a_pos])
            out_pos = index_to_position(idx, out_shape, out_strides)
            out[out_pos] = sum_val

    return njit(parallel=True)(_reduce)

def _tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None:
    for n in prange(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                sum_val = 0.0
                for k in range(a_shape[2]):
                    a_pos = n * a_strides[0] + i * a_strides[1] + k * a_strides[2]
                    b_pos = n * b_strides[0] + k * b_strides[1] + j * b_strides[2]
                    sum_val += a_storage[a_pos] * b_storage[b_pos]
                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                out[out_pos] = sum_val

tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
