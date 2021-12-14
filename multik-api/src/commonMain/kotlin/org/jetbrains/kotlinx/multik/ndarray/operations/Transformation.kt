/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.complex.copyInto
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(vararg value: T): D1Array<T> {
    val newSize = this.size + value.size
    val data = this.copyFromTwoArrays(value.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 *
 */
public infix fun <T, D : Dimension, ID : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, ID>): D1Array<T> {
    val newSize = this.size + arr.size
    val data = this.copyFromTwoArrays(arr.iterator(), newSize)
    return D1Array(data, shape = intArrayOf(newSize), dim = D1)
}

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(arr: MultiArray<T, D>, axis: Int): NDArray<T, D> =
    this.cat(arr, axis)

@JvmName("stackD1")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D1>, axis: Int = 0): NDArray<T, D2> =
    stack(arr.toList(), axis)

@JvmName("stackD1")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D1>>, axis: Int = 0): NDArray<T, D2> =
    stackArrays(arrays, axis)

@JvmName("stackD2")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D2>, axis: Int = 0): NDArray<T, D3> =
    stack(arr.toList(), axis)

@JvmName("stackD2")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D2>>, axis: Int = 0): NDArray<T, D3> =
    stackArrays(arrays, axis)

@JvmName("stackD3")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D3>, axis: Int = 0): NDArray<T, D4> =
    stack(arr.toList(), axis)

@JvmName("stackD3")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D3>>, axis: Int = 0): NDArray<T, D4> =
    stackArrays(arrays, axis)

@JvmName("stackD4")
public fun <T> Multik.stack(vararg arr: MultiArray<T, D4>, axis: Int = 0): NDArray<T, DN> =
    stack(arr.toList(), axis)

@JvmName("stackD4")
public fun <T> Multik.stack(arrays: List<MultiArray<T, D4>>, axis: Int = 0): NDArray<T, DN> =
    stackArrays(arrays, axis)

private fun <T, ID: Dimension, OD: Dimension> stackArrays(arrays: List<MultiArray<T, ID>>, axis: Int = 0): NDArray<T, OD> {
    require(arrays.isNotEmpty() && arrays.size > 1) { "Arrays list is empty or contains one element." }
    val firstArray = arrays.first()
    val actualAxis = firstArray.actualAxis(axis)
    val dim = dimensionOf<OD>(firstArray.dim.d + 1)
    require(axis in 0..firstArray.dim.d) { "Axis $axis is out of bounds for array of dimension $dim" }
    val arrShape = firstArray.shape
    require(arrays.all { it.shape.contentEquals(arrShape) }) { "Arrays must be of the same shape." }

    val shape = arrShape.toMutableList().apply { add(actualAxis, arrays.size) }.toIntArray()
    val size = shape.fold(1, Int::times)
    val result = NDArray(initMemoryView<T>(size, arrays.first().dtype), shape = shape, dim = dim)
    concatenate(arrays, result, axis = axis)
    return result
}

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.repeat(n: Int): D1Array<T> {
    require(n >= 1) { "The number of repetitions must be more than one." }
    val data = initMemoryView<T>(size * n, dtype)
    if (consistent) {
        when (this.dtype) {
            DataType.ComplexFloatDataType -> this.data.getComplexFloatArray().copyInto(data.getComplexFloatArray())
            DataType.ComplexDoubleDataType -> this.data.getComplexDoubleArray().copyInto(data.getComplexDoubleArray())
            else -> (this.data.data as Array<*>).copyInto(data.data as Array<Any?>, 0, 0, this.size)
            //else -> System.arraycopy(this.data.data, 0, data.data, 0, this.size)
        }
    } else {
        var index = 0
        for (el in this)
            data[index++] = el
    }

    for (i in size until (size * n) step size) {
        val startIndex = i - size
        val endIndex = i
        val startIndexComplex = startIndex * 2
        val endIndexComplex = startIndexComplex + (size * 2)
        when (this.dtype) {
            DataType.ComplexFloatDataType ->
                data.getFloatArray().copyInto(data.getFloatArray(), i * 2, startIndexComplex, endIndexComplex)
                //System.arraycopy(data.getFloatArray(), (i - size) * 2, data.getFloatArray(), i * 2, size * 2)
            DataType.ComplexDoubleDataType ->
                data.getDoubleArray().copyInto(data.getDoubleArray(), i * 2, startIndexComplex, endIndexComplex)
                //System.arraycopy(data.getDoubleArray(), (i - size) * 2, data.getDoubleArray(), i * 2, size * 2)
            else -> (data.data as Array<*>).copyInto(data.data as Array<Any?>, i, i - size, endIndex)
            //else -> System.arraycopy(data.data, i - size, data.data, i, size)
        }
    }
    return D1Array(data, shape = intArrayOf(size * n), dim = D1)
}


internal fun <T, D : Dimension> MultiArray<T, D>.copyFromTwoArrays(iter: Iterator<T>, size: Int): MemoryView<T> {
    val data = initMemoryView<T>(size, this.dtype)
    if (this.consistent) {
        when (this.dtype) {
            DataType.ComplexFloatDataType -> this.data.getComplexFloatArray().copyInto(data.getComplexFloatArray())
            DataType.ComplexDoubleDataType -> this.data.getComplexDoubleArray().copyInto(data.getComplexDoubleArray())
            else -> (this.data.data as Array<*>).copyInto(data.data as Array<Any?>, 0, 0, this.size)
            //else -> System.arraycopy(this.data.data, 0, data.data, 0, this.size)
        }
    } else {
        var index = 0
        for (el in this)
            data[index++] = el
    }

    var index = this.size
    for (el in iter)
        data[index++] = el

    return data
}

internal fun <T, D : Dimension, O : Dimension> concatenate(
    arrays: List<MultiArray<T, D>>,
    dest: NDArray<T, O>,
    indices: MultiIndexProgression = dest.multiIndices,
    axis: Int = 0
): NDArray<T, O> {
    if (axis == 0) {
        var offset = 0
        arrays.forEachIndexed { i: Int, arr: MultiArray<T, D> ->
            if (arr.consistent) {
                when (dest.dtype) {
                    DataType.ComplexFloatDataType -> arr.data.getComplexFloatArray()
                        .copyInto(dest.data.getComplexFloatArray(), offset)
                    DataType.ComplexDoubleDataType -> arr.data.getComplexDoubleArray()
                        .copyInto(dest.data.getComplexDoubleArray(), offset)
                    else -> (arr.data.data as Array<*>).copyInto(dest.data.data as Array<Any?>, offset, 0, arr.size)
                    //else -> System.arraycopy(arr.data.data, 0, dest.data.data, offset, arr.size)
                }
            } else {
                var index = offset
                for (el in arr)
                    dest.data[index++] = el
            }
            offset += arr.size
        }
    } else {
        when (arrays.first().dim) {
            D1 -> {
                (arrays as List<MultiArray<T, D1>>).apply {
                    copy(dest, indices) { i, k -> this[k][i] }
                }
            }
            D2 -> {
                (arrays as List<MultiArray<T, D2>>).apply {
                    when (axis) {
                        1 -> copy(dest, indices) { i, k, j -> this[k][i, j] }
                        2 -> copy(dest, indices) { i, j, k -> this[k][i, j] }
                    }
                }
            }
            D3 -> {
                (arrays as List<MultiArray<T, D3>>).apply {
                    when (axis) {
                        1 -> copy(dest, indices) { i, k, j, l -> this[k][i, j, l] }
                        2 -> copy(dest, indices) { i, j, k, l -> this[k][i, j, l] }
                        3 -> copy(dest, indices) { i, j, l, k -> this[k][i, j, l] }
                    }
                }
            }
            D4 -> {
                (arrays as List<MultiArray<T, D4>>).apply {
                    when (axis) {
                        1 -> copy(dest, indices) { i, k, j, l, h -> this[k][i, j, l, h] }
                        2 -> copy(dest, indices) { i, j, k, l, h -> this[k][i, j, l, h] }
                        3 -> copy(dest, indices) { i, j, l, k, h -> this[k][i, j, l, h] }
                        4 -> copy(dest, indices) { i, j, l, h, k -> this[k][i, j, l, h] }
                    }
                }
            }
            else -> throw UnsupportedOperationException()
        }
    }
    return dest
}

private inline fun <T, D : Dimension> copy(
    destination: NDArray<T, D>, indices: MultiIndexProgression, copy: (Int, Int) -> T
) {
    var index = 0
    for (i in indices) {
        destination.data[index++] = copy(i[0], i[1])
    }
}

private inline fun <T, D : Dimension> copy(
    destination: NDArray<T, D>, indices: MultiIndexProgression, copy: (Int, Int, Int) -> T
) {
    var index = 0
    for (i in indices) {
        destination.data[index++] = copy(i[0], i[1], i[2])
    }
}

private inline fun <T, D : Dimension> copy(
    destination: NDArray<T, D>, indices: MultiIndexProgression, copy: (Int, Int, Int, Int) -> T
) {
    var index = 0
    for (i in indices) {
        destination.data[index++] = copy(i[0], i[1], i[2], i[3])
    }
}

private inline fun <T, D : Dimension> copy(
    destination: NDArray<T, D>, indices: MultiIndexProgression, copy: (Int, Int, Int, Int, Int) -> T
) {
    var index = 0
    for (i in indices) {
        destination.data[index++] = copy(i[0], i[1], i[2], i[3], i[4])
    }
}