/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.*

public fun <T : Number> MultiArray<T, D1>.append(n: MultiArray<T, D1>): NDArray<T, D1> {
    val _totalSize: Int = this.size + n.size
    var data = initMemoryView<T>(_totalSize, this.dtype)
    if (this.consistent && n.consistent) {
        data = when (this.dtype) {
            DataType.IntDataType -> {
                val dest = IntArray(_totalSize)
                System.arraycopy(this.data.getIntArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getIntArray(), 0, dest, this.size, n.size)
                MemoryViewIntArray(dest)
            }
            DataType.ByteDataType -> {
                val dest = ByteArray(_totalSize)
                System.arraycopy(this.data.getByteArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getByteArray(), 0, dest, this.size, n.size)
                MemoryViewByteArray(dest)
            }
            DataType.ShortDataType -> {
                val dest = ShortArray(_totalSize)
                System.arraycopy(this.data.getShortArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getShortArray(), 0, dest, this.size, n.size)
                MemoryViewShortArray(dest)
            }
            DataType.LongDataType -> {
                val dest = LongArray(_totalSize)
                System.arraycopy(this.data.getLongArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getLongArray(), 0, dest, this.size, n.size)
                MemoryViewLongArray(dest)
            }
            DataType.FloatDataType -> {
                val dest = FloatArray(_totalSize)
                System.arraycopy(this.data.getFloatArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getFloatArray(), 0, dest, this.size, n.size)
                MemoryViewFloatArray(dest)
            }
            DataType.DoubleDataType -> {
                val dest = DoubleArray(_totalSize)
                System.arraycopy(this.data.getDoubleArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getDoubleArray(), 0, dest, this.size, n.size)
                MemoryViewDoubleArray(dest)
            }
            DataType.ComplexFloatDataType -> {
                val dest = ComplexFloatArray(_totalSize)
                System.arraycopy(this.data.getComplexFloatArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getComplexFloatArray(), 0, dest, this.size, n.size)
                MemoryViewComplexFloatArray(dest)
            }
            DataType.ComplexDoubleDataType -> {
                val dest = ComplexDoubleArray(_totalSize)
                System.arraycopy(this.data.getComplexDoubleArray(), 0, dest, 0, this.size)
                System.arraycopy(n.data.getComplexDoubleArray(), 0, dest, this.size, n.size)
                MemoryViewComplexDoubleArray(dest)
            }
        } as MemoryView<T>
    } else {
        var index = 0
        for (el in this) {
            data[index++] = el
        }
        for (el in n) {
            data[index++] = el
        }
    }
    return NDArray(data = data, dim = D1, shape = intArrayOf(_totalSize), dtype = this.dtype)
}

public fun <T : Number> MultiArray<T, D1>.append(n: T): NDArray<T, D1> {
    val _totalSize: Int = this.size + 1

    var data = initMemoryView<T>(_totalSize, this.dtype)

    if (this.consistent) {
        data = when (this.dtype) {
            DataType.ByteDataType -> {
                val dest = ByteArray(_totalSize)
                System.arraycopy(this.data.getByteArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Byte
                MemoryViewByteArray(dest)
            }
            DataType.ShortDataType -> {
                val dest = ShortArray(_totalSize)
                System.arraycopy(this.data.getShortArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Short
                MemoryViewShortArray(dest)
            }
            DataType.IntDataType -> {
                val dest = IntArray(_totalSize)
                System.arraycopy(this.data.getIntArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Int
                MemoryViewIntArray(dest)
            }
            DataType.LongDataType -> {
                val dest = LongArray(_totalSize)
                System.arraycopy(this.data.getLongArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Long
                MemoryViewLongArray(dest)
            }
            DataType.FloatDataType -> {
                val dest = FloatArray(_totalSize)
                System.arraycopy(this.data.getFloatArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Float
                MemoryViewFloatArray(dest)
            }
            DataType.DoubleDataType -> {
                val dest = DoubleArray(_totalSize)
                System.arraycopy(this.data.getDoubleArray(), 0, dest, 0, this.size)
                dest[this.size] = n as Double
                MemoryViewDoubleArray(dest)
            }
            DataType.ComplexFloatDataType -> {
                val dest = ComplexFloatArray(_totalSize)
                System.arraycopy(this.data.getComplexFloatArray(), 0, dest, 0, this.size)
                dest[this.size] = n as ComplexFloat
                MemoryViewComplexFloatArray(dest)
            }
            DataType.ComplexDoubleDataType -> {
                val dest = ComplexDoubleArray(_totalSize)
                System.arraycopy(this.data.getComplexDoubleArray(), 0, dest, 0, this.size)
                dest[this.size] = n as ComplexDouble
                MemoryViewComplexDoubleArray(dest)
            }
        } as MemoryView<T>
    } else {
        var index = 0
        for (el in this) {
            data[index++] = el
        }
        data[index] = n
    }
    return NDArray(data = data, dim = D1, shape = intArrayOf(_totalSize), dtype = this.dtype)
}
