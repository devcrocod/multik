/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import kotlin.jvm.JvmName
import kotlin.math.ceil


/**
 * Returns a new zero array with the specified shape.
 */
@Deprecated("Use zeros instead.", ReplaceWith("mk.zeros(dims)"), level = DeprecationLevel.ERROR)
public inline fun <reified T : Any, reified D : Dimension> Multik.empty(vararg dims: Int): NDArray<T, D> {
    val dim = dimensionClassOf<D>(dims.size)
    requireDimension(dim, dims.size)
    val dtype = DataType.ofKClass(T::class)
    val size = dims.reduce { acc, el -> acc * el }
    val data = initMemoryView<T>(size, dtype)
    return NDArray(data, shape = dims, dim = dim)
}

/**
 * Returns a new zero array of type [dtype] with the specified shape.
 *
 * Note: Generic type of elements [T] must match [dtype].
 */
@Deprecated("Use zeros instead.", ReplaceWith("mk.zeros(dims)"), level = DeprecationLevel.ERROR)
public fun <T, D : Dimension> Multik.empty(dims: IntArray, dtype: DataType): NDArray<T, D> {
    val dim = dimensionOf<D>(dims.size)
    requireDimension(dim, dims.size)
    val size = dims.fold(1, Int::times)
    val data = initMemoryView<T>(size, dtype)
    return NDArray(data, shape = dims, dim = dim)
}

/**
 * Returns a new zero array of 1 dim of type [T] with [dim1] size.
 *
 * @sample samples.NDArrayTest.zerosD1
 */
public inline fun <reified T : Any> Multik.zeros(dim1: Int): D1Array<T> =
    zeros(intArrayOf(dim1), DataType.ofKClass(T::class))

/**
 * Returns a new zero array of 2 dim of type [T] with ([dim1], [dim2]) shape.
 *
 * @sample samples.NDArrayTest.zerosD2
 */
public inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int): D2Array<T> =
    zeros(intArrayOf(dim1, dim2), DataType.ofKClass(T::class))

/**
 * Returns a new zero array of 3 dim of type [T] with ([dim1], [dim2], [dim3]) shape.
 *
 * @sample samples.NDArrayTest.zerosD3
 */
public inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int, dim3: Int): D3Array<T> =
    zeros(intArrayOf(dim1, dim2, dim3), DataType.ofKClass(T::class))

/**
 * Returns a new zero array of 4 dim of type [T] with ([dim1], [dim2], [dim3], [dim4]) shape.
 *
 * @sample samples.NDArrayTest.zerosD4
 */
public inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T> =
    zeros(intArrayOf(dim1, dim2, dim3, dim4), DataType.ofKClass(T::class))

/**
 * Returns a new zero array of N dim of type [T] with ([dim1], [dim2], [dim3], [dim4], [dims]) shape.
 *
 * @sample samples.NDArrayTest.zerosDN
 */
public inline fun <reified T : Any> Multik.zeros(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN> =
    zeros(intArrayOf(dim1, dim2, dim3, dim4) + dims, DataType.ofKClass(T::class))

/**
 * Returns a new zero array of type [T] with specify shape.
 *
 * @sample samples.NDArrayTest.zerosDNWithDtype
 */
public fun <T, D : Dimension> Multik.zeros(dims: IntArray, dtype: DataType): NDArray<T, D> {
    val dim = dimensionOf<D>(dims.size)
    requireDimension(dim, dims.size)
    val size = dims.fold(1, Int::times)
    val data = initMemoryView<T>(size, dtype)
    return NDArray(data, shape = dims, dim = dim)
}

/**
 * Returns a new ones array of 1 dim of type [T] with [dim1] size.
 *
 * @sample samples.NDArrayTest.onesD1
 */
public inline fun <reified T : Any> Multik.ones(dim1: Int): D1Array<T> =
    ones(intArrayOf(dim1), DataType.ofKClass(T::class))

/**
 * Returns a new ones array of 2 dim of type [T] with ([dim1], [dim2]) shape.
 *
 * @sample samples.NDArrayTest.onesD2
 */
public inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int): D2Array<T> =
    ones(intArrayOf(dim1, dim2), DataType.ofKClass(T::class))

/**
 * Returns a new ones array of 3 dim of type [T] with ([dim1], [dim2], [dim3]) shape.
 *
 * @sample samples.NDArrayTest.onesD3
 */
public inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int, dim3: Int): D3Array<T> =
    ones(intArrayOf(dim1, dim2, dim3), DataType.ofKClass(T::class))

/**
 * Returns a new ones array of 4 dim of type [T] with ([dim1], [dim2], [dim3], [dim4]) shape.
 *
 * @sample samples.NDArrayTest.onesD4
 */
public inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T> =
    ones(intArrayOf(dim1, dim2, dim3, dim4), DataType.ofKClass(T::class))

/**
 * Returns a new ones array of N dim of type [T] with ([dim1], [dim2], [dim3], [dim4], [dims]) shape.
 *
 * @sample samples.NDArrayTest.onesDN
 */
public inline fun <reified T : Any> Multik.ones(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN> =
    ones(intArrayOf(dim1, dim2, dim3, dim4) + dims, DataType.ofKClass(T::class))

/**
 * Returns a new ones array of N dim of type [T] with specify shape.
 *
 * @sample samples.NDArrayTest.onesDNWithDtype
 */
@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
public fun <T, D : Dimension> Multik.ones(dims: IntArray, dtype: DataType): NDArray<T, D> {
    val dim = dimensionOf<D>(dims.size)
    requireDimension(dim, dims.size)
    val size = dims.fold(1, Int::times)
    val one: T = when (dtype) {
        DataType.ByteDataType -> 1.toByte()
        DataType.ShortDataType -> 1.toShort()
        DataType.IntDataType -> 1
        DataType.LongDataType -> 1L
        DataType.FloatDataType -> 1f
        DataType.DoubleDataType -> 1.0
        DataType.ComplexFloatDataType -> ComplexFloat.one
        DataType.ComplexDoubleDataType -> ComplexDouble.one
    } as T
    val data = initMemoryView(size, dtype) { one }
    return NDArray(data, shape = dims, dim = dim)
}

/**
 * Returns the identity array.
 *
 * @param n number of rows and columns.
 * @return [D2Array].
 * @sample samples.NDArrayTest.identity
 */
public inline fun <reified T : Any> Multik.identity(n: Int): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    return identity(n, dtype)
}

/**
 * Returns the identity array.
 *
 * Note: Generic type of elements must match [dtype].
 *
 * @param n number of rows and columns.
 * @param dtype array type.
 * @return [D2Array]
 * @sample samples.NDArrayTest.identityWithDtype
 */
public fun <T> Multik.identity(n: Int, dtype: DataType): D2Array<T> {
    val shape = intArrayOf(n, n)
    val ret = D2Array(initMemoryView<T>(n * n, dtype), shape = shape, dim = D2)
    val one: Any = when (dtype) {
        DataType.ByteDataType -> 1.toByte()
        DataType.ShortDataType -> 1.toShort()
        DataType.IntDataType -> 1
        DataType.LongDataType -> 1L
        DataType.FloatDataType -> 1f
        DataType.DoubleDataType -> 1.0
        DataType.ComplexFloatDataType -> ComplexFloat.one
        DataType.ComplexDoubleDataType -> ComplexDouble.one
    }
    for (i in 0 until n) {
        @Suppress("UNCHECKED_CAST")
        ret[i, i] = one as T
    }
    return ret
}

/**
 * Returns a diagonal array.
 *
 * @param elements the elements along the main diagonal.
 * @return [D2Array].
 * @sample samples.NDArrayTest.diagonal
 */
public inline fun <reified T : Any> Multik.diagonal(elements: List<T>): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    return diagonal(elements = elements, dtype = dtype)
}

/**
 * Returns a diagonal array.
 *
 * Note: Generic type of elements must match [dtype].
 *
 * @param dtype array type.
 * @param elements the elements on the main diagonal.
 * @return [D2Array]
 * @sample samples.NDArrayTest.diagonalWithDtype
 */
public fun <T> Multik.diagonal(elements: List<T>, dtype: DataType): D2Array<T> {
    val n = elements.size
    val shape = intArrayOf(n, n)
    val ret = D2Array(initMemoryView<T>(n * n, dtype), shape = shape, dim = D2)
    for (i in 0 until n) {
        ret[i, i] = elements[i]
    }
    return ret
}

/**
 * Creates the 1-dimension array from [arg] of Number type.
 *
 * Example:
 * ```
 * mk.ndarray(mk[1, 2, 3]
 * ```
 * @param arg list of elements.
 * @return [D1Array].
 * @sample samples.NDArrayTest.ndarray1D
 */
@JvmName("ndarray1D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<T>): D1Array<T> = ndarrayCommon1D(arg)

/**
 * Creates the 1-dimension array from [arg] of Complex type.
 *
 * Example:
 * ```
 * mk.ndarray(mk[1, 2, 3]
 * ```
 * @param arg list of elements.
 * @return [D1Array].
 * @sample samples.NDArrayTest.ndarray1D
 */
@JvmName("ndarrayComplex1D")
public inline fun <reified T : Complex> Multik.ndarray(arg: List<T>): D1Array<T> = ndarrayCommon1D(arg)

@PublishedApi
internal inline fun <reified T : Any> ndarrayCommon1D(arg: List<T>): D1Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val data = arg.toViewPrimitiveArray(dtype)
    return D1Array(data, 0, intArrayOf(arg.size), dim = D1)
}

/**
 * Creates the 2-dimension array from [arg] of Number type.
 *
 * @param arg list of rows.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarray2D
 */
@JvmName("ndarray2D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<T>>): D2Array<T> = ndarrayCommon2D(arg)

/**
 * Creates the 2-dimension array from [arg] of Complex type.
 *
 * @param arg list of rows.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarray2D
 */
@JvmName("ndarrayComplex2D")
public inline fun <reified T : Complex> Multik.ndarray(arg: List<List<T>>): D2Array<T> = ndarrayCommon2D(arg)

@PublishedApi
internal inline fun <reified T : Any> ndarrayCommon2D(arg: List<List<T>>): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val size = IntArray(2).apply {
        this[0] = arg.size
        this[1] = arg.first().size
    }
    val res = ArrayList<T>()
    for (ax0 in arg) {
        require(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        res.addAll(ax0)
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D2Array(data, 0, size, dim = D2)
}

/**
 * Creates the 3-dimension array from [arg] of Number type.
 *
 * @param arg elements.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarray3D
 */
@JvmName("ndarray3D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<List<T>>>): D3Array<T> = ndarrayCommon3D(arg)

/**
 * Creates the 3-dimension array from [arg] of Complex type.
 *
 * @param arg elements.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarray3D
 */
@JvmName("ndarrayComplex3D")
public inline fun <reified T : Complex> Multik.ndarray(arg: List<List<List<T>>>): D3Array<T> = ndarrayCommon3D(arg)

@PublishedApi
internal inline fun <reified T : Any> ndarrayCommon3D(arg: List<List<List<T>>>): D3Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val size = IntArray(3).apply {
        this[0] = arg.size
        this[1] = arg.first().size
        this[2] = arg.first().first().size
    }
    val res = ArrayList<T>()
    for (ax0 in arg) {
        require(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        for (ax1 in ax0) {
            require(size[2] == ax1.size) { "The size of the incoming array $ax1 does not match the rest" }
            res.addAll(ax1)
        }
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D3Array(data, 0, size, dim = D3)
}

/**
 * Creates the 4-dimension array from [arg] of Number type.
 *
 * @param arg elements.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarray4D
 */
@JvmName("ndarray4D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<List<List<T>>>>): D4Array<T> = ndarrayCommon4D(arg)

/**
 * Creates the 4-dimension array from [arg] of Complex type.
 *
 * @param arg elements.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarray4D
 */
@JvmName("ndarrayComplex4D")
public inline fun <reified T : Complex> Multik.ndarray(arg: List<List<List<List<T>>>>): D4Array<T> =
    ndarrayCommon4D(arg)

@PublishedApi
internal inline fun <reified T : Any> ndarrayCommon4D(arg: List<List<List<List<T>>>>): D4Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val size = IntArray(4).apply {
        this[0] = arg.size
        this[1] = arg.first().size
        this[2] = arg.first().first().size
        this[3] = arg.first().first().first().size
    }
    val res = ArrayList<T>()
    for (ax0 in arg) {
        check(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        for (ax1 in ax0) {
            require(size[2] == ax1.size) { "The size of the incoming array $ax1 does not match the rest" }
            for (ax2 in ax1) {
                require(size[3] == ax2.size) { "The size of the incoming array $ax2 does not match the rest" }
                res.addAll(ax2)
            }
        }
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D4Array(data, 0, size, dim = D4)
}


/**
 * Returns a new array given [shape] from collection.
 *
 * @param elements collection of elements.
 * @param shape array shape.
 * @return [NDArray] of [D] dimension.
 * @sample samples.NDArrayTest.ndarrayCollections
 */
public inline fun <T : Number, reified D : Dimension> Multik.ndarray(
    elements: Collection<T>, shape: IntArray
): NDArray<T, D> =
    ndarrayCommon(elements, shape, dimensionClassOf(shape.size))

/**
 * Returns a new array given [shape] from collection.
 *
 * @param elements collection of elements.
 * @param shape array shape.
 * @return [NDArray] of [D] dimension.
 * @sample samples.NDArrayTest.ndarrayCollections
 */
@JvmName("ndarrayComplex")
public inline fun <T : Complex, reified D : Dimension> Multik.ndarray(
    elements: Collection<T>, shape: IntArray
): NDArray<T, D> =
    ndarrayCommon(elements, shape, dimensionClassOf(shape.size))

/**
 * Returns a new array given shape and dimension from collection.
 *
 * Note: Generic type of dimension [D] must match dim.
 *
 * @param elements collection of elements.
 * @param shape array shape.
 * @param dim array dimension.
 * @return [NDArray] of [D] dimension.
 * @sample samples.NDArrayTest.ndarrayCollectionsWithDim
 */
public fun <T : Number, D : Dimension> Multik.ndarray(elements: Collection<T>, shape: IntArray, dim: D): NDArray<T, D> =
    ndarrayCommon(elements, shape, dim)

/**
 * Returns a new array given shape and dimension from collection.
 *
 * Note: Generic type of dimension [D] must match dim.
 *
 * @param elements collection of elements.
 * @param shape array shape.
 * @param dim array dimension.
 * @return [NDArray] of [D] dimension.
 * @sample samples.NDArrayTest.ndarrayCollectionsWithDim
 */
@JvmName("ndarrayComplex")
public fun <T : Complex, D : Dimension> Multik.ndarray(
    elements: Collection<T>,
    shape: IntArray,
    dim: D
): NDArray<T, D> =
    ndarrayCommon(elements, shape, dim)

@PublishedApi
internal fun <T, D : Dimension> ndarrayCommon(elements: Collection<T>, shape: IntArray, dim: D, dtype : DataType? = null): NDArray<T, D> {
    requireShapeEmpty(shape)
    requireDimension(dim, shape.size)
    requireElementsWithShape(elements.size, shape.fold(1, Int::times))
    val size = shape.reduce { acc, el -> acc * el }

    val dtypeFromFirst = DataType.of(elements.first())
    val data = initMemoryView<T>(size, dtype ?: dtypeFromFirst).apply {// TODO: boxing/unboxing!!! Find all usages
        var count = 0
        for (el in elements)
            this[count++] = el
    }
    return NDArray(data, shape = shape, dim = dim)
}

//_________________________________________________D1___________________________________________________________________

/**
 * Returns a new 1-dimension array given shape from a collection.
 *
 * @param elements collection of number elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayCollections1D
 */
public fun <T : Number> Multik.ndarray(elements: Collection<T>): D1Array<T> =
    ndarray(elements, intArrayOf(elements.size), D1)

/**
 * Returns a new 1-dimension array given shape from a collection.
 *
 * @param elements collection of complex elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayCollections1D
 */
@JvmName("ndarrayComplex")
public fun <T : Complex> Multik.ndarray(elements: Collection<T>): D1Array<T> =
    ndarray(elements, intArrayOf(elements.size), D1)

/**
 * Returns a new 1-dimension array from [ByteArray].
 *
 * @param args [ByteArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayByteArray1D
 */
public fun Multik.ndarray(args: ByteArray): D1Array<Byte> {
    val data = MemoryViewByteArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [ShortArray].
 *
 * @param args [ShortArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayShortArray1D
 */
public fun Multik.ndarray(args: ShortArray): D1Array<Short> {
    val data = MemoryViewShortArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [IntArray].
 *
 * @param args [IntArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayIntArray1D
 */
public fun Multik.ndarray(args: IntArray): D1Array<Int> {
    val data = MemoryViewIntArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [LongArray].
 *
 * @param args [LongArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayLongArray1D
 */
public fun Multik.ndarray(args: LongArray): D1Array<Long> {
    val data = MemoryViewLongArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [FloatArray].
 *
 * @param args [FloatArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayFloatArray1D
 */
public fun Multik.ndarray(args: FloatArray): D1Array<Float> {
    val data = MemoryViewFloatArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [DoubleArray].
 *
 * @param args [DoubleArray] of elements.
 * @return [D1Array]
 * @sample samples.NDArrayTest.ndarrayDoubleArray1D
 */
public fun Multik.ndarray(args: DoubleArray): D1Array<Double> {
    val data = MemoryViewDoubleArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [ComplexFloatArray].
 *
 * @param args [ComplexFloatArray] of elements.
 * @return [D1Array]
 */
public fun Multik.ndarray(args: ComplexFloatArray): D1Array<ComplexFloat> {
    val data = MemoryViewComplexFloatArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

/**
 * Returns a new 1-dimension array from [ComplexDoubleArray].
 *
 * @param args [ComplexDoubleArray] of elements.
 * @return [D1Array]
 */
public fun Multik.ndarray(args: ComplexDoubleArray): D1Array<ComplexDouble> {
    val data = MemoryViewComplexDoubleArray(args)
    return D1Array(data, shape = intArrayOf(args.size), dim = D1)
}

//_________________________________________________D2___________________________________________________________________

/**
 * Returns a new 2-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayCollections2D
 */
public fun <T : Number> Multik.ndarray(elements: Collection<T>, dim1: Int, dim2: Int): D2Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2), D2)

/**
 * Returns a new 2-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayCollections2D
 */
@JvmName("ndarrayComplex")
public fun <T : Complex> Multik.ndarray(elements: Collection<T>, dim1: Int, dim2: Int): D2Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2), D2)

/**
 * Returns a new 2-dimensions array from [ByteArray].
 *
 * @param args [ByteArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayByteArray2D
 */
public fun Multik.ndarray(args: ByteArray, dim1: Int, dim2: Int): D2Array<Byte> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewByteArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [ShortArray].
 *
 * @param args [ShortArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayShortArray2D
 */
public fun Multik.ndarray(args: ShortArray, dim1: Int, dim2: Int): D2Array<Short> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewShortArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [IntArray].
 *
 * @param args [IntArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayIntArray2D
 */
public fun Multik.ndarray(args: IntArray, dim1: Int, dim2: Int): D2Array<Int> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewIntArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [LongArray].
 *
 * @param args [LongArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayLongArray2D
 */
public fun Multik.ndarray(args: LongArray, dim1: Int, dim2: Int): D2Array<Long> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewLongArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [FloatArray].
 *
 * @param args [FloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayFloatArray2D
 */
public fun Multik.ndarray(args: FloatArray, dim1: Int, dim2: Int): D2Array<Float> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewFloatArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [DoubleArray].
 *
 * @param args [DoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 * @sample samples.NDArrayTest.ndarrayDoubleArray2D
 */
public fun Multik.ndarray(args: DoubleArray, dim1: Int, dim2: Int): D2Array<Double> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewDoubleArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [ComplexFloatArray].
 *
 * @param args [ComplexFloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 */
public fun Multik.ndarray(args: ComplexFloatArray, dim1: Int, dim2: Int): D2Array<ComplexFloat> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewComplexFloatArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

/**
 * Returns a new 2-dimensions array from [ComplexDoubleArray].
 *
 * @param args [ComplexDoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 1-dimension.
 * @return [D2Array].
 */
public fun Multik.ndarray(args: ComplexDoubleArray, dim1: Int, dim2: Int): D2Array<ComplexDouble> {
    requireElementsWithShape(args.size, dim1 * dim2)
    val data = MemoryViewComplexDoubleArray(args)
    return D2Array(data, shape = intArrayOf(dim1, dim2), dim = D2)
}

//_________________________________________________D3___________________________________________________________________

/**
 * Returns a new 3-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayCollections3D
 */
public fun <T : Number> Multik.ndarray(elements: Collection<T>, dim1: Int, dim2: Int, dim3: Int): D3Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2, dim3), D3)

/**
 * Returns a new 3-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayCollections3D
 */
@JvmName("ndarrayComplex")
public fun <T : Complex> Multik.ndarray(elements: Collection<T>, dim1: Int, dim2: Int, dim3: Int): D3Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2, dim3), D3)

/**
 * Returns a new 3-dimensions array from [ByteArray].
 *
 * @param args [ByteArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayByteArray3D
 */
public fun Multik.ndarray(args: ByteArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Byte> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewByteArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [ShortArray].
 *
 * @param args [ShortArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayShortArray3D
 */
public fun Multik.ndarray(args: ShortArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Short> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewShortArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [IntArray].
 *
 * @param args [IntArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayIntArray3D
 */
public fun Multik.ndarray(args: IntArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Int> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewIntArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [LongArray].
 *
 * @param args [LongArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayLongArray3D
 */
public fun Multik.ndarray(args: LongArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Long> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewLongArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [FloatArray].
 *
 * @param args [FloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayFloatArray3D
 */
public fun Multik.ndarray(args: FloatArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Float> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewFloatArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [DoubleArray].
 *
 * @param args [DoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 * @sample samples.NDArrayTest.ndarrayDoubleArray3D
 */
public fun Multik.ndarray(args: DoubleArray, dim1: Int, dim2: Int, dim3: Int): D3Array<Double> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewDoubleArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [ComplexFloatArray].
 *
 * @param args [ComplexFloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 */
public fun Multik.ndarray(args: ComplexFloatArray, dim1: Int, dim2: Int, dim3: Int): D3Array<ComplexFloat> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewComplexFloatArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

/**
 * Returns a new 3-dimensions array from [ComplexDoubleArray].
 *
 * @param args [ComplexDoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @return [D3Array].
 */
public fun Multik.ndarray(args: ComplexDoubleArray, dim1: Int, dim2: Int, dim3: Int): D3Array<ComplexDouble> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3)
    val data = MemoryViewComplexDoubleArray(args)
    return D3Array(data, shape = intArrayOf(dim1, dim2, dim3), dim = D3)
}

//_________________________________________________D4___________________________________________________________________

/**
 * Returns a new 4-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayCollections4D
 */
public fun <T : Number> Multik.ndarray(
    elements: Collection<T>,
    dim1: Int,
    dim2: Int,
    dim3: Int,
    dim4: Int
): D4Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2, dim3, dim4), D4)

/**
 * Returns a new 4-dimensions array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayCollections4D
 */
@JvmName("ndarrayComplex")
public fun <T : Complex> Multik.ndarray(
    elements: Collection<T>,
    dim1: Int,
    dim2: Int,
    dim3: Int,
    dim4: Int
): D4Array<T> =
    ndarray(elements, intArrayOf(dim1, dim2, dim3, dim4), D4)

/**
 * Returns a new 4-dimensions array from [ByteArray].
 *
 * @param args [ByteArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayByteArray4D
 */
public fun Multik.ndarray(args: ByteArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Byte> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewByteArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [ShortArray].
 *
 * @param args [ShortArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayShortArray4D
 */
public fun Multik.ndarray(args: ShortArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Short> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewShortArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [IntArray].
 *
 * @param args [IntArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayIntArray4D
 */
public fun Multik.ndarray(args: IntArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Int> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewIntArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [LongArray].
 *
 * @param args [LongArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayLongArray4D
 */
public fun Multik.ndarray(args: LongArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Long> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewLongArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [FloatArray].
 *
 * @param args [FloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayFloatArray4D
 */
public fun Multik.ndarray(args: FloatArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Float> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewFloatArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [DoubleArray].
 *
 * @param args [DoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 * @sample samples.NDArrayTest.ndarrayDoubleArray4D
 */
public fun Multik.ndarray(args: DoubleArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<Double> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewDoubleArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [ComplexFloatArray].
 *
 * @param args [ComplexFloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 */
public fun Multik.ndarray(args: ComplexFloatArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<ComplexFloat> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewComplexFloatArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

/**
 * Returns a new 4-dimensions array from [ComplexDoubleArray].
 *
 * @param args [ComplexDoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @return [D4Array].
 */
public fun Multik.ndarray(
    args: ComplexDoubleArray,
    dim1: Int,
    dim2: Int,
    dim3: Int,
    dim4: Int
): D4Array<ComplexDouble> {
    requireElementsWithShape(args.size, dim1 * dim2 * dim3 * dim4)
    val data = MemoryViewComplexDoubleArray(args)
    return D4Array(data, shape = intArrayOf(dim1, dim2, dim3, dim4), dim = D4)
}

//_________________________________________________DN___________________________________________________________________

/**
 * Returns a new n-dimension array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayCollectionsDN
 */
public fun <T : Number> Multik.ndarray(
    elements: Collection<T>, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    return ndarray(elements, shape, DN(shape.size))
}

/**
 * Returns a new n-dimension array given shape from a collection.
 *
 * @param elements collection of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayCollectionsDN
 */
@JvmName("ndarrayComplex")
public fun <T : Complex> Multik.ndarray(
    elements: Collection<T>, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    return ndarray(elements, shape, DN(shape.size))
}

/**
 * Returns a new n-dimension array from [ByteArray].
 *
 * @param args [ByteArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayByteArrayDN
 */
public fun Multik.ndarray(
    args: ByteArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Byte, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewByteArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [ShortArray].
 *
 * @param args [ShortArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayShortArrayDN
 */
public fun Multik.ndarray(
    args: ShortArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Short, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewShortArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [IntArray].
 *
 * @param args [IntArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayIntArrayDN
 */
public fun Multik.ndarray(
    args: IntArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Int, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewIntArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [LongArray].
 *
 * @param args [LongArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayLongArrayDN
 */
public fun Multik.ndarray(
    args: LongArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Long, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewLongArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [FloatArray].
 *
 * @param args [FloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayFloatArrayDN
 */
public fun Multik.ndarray(
    args: FloatArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Float, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewFloatArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [DoubleArray].
 *
 * @param args [DoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 * @sample samples.NDArrayTest.ndarrayDoubleArrayDN
 */
public fun Multik.ndarray(
    args: DoubleArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<Double, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewDoubleArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [ComplexFloatArray].
 *
 * @param args [ComplexFloatArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 */
public fun Multik.ndarray(
    args: ComplexFloatArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<ComplexFloat, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewComplexFloatArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new n-dimension array from [ComplexDoubleArray].
 *
 * @param args [ComplexDoubleArray] of elements.
 * @param dim1 value of 1-dimension.
 * @param dim2 value of 2-dimension.
 * @param dim3 value of 3-dimension.
 * @param dim4 value of 4-dimension.
 * @param dims values of other dimensions.
 * @return [NDArray] of [DN] dimension.
 */
public fun Multik.ndarray(
    args: ComplexDoubleArray, dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<ComplexDouble, DN> {
    val shape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    requireElementsWithShape(args.size, shape.fold(1, Int::times))
    val data = MemoryViewComplexDoubleArray(args)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

//______________________________________________________________________________________________________________________
/**
 * Returns a 1-dimension array.
 *
 * @param sizeD1 value of 1-dimension.
 * @param init initialization function
 * @return [D1Array].
 * @sample samples.NDArrayTest.d1array
 */
public inline fun <reified T : Any> Multik.d1array(sizeD1: Int, noinline init: (Int) -> T): D1Array<T> {
    require(sizeD1 > 0) { "Dimension must be positive." }
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1)
    val data = initMemoryView(sizeD1, dtype, init)
    return D1Array(data, shape = shape, dim = D1)
}

/**
 * Creates a new array of the specified ([sizeD1], [sizeD2]) shape, where each element is calculated by calling
 * the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its index.
 * @sample samples.NDArrayTest.d2array
 */
public inline fun <reified T : Any> Multik.d2array(sizeD1: Int, sizeD2: Int, noinline init: (Int) -> T): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val data = initMemoryView(sizeD1 * sizeD2, dtype, init)
    return D2Array(data, shape = shape, dim = D2)
}

/**
 * Creates a new 2-dimensions array of the specified ([sizeD1], [sizeD2]) shape, where each element is calculated by calling
 * the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its indices.
 * @sample samples.NDArrayTest.d2arrayIndices
 */
public inline fun <reified T : Any> Multik.d2arrayIndices(
    sizeD1: Int, sizeD2: Int, init: (i: Int, j: Int) -> T
): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val ret = D2Array<T>(initMemoryView(sizeD1 * sizeD2, dtype), shape = shape, dim = D2)
    for (i in 0 until sizeD1) {
        for (j in 0 until sizeD2) {
            ret[i, j] = init(i, j)
        }
    }
    return ret
}

/**
 * Creates a new 3-dimensions array of the specified ([sizeD1], [sizeD2], [sizeD3]) shape, where each element is calculated by calling
 * the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its index.
 * @sample samples.NDArrayTest.d3array
 */
public inline fun <reified T : Any> Multik.d3array(
    sizeD1: Int, sizeD2: Int, sizeD3: Int, noinline init: (Int) -> T
): D3Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val data = initMemoryView(sizeD1 * sizeD2 * sizeD3, dtype, init)
    return D3Array(data, shape = shape, dim = D3)
}

/**
 * Creates a new 3-dimensions array of the specified ([sizeD1], [sizeD2], [sizeD3]) shape, where each element is calculated by calling
 * the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its indices.
 * @sample samples.NDArrayTest.d3arrayIndices
 */
public inline fun <reified T : Any> Multik.d3arrayIndices(
    sizeD1: Int, sizeD2: Int, sizeD3: Int, init: (i: Int, j: Int, k: Int) -> T
): D3Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }

    val ret = D3Array<T>(initMemoryView(sizeD1 * sizeD2 * sizeD3, dtype), shape = shape, dim = D3)
    for (i in 0 until sizeD1) {
        for (j in 0 until sizeD2) {
            for (k in 0 until sizeD3) {
                ret[i, j, k] = init(i, j, k)
            }
        }
    }

    return ret
}

/**
 * Creates a new 4-dimensions array of the specified ([sizeD1], [sizeD2], [sizeD3], [sizeD4]) shape,
 * where each element is calculated by calling the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its index.
 * @sample samples.NDArrayTest.d4array
 */
public inline fun <reified T : Any> Multik.d4array(
    sizeD1: Int, sizeD2: Int, sizeD3: Int, sizeD4: Int, noinline init: (Int) -> T
): D4Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3, sizeD4)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val data = initMemoryView(sizeD1 * sizeD2 * sizeD3 * sizeD4, dtype, init)
    return D4Array(data, shape = shape, dim = D4)
}

/**
 * Creates a new 4-dimensions array of the specified ([sizeD1], [sizeD2], [sizeD3], [sizeD4]) shape,
 * where each element is calculated by calling the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its indices.
 * @sample samples.NDArrayTest.d4arrayIndices
 */
public inline fun <reified T : Any> Multik.d4arrayIndices(
    sizeD1: Int, sizeD2: Int, sizeD3: Int, sizeD4: Int, init: (i: Int, j: Int, k: Int, m: Int) -> T
): D4Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3, sizeD4)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }

    val ret = D4Array<T>(initMemoryView(sizeD1 * sizeD2 * sizeD3 * sizeD4, dtype), shape = shape, dim = D4)
    for (i in 0 until sizeD1) {
        for (j in 0 until sizeD2) {
            for (k in 0 until sizeD3) {
                for (m in 0 until sizeD4) {
                    ret[i, j, k, m] = init(i, j, k, m)
                }
            }
        }
    }

    return ret
}

/**
 * Creates a new 4-dimensions array of the specified ([sizeD1], [sizeD2], [sizeD3], [sizeD4], [dims]) shape,
 * where each element is calculated by calling the specified [init] function.
 *
 * The function [init] is called for each array element sequentially starting from the first one.
 * It should return the value for an array element given its index.
 * @sample samples.NDArrayTest.dnarray
 */
public inline fun <reified T : Any> Multik.dnarray(
    sizeD1: Int, sizeD2: Int, sizeD3: Int, sizeD4: Int, vararg dims: Int, noinline init: (Int) -> T
): NDArray<T, DN> = dnarray(intArrayOf(sizeD1, sizeD2, sizeD3, sizeD4, *dims), init)

/**
 * Returns a new array with the specified [shape], where each element is calculated by calling the specified
 * [init] function.
 *
 * @sample samples.NDArrayTest.dnarrayWithDims
 */
public inline fun <reified T : Any, reified D : Dimension> Multik.dnarray(
    shape: IntArray, noinline init: (Int) -> T
): NDArray<T, D> {
    val dtype = DataType.ofKClass(T::class)
    val dim = dimensionClassOf<D>(shape.size)
    requireDimension(dim, shape.size)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val size = shape.fold(1, Int::times)
    val data = initMemoryView(size, dtype, init)
    return NDArray(data, shape = shape, dim = dimensionOf(shape.size))
}

/**
 * Returns a new 1-dimension array from [items].
 */
public fun Multik.ndarrayOf(vararg items: Byte): D1Array<Byte> = ndarrayOfCommon(items.toTypedArray(), DataType.ByteDataType)
/**
 * Returns a new 1-dimension array from [items].
 */
public fun Multik.ndarrayOf(vararg items: Short): D1Array<Short> = ndarrayOfCommon(items.toTypedArray(), DataType.ShortDataType)
/**
 * Returns a new 1-dimension array from [items].
 * @sample samples.NDArrayTest.ndarrayOf
 */
public fun Multik.ndarrayOf(vararg items: Int): D1Array<Int> = ndarrayOfCommon(items.toTypedArray(), DataType.IntDataType)
/**
 * Returns a new 1-dimension array from [items].
 */
public fun Multik.ndarrayOf(vararg items: Long): D1Array<Long> = ndarrayOfCommon(items.toTypedArray(), DataType.LongDataType)
/**
 * Returns a new 1-dimension array from [items].
 */
public fun Multik.ndarrayOf(vararg items: Float): D1Array<Float> = ndarrayOfCommon(items.toTypedArray(), DataType.FloatDataType)
/**
 * Returns a new 1-dimension array from [items].
 */
public fun Multik.ndarrayOf(vararg items: Double): D1Array<Double> = ndarrayOfCommon(items.toTypedArray(), DataType.DoubleDataType)

/**
 * Returns a one-dimensional array of generic complex type T consisting of values specified in input items.
 *
 * @param items Variable number of arguments of generic complex type T.
 * @return One-dimensional array of generic complex type T.
 */
public inline fun <reified T: Complex> Multik.ndarrayOf(vararg items: T): D1Array<T> = ndarrayOfCommon(items, DataType.ofKClass(T::class))

// TODO(https://youtrack.jetbrains.com/issue/KT-33565/Allow-vararg-parameter-of-inline-class-type)
///**
// * Returns a new 1-dimension array from [items].
// */
//public fun Multik.ndarrayOf(vararg items: ComplexFloat): D1Array<ComplexFloat> = ndarrayOfCommon(items, DataType.ComplexFloatDataType)
///**
// * Returns a new 1-dimension array from [items].
// */
//public fun Multik.ndarrayOf(vararg items: ComplexDouble): D1Array<ComplexDouble> = ndarrayOfCommon(items, DataType.ComplexDoubleDataType)


@PublishedApi
internal fun <T> ndarrayOfCommon(items: Array<out T>, dtype: DataType): D1Array<T> {
    val shape = intArrayOf(items.size)
    val data = initMemoryView(items.size, dtype) { items[it] }
    return D1Array(data, shape = shape, dim = D1)
}

/**
 * Returns evenly spaced values within a given interval, where [step] is Integer.
 *
 * @param start starting value of the interval. The interval includes this value.
 * @param stop end value of the interval. The interval doesn't include this value.
 * @param step spacing between value. The default step is 1.
 * @return [D1Array].
 * @sample samples.NDArrayTest.arange
 */
public inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Int = 1): D1Array<T> {
    return arange(start, stop, step.toDouble())
}

/**
 * Returns evenly spaced values within a given interval, where [step] is Double.
 *
 * @param start starting value of the interval. The interval includes this value.
 * @param stop end value of the interval. The interval doesn't include this value.
 * @param step spacing between value. The step is [Double].
 * @return [D1Array].
 * @sample samples.NDArrayTest.arangeDoubleStep
 */
public inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Double): D1Array<T> {
    if (start < stop) require(step > 0) { "Step must be positive." }
    else if (start > stop) require(step < 0) { "Step must be negative." }

    val size = ceil((stop.toDouble() - start) / step).toInt()
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(size)
    val data = initMemoryView<T>(size, dtype).apply {
        var tmp = start.toDouble()
        for (i in indices) {
            this[i] = tmp.toPrimitiveType()
            tmp += step
        }
    }
    return D1Array(data, shape = shape, dim = D1)
}

/**
 * Returns evenly spaced values within a given interval, where start of interval is 0.
 * see [Multik.arange]
 *
 * @param stop end of the interval. The interval doesn't include this value.
 * @param step spacing between value. The default step is 1.
 * @return [D1Array].
 * @sample samples.NDArrayTest.arangeWithoutStart
 */
public inline fun <reified T : Number> Multik.arange(stop: Int, step: Int = 1): D1Array<T> = arange(0, stop, step)

/**
 * Returns evenly spaced values within a given interval, where start of interval is 0 and the [step] is Double.
 * see [Multik.arange]
 *
 * @param stop end value of the interval. The interval doesn't include this value.
 * @param step spacing between value. The step is [Double].
 * @return [D1Array].
 * @sample samples.NDArrayTest.arangeDoubleStepWithoutStart
 */
public inline fun <reified T : Number> Multik.arange(stop: Int, step: Double): D1Array<T> = arange(0, stop, step)


/**
 * Returns evenly spaced values within a given interval [start, stop], more precisely than [arange].
 *
 * @param start starting value of the interval.
 * @param stop end of the interval.
 * @param num number of values. Default is 50.
 * @return [D1Array].
 * @sample samples.NDArrayTest.linspace
 */
public inline fun <reified T : Number> Multik.linspace(start: Int, stop: Int, num: Int = 50): D1Array<T> {
    return linspace(start.toDouble(), stop.toDouble(), num)
}

/**
 * Returns evenly spaced values within a given interval [start, stop], where [start] and [stop] is Double.
 * @see [Multik.linspace]
 *
 * @param start starting value of the interval.
 * @param stop end of the interval.
 * @param num number of values.
 * @return [D1Array].
 * @sample samples.NDArrayTest.linspaceDouble
 */
public inline fun <reified T : Number> Multik.linspace(start: Double, stop: Double, num: Int = 50): D1Array<T> {
    require(num > 0) { "The number of elements cannot be less than zero or equal to zero." }
    val div = num - 1.0
    val delta = stop - start
    val ret = arange<Double>(0, stop = num)
    if (num > 1) {
        val step = delta / div
        ret *= step
    }

    ret += start
    ret[ret.size - 1] = stop

    val data = initMemoryView(ret.data.size, DataType.ofKClass(T::class)) { ret.data[it].toPrimitiveType<T>() }
    return D1Array(data, ret.offset, ret.shape, ret.strides, ret.dim)
}

/**
 * Return coordinate matrices from coordinate vectors.
 */
public fun <T : Number> Multik.meshgrid(x: MultiArray<T, D1>, y: MultiArray<T, D1>): Pair<D2Array<T>, D2Array<T>> =
    Pair(mk.stack(List(y.size) { x }), mk.stack(List(x.size) { y }, axis = 1))

/**
 * Returns [D1Array] containing all elements.
 * @sample samples.NDArrayTest.toNDArray
 */
public inline fun <reified T : Number> Iterable<T>.toNDArray(): D1Array<T> = this.toCommonNDArray()

/**
 * Returns [D1Array] containing all elements.
 * @sample samples.NDArrayTest.toNDArray
 */
@JvmName("toComplexNDArray")
public inline fun <reified T : Complex> Iterable<T>.toNDArray(): D1Array<T> = this.toCommonNDArray()

/**
 * Returns [D2Array] containing all elements.
 */
@JvmName("List2DToNDArrayNumber")
public inline fun <reified T : Number> List<List<T>>.toNDArray(): D2Array<T> = Multik.ndarray(this)

/**
 * Returns [D2Array] containing all elements.
 */
@JvmName("List2DToNDArrayComplex")
public inline fun <reified T : Complex> List<List<T>>.toNDArray(): D2Array<T> = Multik.ndarray(this)

/**
 * Returns [D3Array] containing all elements.
 */
@JvmName("List3DToNDArrayNumber")
public inline fun <reified T : Number> List<List<List<T>>>.toNDArray(): D3Array<T> = Multik.ndarray(this)

/**
 * Returns [D3Array] containing all elements.
 */
@JvmName("List3DToNDArrayComplex")
public inline fun <reified T : Complex> List<List<List<T>>>.toNDArray(): D3Array<T> = Multik.ndarray(this)

/**
 * Returns [D4Array] containing all elements.
 */
@JvmName("List4DToNDArrayNumber")
public inline fun <reified T : Number> List<List<List<List<T>>>>.toNDArray(): D4Array<T> = Multik.ndarray(this)

/**
 * Returns [D4Array] containing all elements.
 */
@JvmName("List4DToNDArrayComplex")
public inline fun <reified T : Complex> List<List<List<List<T>>>>.toNDArray(): D4Array<T> = Multik.ndarray(this)

@PublishedApi
@Suppress("UNCHECKED_CAST")
internal fun <T> Iterable<T>.toCommonNDArray(dtype: DataType): D1Array<T> {
    if (this is Collection<T>)
        return ndarrayCommon(this, intArrayOf(this.size), D1, dtype) as D1Array<T>

    val tmp = ArrayList<T>()
    for (item in this) {
        tmp.add(item)
    }
    return ndarrayCommon(tmp, intArrayOf(tmp.size), D1, dtype) as D1Array<T>
}

@PublishedApi
@Suppress("UNCHECKED_CAST")
internal inline fun <reified T: Any> Iterable<T>.toCommonNDArray(): D1Array<T> {
    val dtype: DataType = DataType.ofKClass(T::class)
    if (this is Collection<T>)
        return ndarrayCommon(this, intArrayOf(this.size), D1, dtype) as D1Array<T>

    val tmp = ArrayList<T>()
    for (item in this) {
        tmp.add(item)
    }
    return ndarrayCommon(tmp, intArrayOf(tmp.size), D1, dtype) as D1Array<T>
}

