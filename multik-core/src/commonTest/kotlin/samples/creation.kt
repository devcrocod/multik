/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package samples

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import kotlin.test.Test

class NDArrayTest {
    @Test
    fun zerosD1() {
        val ndarray = mk.zeros<Int>(5)
        println(ndarray) // [0, 0, 0, 0, 0]
    }

    @Test
    fun zerosD2() {
        val ndarray = mk.zeros<Float>(2, 2)
        println(ndarray)
        /*
         [[0.0, 0.0],
          [0.0, 0.0]]
         */
    }

    @Test
    fun zerosD3() {
        val ndarray = mk.zeros<Double>(2, 2, 2)
        println(ndarray)
        /*
         [[[0.0, 0.0],
           [0.0, 0.0]],

          [[0.0, 0.0],
           [0.0, 0.0]]]
         */
    }

    @Test
    fun zerosD4() {
        val ndarray = mk.zeros<Double>(2, 1, 2, 1)
        println(ndarray)
        /*
         [[[[0.0],
            [0.0]]],


          [[[0.0],
            [0.0]]]]
         */
}

    @Test
    fun zerosDN() {
        val ndarray = mk.zeros<Double>(1, 1, 1, 2, 2)
        println(ndarray)
        /*
         [[[[[0.0, 0.0],
             [0.0, 0.0]]]]]
         */
    }

    @Test
    fun zerosDNWithDtype() {
        val dims = intArrayOf(3, 2)
        val ndarray = mk.zeros<Float, D2>(dims, DataType.FloatDataType)
        println(ndarray)
        /*
         [[0.0, 0.0],
          [0.0, 0.0],
          [0.0, 0.0]]
         */
    }

    @Test
    fun onesD1() {
        val ndarray = mk.ones<Int>(5)
        println(ndarray) // [1, 1, 1, 1, 1]
    }

    @Test
    fun onesD2() {
        val ndarray = mk.ones<Float>(2, 2)
        println(ndarray)
        /*
         [[1.0, 1.0],
          [1.0, 1.0]]
         */
    }

    @Test
    fun onesD3() {
        val ndarray = mk.ones<Double>(2, 2, 2)
        println(ndarray)
        /*
         [[[1.0, 1.0],
           [1.0, 1.0]],

          [[0.0, 0.0],
           [0.0, 0.0]]]
         */
    }

    @Test
    fun onesD4() {
        val ndarray = mk.ones<Double>(2, 1, 2, 1)
        println(ndarray)
        /*
         [[[[1.0],
            [1.0]]],


          [[[1.0],
            [1.0]]]]
         */
    }

    @Test
    fun onesDN() {
        val ndarray = mk.ones<Double>(1, 1, 1, 2, 2)
        println(ndarray)
        /*
         [[[[[1.0, 1.0],
             [1.0, 1.0]]]]]
         */
    }

    @Test
    fun onesDNWithDtype() {
        val dims = intArrayOf(3, 2)
        val ndarray = mk.ones<Float, D2>(dims, DataType.FloatDataType)
        println(ndarray)
        /*
         [[1.0, 1.0],
          [1.0, 1.0],
          [1.0, 1.0]]
         */
    }

    @Test
    fun identity() {
        val identNDArray = mk.identity<Long>(3)
        println(identNDArray)
        /*
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
         */
    }

    @Test
    fun identityWithDtype() {
        val identNDArray = mk.identity<Long>(3, DataType.LongDataType)
        println(identNDArray)
        /*
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
         */
    }

    @Test
    fun diagonal() {
        val diagNDArray = mk.diagonal(mk[2, 4, 8])
        println(diagNDArray)
        /*
        [[2, 0, 0],
        [0, 4, 0],
        [0, 0, 8]]
         */
    }

    @Test
    fun diagonalWithDtype() {
        val diagNDArray = mk.diagonal(mk[2, 4, 8], DataType.LongDataType)
        println(diagNDArray)
        /*
        [[2, 0, 0],
        [0, 4, 0],
        [0, 0, 8]]
         */
    }

    @Test
    fun ndarray1D() {
        val ndarray = mk.ndarray(mk[1, 2, 3])
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(3), dim=1
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun ndarray2D() {
        val ndarray = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(2, 2), dim=2
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarray3D() {
        val ndarray = mk.ndarray(mk[mk[mk[1, 2], mk[3, 4]], mk[mk[5, 6], mk[7, 8]]])
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(2, 2, 2), dim=3
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarray4D() {
        val ndarray = mk.ndarray(mk[mk[mk[mk[1, 2], mk[3, 4]], mk[mk[5, 6], mk[7, 8]]]])
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(1, 2, 2, 2), dim=4
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]]
         */
    }

    @Test
    fun ndarrayCollections() {
        val set = setOf(1, 2, 3, 4)
        val dims = intArrayOf(2, 1, 2)
        val ndarray = mk.ndarray<Int, D3>(set, dims)
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(2, 1, 2), dim=3
        println(ndarray)
        /*
        [[[1, 2]],

        [[3, 4]]]
         */
    }

    @Test
    fun ndarrayCollectionsWithDim() {
        val set = setOf(1, 2, 3, 4)
        val dims = intArrayOf(2, 1, 2)
        val ndarray = mk.ndarray(set, dims, D3)
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(2, 1, 2), dim=3
        println(ndarray)
        /*
        [[[1, 2]],

        [[3, 4]]]
         */
    }

    @Test
    fun ndarrayCollections1D() {
        val set = setOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(set)
        println("shape=(${ndarray.shape.joinToString()}), dim=${ndarray.dim.d}") // shape=(4), dim=1
        println(ndarray) // [1, 2, 3, 4]
    }

    @Test
    fun ndarrayCollections2D() {
        val set = setOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(set, 2, 2)
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarrayCollections3D() {
        val set = setOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(set, 2, 2, 2)
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarrayCollections4D() {
        val set = setOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(set, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]],


        [[[9, 10],
        [11, 12]],

        [[13, 14],
        [15, 16]]]]
         */
    }

    @Test
    fun ndarrayCollectionsDN() {
        val set = setOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(set, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1, 2],
        [3, 4]]],


        [[[5, 6],
        [7, 8]]]],



        [[[[9, 10],
        [11, 12]]],


        [[[13, 14],
        [15, 16]]]]]
         */
    }

    @Test
    fun ndarrayByteArray1D() {
        val byteArray = byteArrayOf(1, 2, 3)
        val ndarray = mk.ndarray(byteArray)
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun ndarrayByteArray2D() {
        val byteArray = byteArrayOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(byteArray, 2, 2)
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarrayByteArray3D() {
        val byteArray = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(byteArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarrayByteArray4D() {
        val byteArray = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(byteArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]],


        [[[9, 10],
        [11, 12]],

        [[13, 14],
        [15, 16]]]]
         */
    }

    @Test
    fun ndarrayByteArrayDN() {
        val byteArray = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(byteArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1, 2],
        [3, 4]]],


        [[[5, 6],
        [7, 8]]]],



        [[[[9, 10],
        [11, 12]]],


        [[[13, 14],
        [15, 16]]]]]
         */
    }

    @Test
    fun ndarrayShortArray1D() {
        val shortArray = shortArrayOf(1, 2, 3)
        val ndarray = mk.ndarray(shortArray)
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun ndarrayShortArray2D() {
        val shortArray = shortArrayOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(shortArray, 2, 2)
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarrayShortArray3D() {
        val shortArray = shortArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(shortArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarrayShortArray4D() {
        val shortArray = shortArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(shortArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]],


        [[[9, 10],
        [11, 12]],

        [[13, 14],
        [15, 16]]]]
         */
    }

    @Test
    fun ndarrayShortArrayDN() {
        val shortArray = shortArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(shortArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1, 2],
        [3, 4]]],


        [[[5, 6],
        [7, 8]]]],



        [[[[9, 10],
        [11, 12]]],


        [[[13, 14],
        [15, 16]]]]]
         */
    }

    @Test
    fun ndarrayIntArray1D() {
        val intArray = intArrayOf(1, 2, 3)
        val ndarray = mk.ndarray(intArray)
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun ndarrayIntArray2D() {
        val intArray = intArrayOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(intArray, 2, 2)
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarrayIntArray3D() {
        val intArray = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(intArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarrayIntArray4D() {
        val intArray = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(intArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]],


        [[[9, 10],
        [11, 12]],

        [[13, 14],
        [15, 16]]]]
         */
    }

    @Test
    fun ndarrayIntArrayDN() {
        val intArray = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(intArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1, 2],
        [3, 4]]],


        [[[5, 6],
        [7, 8]]]],



        [[[[9, 10],
        [11, 12]]],


        [[[13, 14],
        [15, 16]]]]]
         */
    }

    @Test
    fun ndarrayLongArray1D() {
        val longArray = longArrayOf(1, 2, 3)
        val ndarray = mk.ndarray(longArray)
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun ndarrayLongArray2D() {
        val longArray = longArrayOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(longArray, 2, 2)
        println(ndarray)
        /*
        [[1, 2],
        [3, 4]]
         */
    }

    @Test
    fun ndarrayLongArray3D() {
        val longArray = longArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(longArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]]
         */
    }

    @Test
    fun ndarrayLongArray4D() {
        val longArray = longArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(longArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1, 2],
        [3, 4]],

        [[5, 6],
        [7, 8]]],


        [[[9, 10],
        [11, 12]],

        [[13, 14],
        [15, 16]]]]
         */
    }

    @Test
    fun ndarrayLongArrayDN() {
        val longArray = longArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        val ndarray = mk.ndarray(longArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1, 2],
        [3, 4]]],


        [[[5, 6],
        [7, 8]]]],



        [[[[9, 10],
        [11, 12]]],


        [[[13, 14],
        [15, 16]]]]]
         */
    }

    @Test
    fun ndarrayFloatArray1D() {
        val floatArray = floatArrayOf(1f, 2f, 3f)
        val ndarray = mk.ndarray(floatArray)
        println(ndarray) // [1.0, 2.0, 3.0]
    }

    @Test
    fun ndarrayFloatArray2D() {
        val floatArray = floatArrayOf(1f, 2f, 3f, 4f)
        val ndarray = mk.ndarray(floatArray, 2, 2)
        println(ndarray)
        /*
        [[1.0, 2.0],
        [3.0, 4.0]]
         */
    }

    @Test
    fun ndarrayFloatArray3D() {
        val floatArray = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val ndarray = mk.ndarray(floatArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1.0, 2.0],
        [3.0, 4.0]],

        [[5.0, 6.0],
        [7.0, 8.0]]]
         */
    }

    @Test
    fun ndarrayFloatArray4D() {
        val floatArray = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)
        val ndarray = mk.ndarray(floatArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1.0, 2.0],
        [3.0, 4.0]],

        [[5.0, 6.0],
        [7.0, 8.0]]],


        [[[9.0, 10.0],
        [11.0, 12.0]],

        [[13.0, 14.0],
        [15.0, 16.0]]]]
         */
    }

    @Test
    fun ndarrayFloatArrayDN() {
        val floatArray = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)
        val ndarray = mk.ndarray(floatArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1.0, 2.0],
        [3.0, 4.0]]],


        [[[5.0, 6.0],
        [7.0, 8.0]]]],



        [[[[9.0, 10.0],
        [11.0, 12.0]]],


        [[[13.0, 14.0],
        [15.0, 16.0]]]]]
         */
    }

    @Test
    fun ndarrayDoubleArray1D() {
        val doubleArray = doubleArrayOf(1.0, 2.0, 3.0)
        val ndarray = mk.ndarray(doubleArray)
        println(ndarray) // [1.0, 2.0, 3.0]
    }

    @Test
    fun ndarrayDoubleArray2D() {
        val doubleArray = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val ndarray = mk.ndarray(doubleArray, 2, 2)
        println(ndarray)
        /*
        [[1.0, 2.0],
        [3.0, 4.0]]
         */
    }

    @Test
    fun ndarrayDoubleArray3D() {
        val doubleArray = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val ndarray = mk.ndarray(doubleArray, 2, 2, 2)
        println(ndarray)
        /*
        [[[1.0, 2.0],
        [3.0, 4.0]],

        [[5.0, 6.0],
        [7.0, 8.0]]]
         */
    }

    @Test
    fun ndarrayDoubleArray4D() {
        val doubleArray =
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
        val ndarray = mk.ndarray(doubleArray, 2, 2, 2, 2)
        println(ndarray)
        /*
        [[[[1.0, 2.0],
        [3.0, 4.0]],

        [[5.0, 6.0],
        [7.0, 8.0]]],


        [[[9.0, 10.0],
        [11.0, 12.0]],

        [[13.0, 14.0],
        [15.0, 16.0]]]]
         */
    }

    @Test
    fun ndarrayDoubleArrayDN() {
        val doubleArray =
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
        val ndarray = mk.ndarray(doubleArray, 2, 2, 1, 2, 2)
        println(ndarray)
        /*
        [[[[[1.0, 2.0],
        [3.0, 4.0]]],


        [[[5.0, 6.0],
        [7.0, 8.0]]]],



        [[[[9.0, 10.0],
        [11.0, 12.0]]],


        [[[13.0, 14.0],
        [15.0, 16.0]]]]]
         */
    }

    @Test
    fun d1array() {
        val ndarray = mk.d1array(3) { -it }
        println(ndarray) // [0, -1, -2]
    }

    @Test
    fun d2array() {
        val ndarray = mk.d2array(2, 2) { it * it }
        println(ndarray)
        /*
        [[0, 1],
         [4, 9]]
         */
    }

    @Test
    fun d2arrayIndices() {
        val ndarray = mk.d2arrayIndices(2, 2) { i, j -> i + j }
        println(ndarray)
        /*
        [[0, 1],
         [1, 2]]
         */
    }

    @Test
    fun d3array() {
        val ndarray = mk.d3array(2, 2, 2) { if (it % 2 == 0) 0 else it * it }
        println(ndarray)
        /*
        [[[0, 1],
        [0, 9]],

        [[0, 25],
        [0, 49]]]
         */
    }

    @Test
    fun d3arrayIndices() {
        val ndarray = mk.d3arrayIndices(2, 2, 2) { i, j, k -> i + j + k }
        println(ndarray)
        /*
        [[[0, 1],
          [1, 2]],

         [[1, 2],
          [2, 3]]]
         */
    }

    @Test
    fun d4array() {
        val ndarray = mk.d4array(2, 1, 2, 1) { (it - 10f) / 10 }
        println(ndarray)
        /*
        [[[[-1.0],
        [-0.9]]],


        [[[-0.8],
        [-0.7]]]]
         */
    }

    @Test
    fun d4arrayIndices() {
        val ndarray = mk.d4arrayIndices(2, 1, 2, 1) {i, j, k, m ->
            (i - j * 5f) / (k + 1) + m
        }
        println(ndarray)
        /*
        [[[[0.0],
           [0.0]]],


         [[[1.0],
           [0.5]]]]
         */
    }

    @Test
    fun dnarray() {
        val ndarray = mk.dnarray(1, 2, 1, 2, 1, 2) { kotlin.math.PI * it }
        println(ndarray)
        /*
        [[[[[[0.0, 3.141592653589793]],

        [[6.283185307179586, 9.42477796076938]]]],



        [[[[12.566370614359172, 15.707963267948966]],

        [[18.84955592153876, 21.991148575128552]]]]]]
         */
    }

    @Test
    fun dnarrayWithDims() {
        val dims = intArrayOf(1, 1, 2, 1, 1)
        val ndarray = mk.dnarray<Double, DN>(dims) { kotlin.math.PI * it }
        println(ndarray)
        /*
        [[[[[0.0]],

        [[3.141592653589793]]]]]
         */
    }

    @Test
    fun ndarrayOf() {
        val ndarray = mk.ndarrayOf(1, 2, 3)
        println(ndarray) // [1, 2, 3]
    }

    @Test
    fun arange() {
        val ndarray = mk.arange<Float>(start = 2, stop = 5)
        val ndarrayWithStep = mk.arange<Int>(2, 7, 2)

        println(ndarray) // [2.0, 3.0, 4.0]
        println(ndarrayWithStep) // [2, 4, 6]
    }

    @Test
    fun arangeDoubleStep() {
        val ndarray = mk.arange<Float>(1, 7, 1.3)
        println(ndarray) // [1.0, 2.3, 3.6, 4.9, 6.2]
    }

    @Test
    fun arangeWithoutStart() {
        val ndarray = mk.arange<Int>(10)
        val ndarrayStep = mk.arange<Long>(10, 2)
        println(ndarray) // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        println(ndarrayStep) // [0, 2, 4, 6, 8]
    }

    @Test
    fun arangeDoubleStepWithoutStart() {
        val ndarray = mk.arange<Double>(5, 0.5)
        println(ndarray) // [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    }

    @Test
    fun linspace() {
        val ndarray = mk.linspace<Float>(-1, 1, 9)
        println(ndarray) // [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    }

    @Test
    fun linspaceDouble() {
        val ndarray = mk.linspace<Double>(-1.5, 2.1, 7)
        println(ndarray) // [-1.5, -0.9, -0.30000000000000004, 0.2999999999999998, 0.8999999999999999, 1.5, 2.1]
    }

    @Test
    fun toNDArray() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = list.toNDArray()
        println(ndarray) // [1, 2, 3, 4]
    }
}