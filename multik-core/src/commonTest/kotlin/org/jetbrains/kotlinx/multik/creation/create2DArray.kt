/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Create2DArrayTests {

    /*___________________________Byte_______________________________________*/

    /**
     * This method checks if a byte array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledByteArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Byte>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0.toByte() } }
    }

    /**
     * Creates a byte array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createByteArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Byte>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1.toByte() } }
    }

    /**
     * Tests the function 'mk.identity<Byte>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityByteMatrix() {
        val n = 7
        val a = mk.identity<Byte>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1, a[i, j], "Expected diagonal elements to be 1")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Byte>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalByteMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { (it + 1).toByte() })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals((i + 1).toByte(), a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromByteList() {
        val list = listOf(listOf<Byte>(1, 3, 8), listOf<Byte>(4, 7, 2))
        val a: D2Array<Byte> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of bytes
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromByteSet() {
        val set = setOf<Byte>(1, 3, 8, 4, 9, 2)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Byte> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive ByteArray
     * and checks if the array's ByteArray representation matches the input ByteArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveByteArray() {
        val array = byteArrayOf(1, 3, 8, 4, 9, 2)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getByteArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte2DArrayWithInitializationFunction() {
        val a = mk.d2array<Byte>(2, 3) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7, 8)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices(2, 3) { i, j -> (i * j + 7).toByte() }
        val expected = byteArrayOf(7, 7, 7, 7, 8, 9)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedByte2DArray() {
        val list = listOf(
            listOf<Byte>(1, 3, 8),
            listOf<Byte>(4, 7),
            listOf<Byte>(2, 5, 9, 10)
        )

        val expected = listOf(
            listOf<Byte>(1, 3, 8, 0),
            listOf<Byte>(4, 7, 0, 0),
            listOf<Byte>(2, 5, 9, 10)
        )

        val a: D2Array<Byte> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________Short_______________________________________*/

    /**
     * This method checks if a short array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledShortArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Short>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0.toShort() } }
    }

    /**
     * Creates a short array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createShortArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Short>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1.toShort() } }
    }

    /**
     * Tests the function 'mk.identity<Short>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityShortMatrix() {
        val n = 7
        val a = mk.identity<Short>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1, a[i, j], "Expected diagonal elements to be 1")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Short>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalShortMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { (it + 1).toShort() })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals((i + 1).toShort(), a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }


    /**
     * Creates a two-dimensional array from a list of short lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromShortList() {
        val list = listOf(listOf<Short>(1, 3, 8), listOf<Short>(4, 7, 2))
        val a: D2Array<Short> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of shorts
     * and checks if the array's set representation matches the input set.
     */
    @Test
    @Ignore
    fun createTwoDimensionalArrayFromShortSet() {
        val set = setOf<Short>(1, 3, 8, 4, 9, 2)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Short> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive ShortArray
     * and checks if the array's ShortArray representation matches the input ShortArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveShortArray() {
        val array = shortArrayOf(1, 3, 8, 4, 9, 2)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getShortArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort2DArrayWithInitializationFunction() {
        val a = mk.d2array<Short>(2, 3) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7, 8)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices(2, 3) { i, j -> (i * j + 7).toShort() }
        val expected = shortArrayOf(7, 7, 7, 7, 8, 9)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedShort2DArray() {
        val list = listOf(
            listOf<Short>(1, 3, 8),
            listOf<Short>(4, 7),
            listOf<Short>(2, 5, 9, 10)
        )

        val expected = listOf(
            listOf<Short>(1, 3, 8, 0),
            listOf<Short>(4, 7, 0, 0),
            listOf<Short>(2, 5, 9, 10)
        )

        val a: D2Array<Short> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________Int_______________________________________*/


    /**
     * This method checks if an integer array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledIntArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Int>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0 } }
    }

    /**
     * Creates an integer array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createIntArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Int>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1 } }
    }

    /**
     * Tests the function 'mk.identity<Int>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityIntMatrix() {
        val n = 7
        val a = mk.identity<Int>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1, a[i, j], "Expected diagonal elements to be 1")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Int>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalIntMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { it + 1 })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(i + 1, a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }


    /**
     * Creates a two-dimensional array from a list of integer lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromIntList() {
        val list = listOf(listOf(1, 3, 8), listOf(4, 7, 2))
        val a: D2Array<Int> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of integers
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromIntSet() {
        val set = setOf(1, 3, 8, 4, 9, 2)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Int> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive IntArray
     * and checks if the array's IntArray representation matches the input IntArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveIntArray() {
        val array = intArrayOf(1, 3, 8, 4, 9, 2)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getIntArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt2DArrayWithInitializationFunction() {
        val a = mk.d2array<Int>(2, 3) { (it + 3) }
        val expected = intArrayOf(3, 4, 5, 6, 7, 8)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices(2, 3) { i, j -> i * j + 7 }
        val expected = intArrayOf(7, 7, 7, 7, 8, 9)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedInt2DArray() {
        val list = listOf(
            listOf(1, 3, 8),
            listOf(4, 7),
            listOf(2, 5, 9, 10)
        )

        val expected = listOf(
            listOf(1, 3, 8, 0),
            listOf(4, 7, 0, 0),
            listOf(2, 5, 9, 10)
        )

        val a: D2Array<Int> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________Long_______________________________________*/


    /**
     * This method checks if a long array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledLongArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Long>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0L } }
    }

    /**
     * Creates a long array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createLongArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Long>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1L } }
    }

    /**
     * Tests the function 'mk.identity<Long>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityLongMatrix() {
        val n = 7
        val a = mk.identity<Long>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1L, a[i, j], "Expected diagonal elements to be 1")
                else
                    assertEquals(0L, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Long>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalLongMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { it + 1L })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(i + 1L, a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0L, a[i, j], "Expected non-diagonal elements to be 0")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of long lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromLongList() {
        val list = listOf(listOf(1L, 3L, 8L), listOf(4L, 7L, 2L))
        val a: D2Array<Long> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of longs
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromLongSet() {
        val set = setOf(1L, 3L, 8L, 4L, 9L, 2L)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Long> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive LongArray
     * and checks if the array's LongArray representation matches the input LongArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveLongArray() {
        val array = longArrayOf(1, 3, 8, 4, 9, 2)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getLongArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong2DArrayWithInitializationFunction() {
        val a = mk.d2array<Long>(2, 3) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7, 8)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices<Long>(2, 3) { i, j -> i * j + 7L }
        val expected = longArrayOf(7, 7, 7, 7, 8, 9)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedLong2DArray() {
        val list = listOf(
            listOf<Long>(1, 3, 8),
            listOf<Long>(4, 7),
            listOf<Long>(2, 5, 9, 10)
        )

        val expected = listOf(
            listOf<Long>(1, 3, 8, 0),
            listOf<Long>(4, 7, 0, 0),
            listOf<Long>(2, 5, 9, 10)
        )

        val a: D2Array<Long> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________Float_______________________________________*/

    /**
     * This method checks if a float array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledFloatArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Float>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0f } }
    }

    /**
     * Creates a float array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createFloatArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Float>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1f } }
    }

    /**
     * Tests the function 'mk.identity<Float>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityFloatMatrix() {
        val n = 7
        val a = mk.identity<Float>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1f, a[i, j], "Expected diagonal elements to be 1.0")
                else
                    assertEquals(0f, a[i, j], "Expected non-diagonal elements to be 0.0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Float>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalFloatMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { it + 1f })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(i + 1f, a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0.0f, a[i, j], "Expected non-diagonal elements to be 0.0")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromFloatList() {
        val list = listOf(listOf(1f, 3f, 8f), listOf(4f, 7f, 2f))
        val a: D2Array<Float> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromFloatSet() {
        val set = setOf(1f, 3f, 8f, 4f, 9f, 2f)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Float> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive FloatArray
     * and checks if the array's FloatArray representation matches the input FloatArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveFloatArray() {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f, 2f)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getFloatArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat2DArrayWithInitializationFunction() {
        val a = mk.d2array<Float>(2, 3) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices<Float>(2, 3) { i, j -> i * j + 7f }
        val expected = floatArrayOf(7f, 7f, 7f, 7f, 8f, 9f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedFloat2DArray() {
        val list = listOf(
            listOf(1f, 3f, 8f),
            listOf(4f, 7f),
            listOf(2f, 5f, 9f, 10f)
        )

        val expected = listOf(
            listOf(1f, 3f, 8f, 0f),
            listOf(4f, 7f, 0f, 0f),
            listOf(2f, 5f, 9f, 10f)
        )

        val a: D2Array<Float> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________Double_______________________________________*/


    /**
     * This method checks if a double array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledDoubleArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<Double>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 0.0 } }
    }

    /**
     * Creates a double array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createDoubleArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<Double>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == 1.0 } }
    }

    /**
     * Tests the function 'mk.identity<Double>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityDoubleMatrix() {
        val n = 7
        val a = mk.identity<Double>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(1.0, a[i, j], "Expected diagonal elements to be 1.0")
                else
                    assertEquals(0.0, a[i, j], "Expected non-diagonal elements to be 0.0")
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<Double>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalDoubleMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { it + 1.0 })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(i + 1.0, a[i, j], "Expected element at [$i:$j] to be ${i + 1}")
                else
                    assertEquals(0.0, a[i, j], "Expected non-diagonal elements to be 0.0")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of double lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromDoubleList() {
        val list = listOf(listOf(1.0, 3.0, 8.0), listOf(4.0, 7.0, 2.0))
        val a: D2Array<Double> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromDoubleSet() {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0)
        val shape = intArrayOf(2, 3)
        val a: D2Array<Double> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive DoubleArray
     * and checks if the array's DoubleArray representation matches the input DoubleArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveDoubleArray() {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getDoubleArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble2DArrayWithInitializationFunction() {
        val a = mk.d2array<Double>(2, 3) { it + 3.0 }
        val expected = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices<Double>(2, 3) { i, j -> i * j + 7.0 }
        val expected = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 8.0, 9.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a two-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedDouble2DArray() {
        val list = listOf(
            listOf(1.0, 3.0, 8.0),
            listOf(4.0, 7.0),
            listOf(2.0, 5.0, 9.0, 10.0)
        )

        val expected = listOf(
            listOf(1.0, 3.0, 8.0, 0.0),
            listOf(4.0, 7.0, 0.0, 0.0),
            listOf(2.0, 5.0, 9.0, 10.0)
        )

        val a: D2Array<Double> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD2())
    }


    /*___________________________ComplexFloat_______________________________________*/

    /**
     * This method checks if a ComplexFloat array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexFloatArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<ComplexFloat>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == ComplexFloat.zero } }
    }

    /**
     * Creates a ComplexFloat array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexFloatArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<ComplexFloat>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == ComplexFloat.one } }
    }

    /**
     * Tests the function 'mk.identity<ComplexFloat>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityComplexFloatMatrix() {
        val n = 7
        val a = mk.identity<ComplexFloat>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(ComplexFloat.one, a[i, j], "Expected diagonal elements to be ${ComplexFloat.one}")
                else
                    assertEquals(
                        ComplexFloat.zero,
                        a[i, j],
                        "Expected non-diagonal elements to be ${ComplexFloat.zero}"
                    )
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<ComplexFloat>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalComplexFloatMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { ComplexFloat(it + 1f, it + 1f) })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(ComplexFloat(i + 1f, i + 1f), a[i, j], "Expected element at [$i:$j] to be ${ComplexFloat(i + 1f, i + 1f)}")
                else
                    assertEquals(ComplexFloat.zero, a[i, j], "Expected non-diagonal elements to be ${ComplexFloat.zero}")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of complex float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromComplexFloatList() {
        val list = listOf(
            listOf(ComplexFloat.one, 3f + 0f.i, 8f + 0f.i),
            listOf(4f + 0f.i, 7f + 0f.i, 2f + 0f.i)
        )
        val a: D2Array<ComplexFloat> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }

    /**
     * Creates a two-dimensional array from a set of complex floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromComplexFloatSet() {
        val set = setOf(1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i)
        val shape = intArrayOf(2, 3)
        val a: D2Array<ComplexFloat> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive ComplexFloatArray
     * and checks if the array's ComplexFloatArray representation matches the input ComplexFloatArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveComplexFloatArray() {
        val array = complexFloatArrayOf(1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getComplexFloatArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat2DArrayWithInitializationFunction() {
        val a = mk.d2array<ComplexFloat>(2, 3) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
        val expected = complexFloatArrayOf(
            3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i,
            6.21f + 2.168f.i, 7.21f + 3.168f.i, 8.21f + 4.168f.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices<ComplexFloat>(2, 3) { i, j -> i * j + ComplexFloat(7) }
        val expected = complexFloatArrayOf(7f + 0f.i, 7f + 0f.i, 7f + 0f.i, 7f + 0f.i, 8f + 0f.i, 9f + 0f.i)

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }


    /*___________________________ComplexDouble_______________________________________*/

    /**
     * This method checks if a ComplexDouble array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexDoubleArray() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.zeros<ComplexDouble>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == ComplexDouble.zero } }
    }

    /**
     * Creates a ComplexDouble array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexDoubleArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val a = mk.ones<ComplexDouble>(dim1, dim2)

        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == ComplexDouble.one } }
    }

    /**
     * Tests the function 'mk.identity<ComplexDouble>(n)' that creates an identity matrix of size n x n.
     * The test asserts that:
     * - The size of the resulting matrix matches n*n.
     * - The diagonal elements of the matrix are 1 and the non-diagonal elements are 0.
     */
    @Test
    fun createIdentityMatrix() {
        val n = 7
        val a = mk.identity<ComplexDouble>(n)

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(ComplexDouble.one, a[i, j], "Expected diagonal elements to be ${ComplexDouble.one}")
                else
                    assertEquals(
                        ComplexDouble.zero,
                        a[i, j],
                        "Expected non-diagonal elements to be ${ComplexDouble.zero}"
                    )
            }
        }
    }

    /**
     * Tests the function 'mk.diagonal<ComplexDouble>(elements)' that creates an identity matrix of size elements.size x elements.size.
     * The test asserts that:
     * - The size of the resulting matrix matches elements.size x elements.size.
     * - The diagonal elements of the matrix are as provided and the non-diagonal elements are 0.
     */
    @Test
    fun createDiagonalComplexDoubleMatrix() {
        val n = 7
        val a = mk.diagonal(List(n) { ComplexDouble(it + 1.0, it + 1.0) })

        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(ComplexDouble(i + 1.0, i + 1.0), a[i, j], "Expected element at [$i:$j] to be ${ComplexDouble(i + 1.0, i + 1.0)}")
                else
                    assertEquals(ComplexDouble.zero, a[i, j], "Expected non-diagonal elements to be ${ComplexDouble.zero}")
            }
        }
    }

    /**
     * Creates a two-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createTwoDimensionalArrayFromComplexDoubleList() {
        val list = listOf(
            listOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i),
            listOf(4.0 + .0.i, 7.0 + .0.i, 2.0 + .0.i)
        )
        val a: D2Array<ComplexDouble> = mk.ndarray(list)
        assertEquals(list, a.toListD2())
    }


    /**
     * Creates a two-dimensional array from a set of complex doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createTwoDimensionalArrayFromComplexDoubleSet() {
        val set = setOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i)
        val shape = intArrayOf(2, 3)
        val a: D2Array<ComplexDouble> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a two-dimensional array from a primitive ComplexDoubleArray
     * and checks if the array's ComplexDoubleArray representation matches the input ComplexDoubleArray.
     */
    @Test
    fun createTwoDimensionalArrayFromPrimitiveComplexDoubleArray() {
        val array = complexDoubleArrayOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i)
        val a = mk.ndarray(array, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getComplexDoubleArray() shouldBe array
    }


    /**
     * Creates a two-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble2DArrayWithInitializationFunction() {
        val a = mk.d2array<ComplexDouble>(2, 3) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
        val expected = complexDoubleArrayOf(
            3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i,
            6.21 + 2.168.i, 7.21 + 3.168.i, 8.21 + 4.168.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }

    /**
     * Creates a two-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble2DArrayWithInitAndIndices() {
        val a = mk.d2arrayIndices<ComplexDouble>(2, 3) { i, j -> i * j + ComplexDouble(7) }
        val expected = complexDoubleArrayOf(7.0 + .0.i, 7.0 + .0.i, 7.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i, 9.0 + .0.i)

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }

}
