/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_kotlin.linAlg


import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.Norm
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.kotlin.linalg.*
import org.jetbrains.kotlinx.multik.kotlin.linalg.KELinAlgEx.solve
import org.jetbrains.kotlinx.multik.kotlin.linalg.KELinAlgEx.solveC
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random
import kotlin.test.*

class KELinAlgTest {

    @Test
    fun vector_vectorDotTestWithFloatComplexNumbers() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexFloat(random.nextFloat(), random.nextFloat()) }
        val vector2 = mk.d1array(9) { ComplexFloat(random.nextFloat(), random.nextFloat()) }

        val actual = KELinAlg.dot(vector1, vector2)
        println(actual)
//        assertFloatingComplexNumber(0.23971967f + 4.6530027f.i, actual)
    }

    @Test
    fun vector_vectorDotTestWithDouble32ComplexNumbers() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexDouble(1.0, 0.0) }
        val vector2 = mk.d1array(9) { ComplexDouble(random.nextFloat(), random.nextFloat()) }

        val actual = KELinAlg.dot(vector1, vector2)
        println(actual)
//        assertFloatingComplexNumber(4.174294710159302 + 5.029674530029297.i, actual)
    }

    @Test
    fun vector_vectorDotTestWithDouble64ComplexNumbers() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexDouble(random.nextDouble(), random.nextDouble()) }
        val vector2 = mk.d1array(9) { ComplexDouble(random.nextDouble(), random.nextDouble()) }

        val actual = KELinAlg.dot(vector1, vector2)
        println(actual)
//        assertFloatingComplexNumber(0.17520206558121712 + 4.552420480555579.i, actual)
    }

    @Test
    fun test_of_norm_function_with_p_equals_1() {
        val d2arrayDouble1 = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val d2arrayDouble2 = mk.ndarray(mk[mk[-1.0, -2.0], mk[-3.0, -4.0]])

        var doubleDiff = abs(5.477225575051661 - mk.linalg.norm(d2arrayDouble1))
        assertTrue(doubleDiff < 1e-8)
        assertEquals(6.0, mk.linalg.norm(d2arrayDouble1, Norm.N1))
        assertEquals(4.0, mk.linalg.norm(d2arrayDouble1, Norm.Max))
        assertEquals(7.0, mk.linalg.norm(d2arrayDouble1, Norm.Inf))
        doubleDiff = abs(5.477225575051661 - mk.linalg.norm(d2arrayDouble2))
        assertTrue(doubleDiff < 1e-8)
        assertEquals(6.0, mk.linalg.norm(d2arrayDouble2, Norm.N1))
        assertEquals(4.0, mk.linalg.norm(d2arrayDouble2, Norm.Max))
        assertEquals(7.0, mk.linalg.norm(d2arrayDouble2, Norm.Inf))

        val d2arrayFloat1 = mk.ndarray(mk[mk[1f, 2f], mk[3f, 4f]])
        val d2arrayFloat2 = mk.ndarray(mk[mk[-1f, -2f], mk[-3f, -4f]])

        var floatDiff = abs(5.477226f - mk.linalg.norm(d2arrayFloat1))
        assertTrue(floatDiff < 1e-6f)
        assertEquals(6f, mk.linalg.norm(d2arrayFloat1, Norm.N1))
        assertEquals(4f, mk.linalg.norm(d2arrayFloat1, Norm.Max))
        assertEquals(7f, mk.linalg.norm(d2arrayFloat1, Norm.Inf))
        floatDiff = abs(5.477226f - mk.linalg.norm(d2arrayFloat2))
        assertTrue(floatDiff < 1e-6f)
        assertEquals(6f, mk.linalg.norm(d2arrayFloat2, Norm.N1))
        assertEquals(4f, mk.linalg.norm(d2arrayFloat2, Norm.Max))
        assertEquals(7f, mk.linalg.norm(d2arrayFloat2, Norm.Inf))
    }

    @Test
    fun test_plu() {

        // Number case
        val procedurePrecision = 1e-5
        val rnd = Random(424242)

        val iters = 1000
        val sideFrom = 1
        val sideUntil = 100
        for (all in 0 until iters) {
            val m = rnd.nextInt(sideFrom, sideUntil)
            val n = rnd.nextInt(sideFrom, sideUntil)

            val a = mk.d2array(m, n) { rnd.nextDouble() }

            val (P, L, U) = KELinAlgEx.plu(a)
            assertTriangular(L, isLowerTriangular = true, requireUnitsOnDiagonal = true)
            assertTriangular(U, isLowerTriangular = false, requireUnitsOnDiagonal = false)
            assertClose(KELinAlg.dot(P, KELinAlg.dot(L, U)), a, procedurePrecision)
        }

        // Complex case
        //ComplexDouble
        for (all in 0 until iters) {
            val m = rnd.nextInt(sideFrom, sideUntil)
            val n = rnd.nextInt(sideFrom, sideUntil)

            val aRe = mk.d2array(m, n) { rnd.nextDouble() }
            val aIm = mk.d2array(m, n) { rnd.nextDouble() }
            val a = composeComplexDouble(aRe, aIm)

            val (P, L, U) = KELinAlgEx.pluC(a)
            assertTriangularComplexDouble(L, isLowerTriangular = true, requireUnitsOnDiagonal = true)
            assertTriangularComplexDouble(U, isLowerTriangular = false, requireUnitsOnDiagonal = false)
            assertCloseComplex(KELinAlgEx.dotMMComplex(P, KELinAlgEx.dotMMComplex(L, U)), a, procedurePrecision)
        }
        //ComplexFloat
        for (all in 0 until iters) {
            val m = rnd.nextInt(sideFrom, sideUntil)
            val n = rnd.nextInt(sideFrom, sideUntil)

            val aRe = mk.d2array(m, n) { rnd.nextFloat() }
            val aIm = mk.d2array(m, n) { rnd.nextFloat() }
            val a = composeComplexFloat(aRe, aIm)

            val (P, L, U) = KELinAlgEx.pluC(a)


            assertTriangularComplexFloat(L, isLowerTriangular = true, requireUnitsOnDiagonal = true)
            assertTriangularComplexFloat(U, isLowerTriangular = false, requireUnitsOnDiagonal = false)
            assertCloseComplex(KELinAlgEx.dotMMComplex(P, KELinAlgEx.dotMMComplex(L, U)), a, procedurePrecision)
        }

    }

    @Test
    fun solve_test() {
        // double case
        val procedurePrecision = 1e-5
        val rnd = Random(424242)

        // corner cases
        val a11 = mk.ndarray(mk[mk[4.0]])
        val b11 = mk.ndarray(mk[mk[3.0]])
        val b1 = mk.ndarray(mk[5.0])
        val b3 = mk.ndarray(mk[3.0, 4.0, 5.0])
        val b13 = mk.ndarray(mk[mk[3.0, 4.0, 5.0]])
        assertClose(solve(a11, b11), mk.ndarray(mk[mk[0.75]]), procedurePrecision)
        assertClose(solve(a11, b1), mk.ndarray(mk[1.25]), procedurePrecision)
        assertFailsWith<IllegalArgumentException>{solve(a11, b3)}
        assertClose(solve(a11, b13), mk.ndarray(mk[mk[0.75, 1.0, 1.25]]), procedurePrecision)

        for (iteration in 0 until 1000) {
            //test when second argument is d2 array
            val maxlen = 100
            val n = rnd.nextInt(1, maxlen)
            val m = rnd.nextInt(1, maxlen)
            val a = mk.d2array(n, n) { rnd.nextDouble() }
            val b = mk.d2array(n, m) { rnd.nextDouble() }
            assertClose(b, KELinAlg.dot(a, solve(a, b)), procedurePrecision)


            //test when second argument is d1 vector
            val bd1 = mk.d1array(n) { rnd.nextDouble() }
            val sol = solve(a, bd1)
            assertClose(KELinAlg.dot(a, sol.reshape(sol.shape[0], 1)).reshape(a.shape[0]), bd1, procedurePrecision)
        }

        // complexDouble case
        val c11 = mk.ndarray(mk[mk[ComplexDouble(4.0, 7.0)]])
        val d11 = mk.ndarray(mk[mk[ComplexDouble(3.0, 8.0)]])
        val d1 = mk.ndarray(mk[ComplexDouble(5.0, 9.0)])
        val d3 = mk.ndarray(mk[ComplexDouble(3.0, 11.0), ComplexDouble(4.0, 13.0), ComplexDouble(5.0, 17.0)])
        val d13 = mk.ndarray(mk[mk[ComplexDouble(3.0, 7.0), ComplexDouble(4.0, 8.0), ComplexDouble(5.0, 22.0)]])
        // d11 / c11
        assertCloseComplex(solveC(c11, d11), mk.ndarray(mk[mk[ComplexDouble(1.04615384615384, 0.1692307692307692)]]), procedurePrecision)
        // d1 / c11
        assertCloseComplex(solveC(c11, d1), mk.ndarray(mk[ComplexDouble(1.27692307692307, 0.01538461538461533)]), procedurePrecision)
        assertFailsWith<IllegalArgumentException>{solveC(c11, d3)}
        // d13 / c11
        assertCloseComplex(solveC(c11, d13), mk.ndarray(mk[mk[ComplexDouble(0.9384615384615385, 0.1076923076923077), ComplexDouble(1.1076923076923078, 0.06153846153846152), ComplexDouble(2.676923076923077, 0.8153846153846155)]]), procedurePrecision)

        for (iteration in 0 until 1000) {
            //test when second argument is d2 array
            val maxlen = 100
            val n = rnd.nextInt(1, maxlen)
            val m = rnd.nextInt(1, maxlen)

            val a = mk.zeros<ComplexDouble>(n, n)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    a[i, j] = ComplexDouble(rnd.nextDouble(), rnd.nextDouble())
                }
            }

            val bRe = mk.d2array(n, m) { rnd.nextDouble() }
            val bIm = mk.d2array(n, m) { rnd.nextDouble() }
            val b = composeComplexDouble(bRe, bIm)
            assertCloseComplex(b, KELinAlg.dot(a, solveC(a, b)), procedurePrecision)


            //test when second argument is d1 vector
            val bd1Re = mk.d1array(n) { rnd.nextDouble() }
            val bd1Im = mk.d1array(n) { rnd.nextDouble() }
            val bd1 = composeComplexDouble(bd1Re, bd1Im)

            val sol = solveC(a, bd1)
            assertCloseComplex(KELinAlg.dot(a, sol.reshape(sol.shape[0], 1)).reshape(a.shape[0]), bd1, procedurePrecision)
        }

        // complexFloat case
        val c11ComplexFloat = mk.ndarray(mk[mk[ComplexFloat(4.0, 7.0)]])
        val d1ComplexFloat1ComplexFloat = mk.ndarray(mk[mk[ComplexFloat(3.0, 8.0)]])
        val d1ComplexFloat = mk.ndarray(mk[ComplexFloat(5.0, 9.0)])
        val d3ComplexFloat = mk.ndarray(mk[ComplexFloat(3.0, 11.0), ComplexFloat(4.0, 13.0), ComplexFloat(5.0, 17.0)])
        val d1ComplexFloat3 = mk.ndarray(mk[mk[ComplexFloat(3.0, 7.0), ComplexFloat(4.0, 8.0), ComplexFloat(5.0, 22.0)]])
        // d1ComplexFloat1ComplexFloat / c11ComplexFloat
        assertCloseComplex(solveC(c11ComplexFloat, d1ComplexFloat1ComplexFloat), mk.ndarray(mk[mk[ComplexFloat(1.04615384615384, 0.1692307692307692)]]), procedurePrecision)
        // d1ComplexFloat / c11ComplexFloat
        assertCloseComplex(solveC(c11ComplexFloat, d1ComplexFloat), mk.ndarray(mk[ComplexFloat(1.27692307692307, 0.01538461538461533)]), procedurePrecision)
        assertFailsWith<IllegalArgumentException>{solveC(c11ComplexFloat, d3ComplexFloat)}
        // d1ComplexFloat3 / c11ComplexFloat
        assertCloseComplex(solveC(c11ComplexFloat, d1ComplexFloat3), mk.ndarray(mk[mk[ComplexFloat(0.9384615384615385, 0.1076923076923077), ComplexFloat(1.1076923076923078, 0.06153846153846152), ComplexFloat(2.676923076923077, 0.8153846153846155)]]), procedurePrecision)

        for (iteration in 0 until 1000) {
            //test when second argument is d2 array
            val maxlen = 100
            val n = rnd.nextInt(1, maxlen)
            val m = rnd.nextInt(1, maxlen)

            val a = mk.zeros<ComplexFloat>(n, n)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    a[i, j] = ComplexFloat(rnd.nextFloat(), rnd.nextFloat())
                }
            }

            val bRe = mk.d2array(n, m) { rnd.nextFloat() }
            val bIm = mk.d2array(n, m) { rnd.nextFloat() }
            val b = composeComplexFloat(bRe, bIm)
            assertCloseComplex(b, KELinAlg.dot(a, solveC(a, b)), 1e-2)


            //test when second argument is d1ComplexFloat vector
            val bd1ComplexFloatRe = mk.d1array(n) { rnd.nextFloat() }
            val bd1ComplexFloatIm = mk.d1array(n) { rnd.nextFloat() }
            val bd1ComplexFloat = composeComplexFloat(bd1ComplexFloatRe, bd1ComplexFloatIm)

            val sol = solveC(a, bd1ComplexFloat)
            assertCloseComplex(KELinAlg.dot(a, sol.reshape(sol.shape[0], 1)).reshape(a.shape[0]), bd1ComplexFloat, 1e-2)
        }


    }

    @Test
    fun testKEDot() {
        // random matrices pool
        val mat1 = mk.ndarray(mk[mk[4, -3, 2], mk[-6, -9, -7], mk[3, 6, 5]])
        val mat2 = mk.ndarray(mk[mk[-9, 4, -8], mk[-8, 2, 6], mk[3, 8, 7]])
        val mat3 = mk.ndarray(mk[mk[8, -2, -1], mk[7, -9, -1], mk[-9, -9, -2]])
        val mat4 = mk.ndarray(mk[mk[-8, 9, -10], mk[-6, 8, -9], mk[5, -5, 3]])
        val vec1 = mk.ndarray(mk[5, -1, 6])
        val vec2 = mk.ndarray(mk[5, -9, 1])
        val vec3 = mk.ndarray(mk[5, -10, 1])
        val vec4 = mk.ndarray(mk[9, -6, 3])

        // true operation results
        val mat1_x_mat1 = mk.ndarray(mk[mk[40, 27, 39], mk[9, 57, 16], mk[-9, -33, -11]])
        val mat1_x_mat2 = mk.ndarray(mk[mk[-6, 26, -36], mk[105, -98, -55], mk[-60, 64, 47]])
        val mat1_x_mat3 = mk.ndarray(mk[mk[-7, 1, -5], mk[-48, 156, 29], mk[21, -105, -19]])
        val mat1_x_mat4 = mk.ndarray(mk[mk[-4, 2, -7], mk[67, -91, 120], mk[-35, 50, -69]])

        val mat2_x_mat2 = mk.ndarray(mk[mk[25, -92, 40], mk[74, 20, 118], mk[-70, 84, 73]])
        val mat2_x_mat3 = mk.ndarray(mk[mk[28, 54, 21], mk[-104, -56, -6], mk[17, -141, -25]])
        val mat2_x_mat4 = mk.ndarray(mk[mk[8, -9, 30], mk[82, -86, 80], mk[-37, 56, -81]])

        val mat3_x_mat3 = mk.ndarray(mk[mk[59, 11, -4], mk[2, 76, 4], mk[-117, 117, 22]])
        val mat3_x_mat4 = mk.ndarray(mk[mk[-57, 61, -65], mk[-7, -4, 8], mk[116, -143, 165]])

        val mat1_x_vec1 = mk.ndarray(mk[35, -63, 39])
        val mat1_x_vec2 = mk.ndarray(mk[49, 44, -34])
        val mat2_x_vec1 = mk.ndarray(mk[-97, -6, 49])
        val mat2_x_vec2 = mk.ndarray(mk[-89, -52, -50])

        val vec1_x_vec1 = 62
        val vec1_x_vec2 = 40
        val vec1_x_vec3 = 41
        val vec1_x_vec4 = 69
        val vec2_x_vec2 = 107
        val vec2_x_vec3 = 116
        val vec2_x_vec4 = 102
        val vec3_x_vec3 = 126
        val vec3_x_vec4 = 108
        val vec4_x_vec4 = 126



        //Start test cases

        //Byte
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toByte() }, mat1.map { it.toByte() }).data, mat1_x_mat1.map { it.toByte() }.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toByte() }, mat2.map { it.toByte() }).data, mat1_x_mat2.map { it.toByte() }.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1.map { it.toByte() }, vec1.map { it.toByte() }).data, mat1_x_vec1.map { it.toByte() }.data)
        assertEquals(KELinAlgEx.dotVV(vec1.map { it.toByte() }, vec2.map { it.toByte() }), vec1_x_vec2.toByte())

        //Short
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toShort() }, mat1.map { it.toShort() }).data, mat1_x_mat1.map { it.toShort() }.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toShort() }, mat2.map { it.toShort() }).data, mat1_x_mat2.map { it.toShort() }.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1.map { it.toShort() }, vec1.map { it.toShort() }).data, mat1_x_vec1.map { it.toShort() }.data)
        assertEquals(KELinAlgEx.dotVV(vec1.map { it.toShort() }, vec2.map { it.toShort() }), vec1_x_vec2.toShort())

        //int
        assertContentEquals(KELinAlgEx.dotMM(mat1, mat1).data, mat1_x_mat1.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1, mat2).data, mat1_x_mat2.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1, vec1).data, mat1_x_vec1.data)
        assertEquals(KELinAlgEx.dotVV(vec1, vec2), vec1_x_vec2)

        //Long
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toLong() }, mat1.map { it.toLong() }).data, mat1_x_mat1.map { it.toLong() }.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toLong() }, mat2.map { it.toLong() }).data, mat1_x_mat2.map { it.toLong() }.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1.map { it.toLong() }, vec1.map { it.toLong() }).data, mat1_x_vec1.map { it.toLong() }.data)
        assertEquals(KELinAlgEx.dotVV(vec1.map { it.toLong() }, vec2.map { it.toLong() }), vec1_x_vec2.toLong())

        //Float
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toFloat() }, mat1.map { it.toFloat() }).data, mat1_x_mat1.map { it.toFloat() }.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toFloat() }, mat2.map { it.toFloat() }).data, mat1_x_mat2.map { it.toFloat() }.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1.map { it.toFloat() }, vec1.map { it.toFloat() }).data, mat1_x_vec1.map { it.toFloat() }.data)
        assertEquals(KELinAlgEx.dotVV(vec1.map { it.toFloat() }, vec2.map { it.toFloat() }), vec1_x_vec2.toFloat())

        //Double
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toDouble() }, mat1.map { it.toDouble() }).data, mat1_x_mat1.map { it.toDouble() }.data)
        assertContentEquals(KELinAlgEx.dotMM(mat1.map { it.toDouble() }, mat2.map { it.toDouble() }).data, mat1_x_mat2.map { it.toDouble() }.data)
        assertContentEquals(KELinAlgEx.dotMV(mat1.map { it.toDouble() }, vec1.map { it.toDouble() }).data, mat1_x_vec1.map { it.toDouble() }.data)
        assertEquals(KELinAlgEx.dotVV(vec1.map { it.toDouble() }, vec2.map { it.toDouble() }), vec1_x_vec2.toDouble())

        //ComplexDouble
        assertContentEquals(
            KELinAlgEx.dotMMComplex(composeComplexDouble(mat1, mat2), composeComplexDouble(mat3, mat4)).data,
            composeComplexDouble(mat1_x_mat3 - mat2_x_mat4, mat1_x_mat4 + mat2_x_mat3).data)

        assertContentEquals(
            KELinAlgEx.dotMVComplex(composeComplexDouble(mat1, mat2), composeComplexDouble(vec1, vec2)).data,
            composeComplexDouble(mat1_x_vec1 - mat2_x_vec2, mat1_x_vec2 + mat2_x_vec1).data
        )
        assertEquals(
            KELinAlgEx.dotVVComplex(composeComplexDouble(vec1, vec2), composeComplexDouble(vec3, vec4)),
            ComplexDouble(vec1_x_vec3 - vec2_x_vec4, vec1_x_vec4 + vec2_x_vec3)
        )

        //ComplexFloat
        assertContentEquals(
            KELinAlgEx.dotMMComplex(composeComplexFloat(mat1, mat2), composeComplexFloat(mat3, mat4)).data,
            composeComplexFloat(mat1_x_mat3 - mat2_x_mat4, mat1_x_mat4 + mat2_x_mat3).data)

        assertContentEquals(
            KELinAlgEx.dotMVComplex(composeComplexFloat(mat1, mat2), composeComplexFloat(vec1, vec2)).data,
            composeComplexFloat(mat1_x_vec1 - mat2_x_vec2, mat1_x_vec2 + mat2_x_vec1).data
        )
        assertEquals(
            KELinAlgEx.dotVVComplex(composeComplexFloat(vec1, vec2), composeComplexFloat(vec3, vec4)),
            ComplexFloat(vec1_x_vec3 - vec2_x_vec4, vec1_x_vec4 + vec2_x_vec3)
        )

    }

    @Test
    fun test_upper_hessenberg_form() {
        val n = 300
        val mat = getRandomMatrixComplexDouble(n, n)
        val (Q, H) = upperHessenbergDouble(mat.deepCopy())

        // check H is upper Hessenberg
        for (i in 2 until H.shape[0]) {
            for (j in 0 until i - 1) {
                assertEquals(H[i, j], ComplexDouble.zero)
            }
        }

        // assert Q is unitary
        val approxId = dotMatrixComplex(Q, Q.conjTranspose())
        val Id = mk.zeros<ComplexDouble>(n, n)

        assertCloseComplex(approxId, idComplexDouble(n), 1e-5)

        // assert decomposition is valid
        val approxmat = dotMatrixComplex(dotMatrixComplex(Q, H), Q.conjTranspose())
        assertCloseComplex(approxmat, mat, 1e-5)
    }


    @Test
    fun test_qr() {
        val n = 100
        val mat = getRandomMatrixComplexDouble(n, n)
        val (q, r) = qrComplexDouble(mat)

        // assert decomposition is valid
        assertCloseComplex(dotMatrixComplex(q, r), mat, 1e-5)

        // assert q is unitary
        assertCloseComplex(dotMatrixComplex(q, q.conjTranspose()), idComplexDouble(n), 1e-5)

        // assert r is upper triangular
        for (i in 1 until r.shape[0]) {
            for (j in 0 until i) {
                if(r[i, j] != ComplexDouble.zero) {
                    assertEquals(r[i, j], ComplexDouble.zero)
                }
            }
        }
    }

    @Test
    fun test_Schur_decomposition() {
        for (attempt in 0 until 5) {

            val n = when(attempt) {
                0 -> 1
                1 -> 2
                2 -> 5
                3 -> 10
                4 -> 100
                else -> 100
            }

            val mat = getRandomMatrixComplexDouble(n, n)
            val (q, r) = schurDecomposition(mat.deepCopy())

            // assert decomposition is valid
            assertCloseComplex(dotMatrixComplex(dotMatrixComplex(q, r), q.conjTranspose()), mat, 1e-5)

            // assert q is unitary
            assertCloseComplex(dotMatrixComplex(q, q.conjTranspose()), idComplexDouble(n), 1e-5)

            // assert r is upper triangular
            for (i in 1 until r.shape[0]) {
                for (j in 0 until i) {
                    if (r[i, j] != ComplexDouble.zero) {
                        assertEquals(r[i, j], ComplexDouble.zero)
                    }
                }
            }
        }
    }


    @Test
    fun test_eigenvalues() {
        val precision = 1e-2
        val n = 50
        val R = getRandomMatrixComplexDouble(n, n, -1000.0, 1000.0)
        for (i in 0 until R.shape[0]) {
            for (j in 0 until i) {
                R[i, j] = ComplexDouble.zero
            }
        }
        val Q = gramShmidtComplexDouble(getRandomMatrixComplexDouble(n, n))

        assertCloseComplex(dotMatrixComplex(Q, Q.conjTranspose()), idComplexDouble(n), precision = 1e-5)

        val mat = dotMatrixComplex(dotMatrixComplex(Q, R), Q.conjTranspose())

        var trueEigavals = List(n) { i -> R[i, i] }

        val eigs = KELinAlgEx.eigValsC(mat)

        var testedEigenvals = List(n) { i -> eigs[i] }

        trueEigavals = trueEigavals.sortedWith(compareBy({ it.re }, { it.im }))
        testedEigenvals = testedEigenvals.sortedWith(compareBy({ it.re }, { it.im }))



        for (i in 0 until n) {
            assertTrue("${trueEigavals[i]} =/= ${testedEigenvals[i]}") { (trueEigavals[i] - testedEigenvals[i]).abs() < precision}
        }

    }

    @Test
    fun compute_norm_for_vector() {
        val vector = mk.ndarray(mk[1.1, 0.0, 3.2, 2.3, 5.0])

        assertEquals(6.460650122085238, mk.linalg.norm(vector, Norm.Fro))
        assertEquals(11.600000000000001, mk.linalg.norm(vector, Norm.Inf))
        assertEquals(5.0, mk.linalg.norm(vector, Norm.N1))
        assertEquals(5.0, mk.linalg.norm(vector, Norm.Max))
    }
}



private fun <T : Complex, D : Dim2> assertCloseComplex(a: MultiArray<T, D>, b: MultiArray<T, D>, precision: Double) {
    assertEquals(a.dim.d, b.dim.d, "matrices have different dimensions")
    assertContentEquals(a.shape, b.shape, "matrices have different shapes")
    assertEquals(a.dtype, b.dtype)

    var maxabs = 0.0
    if (a.dim.d == 1) {
        when (a.dtype) {
            DataType.ComplexDoubleDataType -> {
                a as D1Array<ComplexDouble>
                b as D1Array<ComplexDouble>
                for (i in 0 until a.size) maxabs = max((a[i] - b[i]).abs(), maxabs)
            }
            DataType.ComplexFloatDataType -> {
                a as D1Array<ComplexFloat>
                b as D1Array<ComplexFloat>
                for (i in 0 until a.size) maxabs = max((a[i] - b[i]).abs().toDouble(), maxabs)
            }
            else -> {
                throw UnsupportedOperationException()
            }
        }
        assertTrue(maxabs < precision, "matrices not close")
        return
    }
    when (a.dtype) {
        DataType.ComplexDoubleDataType -> {
            a as D2Array<ComplexDouble>
            b as D2Array<ComplexDouble>
            for (i in 0 until a.shape[0]) {
                for (j in 0 until a.shape[1]) {
                    maxabs = max((a[i, j] - b[i, j]).abs(), maxabs)
                }
            }
        }
        DataType.ComplexFloatDataType -> {
            a as D2Array<ComplexFloat>
            b as D2Array<ComplexFloat>
            for (i in 0 until a.shape[0]) {
                for (j in 0 until a.shape[1]) {
                    maxabs = max((a[i, j] - b[i, j]).abs().toDouble(), maxabs)
                }
            }
        }
        else -> {
            throw UnsupportedOperationException()
        }
    }
    assertTrue(maxabs < precision, "matrices not close")
    return

}



private fun <T : Number, D : Dim2> assertClose(a: MultiArray<T, D>, b: MultiArray<T, D>, precision: Double) {
    assertEquals(a.dim.d, b.dim.d, "matrices have different dimensions")
    assertContentEquals(a.shape, b.shape, "matrices have different shapes")
    var maxabs = 0.0
    if (a.dim.d == 1) {
        a as D1Array<T>
        b as D1Array<T>
        for (i in 0 until a.size) maxabs = max(abs(a[i].toDouble() - b[i].toDouble()), maxabs)
    } else {
        a as D2Array<T>
        b as D2Array<T>
        for (i in 0 until a.shape[0]) {
            for (j in 0 until a.shape[1]) {
                maxabs = max(abs(a[i, j].toDouble() - b[i, j].toDouble()), maxabs)
            }
        }
    }
    assertTrue(maxabs < precision, "matrices not close")
}

private fun <T : Number> assertTriangular(a: MultiArray<T, D2>, isLowerTriangular: Boolean, requireUnitsOnDiagonal: Boolean) {
    if (requireUnitsOnDiagonal) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            if (a[i, i].toDouble() != 1.0)  throw AssertionError("element at position [$i, $i] of matrix \n$a\n is not unit")
        }
    }
    if (isLowerTriangular) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in i + 1 until a.shape[1]) {
                if(a[i, j].toDouble() != 0.0) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    } else {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in 0 until i) {
                if(a[i, j].toDouble() != 0.0) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    }
}

private fun assertTriangularComplexDouble(a: MultiArray<ComplexDouble, D2>, isLowerTriangular: Boolean, requireUnitsOnDiagonal: Boolean) {
    if (requireUnitsOnDiagonal) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            if (a[i, i] != ComplexDouble.one)  throw AssertionError("element at position [$i, $i] of matrix \n$a\n is not unit")
        }
    }
    if (isLowerTriangular) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in i + 1 until a.shape[1]) {
                if(a[i, j] != ComplexDouble.zero) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    } else {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in 0 until i) {
                if(a[i, j] != ComplexDouble.zero) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    }
}

private fun assertTriangularComplexFloat(a: MultiArray<ComplexFloat, D2>, isLowerTriangular: Boolean, requireUnitsOnDiagonal: Boolean) {
    if (requireUnitsOnDiagonal) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            if (a[i, i] != ComplexFloat.one)  throw AssertionError("element at position [$i, $i] of matrix \n$a\n is not unit")
        }
    }
    if (isLowerTriangular) {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in i + 1 until a.shape[1]) {
                if(a[i, j] != ComplexFloat.zero) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    } else {
        for (i in 0 until min(a.shape[0], a.shape[1])) {
            for (j in 0 until i) {
                if(a[i, j] != ComplexFloat.zero) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
            }
        }
    }
}



private fun getRandomMatrixComplexDouble(n: Int, m: Int, from: Double = 0.0, to: Double = 1.0, rnd: Random = Random(424242)): D2Array<ComplexDouble> {
    val a = mk.zeros<ComplexDouble>(n, m)

    for (i in 0 until n) {
        for (j in 0 until m) {
            a[i, j] = ComplexDouble(rnd.nextDouble()* (to - from) + from, rnd.nextDouble()* (to - from) + from)
        }
    }
    return a
}

private fun idComplexDouble(n: Int): D2Array<ComplexDouble> {
    val ans = mk.zeros<ComplexDouble>(n, n)
    for (i in 0 until n) {
        ans[i, i] = 1.0.toComplexDouble()
    }
    return ans
}


private fun<T : Number, D: Dim2> composeComplexDouble(rePart: NDArray<T, D>, imPart: NDArray<T, D>): NDArray<ComplexDouble, D> {
    if (rePart.dim.d == 1) {
        rePart as D1Array<T>
        imPart as D1Array<T>
        val ans = mk.zeros<ComplexDouble>(rePart.shape[0])
        for (i in 0 until ans.shape[0]) {
            ans[i] = ComplexDouble(rePart[i].toDouble(), imPart[i].toDouble())
        }
        return ans as NDArray<ComplexDouble, D>
    }

    rePart as D2Array<T>
    imPart as D2Array<T>
    val ans = mk.zeros<ComplexDouble>(rePart.shape[0], rePart.shape[1])
    for (i in 0 until ans.shape[0]) {
        for (j in 0 until ans.shape[1]) {
            ans[i, j] = ComplexDouble(rePart[i, j].toDouble(), imPart[i, j].toDouble())
        }
    }
    return ans as NDArray<ComplexDouble, D>
}

private fun<T : Number, D: Dim2> composeComplexFloat(rePart: NDArray<T, D>, imPart: NDArray<T, D>): NDArray<ComplexFloat, D> {
    if (rePart.dim.d == 1) {
        rePart as D1Array<T>
        imPart as D1Array<T>
        val ans = mk.zeros<ComplexFloat>(rePart.shape[0])
        for (i in 0 until ans.shape[0]) {
            ans[i] = ComplexFloat(rePart[i].toFloat(), imPart[i].toFloat())
        }
        return ans as NDArray<ComplexFloat, D>
    }

    rePart as D2Array<T>
    imPart as D2Array<T>
    val ans = mk.zeros<ComplexFloat>(rePart.shape[0], rePart.shape[1])
    for (i in 0 until ans.shape[0]) {
        for (j in 0 until ans.shape[1]) {
            ans[i, j] = ComplexFloat(rePart[i, j].toFloat(), imPart[i, j].toFloat())
        }
    }
    return ans as NDArray<ComplexFloat, D>
}