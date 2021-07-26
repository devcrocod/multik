package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

internal fun invDouble(a: MultiArray<Double, D2>): D2Array<Double> {
    requireSquare(a)
    return solveDouble(a, mk.identity(a.shape[0]))
}

internal fun invFloat(a: MultiArray<Float, D2>): D2Array<Float> {
    requireSquare(a)
    return solveFloat(a, mk.identity(a.shape[0]))
}