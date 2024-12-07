[![Kotlin Alpha](https://kotl.in/badges/alpha.svg)](https://kotlinlang.org/docs/components-stability.html)
[![JetBrains incubator project](https://jb.gg/badges/incubator.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Maven Central](https://img.shields.io/maven-central/v/org.jetbrains.kotlinx/multik-core)](https://mvnrepository.com/artifact/org.jetbrains.kotlinx/multik-core)
[![GitHub license](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

# Multik

Multidimensional array library for Kotlin.

## Modules
* `multik-core` &mdash; contains ndarrays, methods called on them and [math], [stat] and [linalg] interfaces.
* `multik-default` &mdash; implementation including `multik-kotlin` and `multik-openblas` for performance.
* `multik-kotlin` &mdash; implementation of [math], [stat] and [linalg] interfaces on JVM.
* `multik-openblas` &mdash; implementation of [math], [stat] and [linalg] interfaces in native code using OpenBLAS.

## Using in your projects
### Gradle
In your Gradle build script:
1. Add the Maven Central Repository.
2. Add the `org.jetbrains.kotlinx:multik-core:$multik_version` api dependency.
3. Add an implementation dependency: `org.jetbrains.kotlinx:multik-default:$multik_version`,
`org.jetbrains.kotlinx:multik-kotlin:$multik_version` or `org.jetbrains.kotlinx:multik-openblas:$multik_version`.

`build.gradle`:
```groovy
repositories {
    mavenCentral()
}

dependencies {
    implementation "org.jetbrains.kotlinx:multik-core:0.2.3"
    implementation "org.jetbrains.kotlinx:multik-default:0.2.3"
}
```

`build.gradle.kts`:
```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
}
```

For a multiplatform project, set the dependency in a common block:

```kotlin
kotlin {
    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
            }
        }
    }
}
```

or in a platform-specific block:

```kotlin
kotlin {
    sourceSets {
        val jvmName by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:multik-core-jvm:0.2.3")
            }
        }
    }
}
```

### Jupyter Notebook
Install [Kotlin kernel](https://github.com/Kotlin/kotlin-jupyter) for
[Jupyter](https://jupyter.org/)
or just visit to [Datalore](https://datalore.jetbrains.com/).

Import stable `multik` version into notebook:
```
%use multik
```

## Support platforms

|       Platforms       |   `multik-core`    |  `multik-kotlin`   |                                                                              `multik-openblas`                                                                              |                       `multik-default`                       |
|:---------------------:|:------------------:|:------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
|        **JS**         | :white_check_mark: | :white_check_mark: |                                                                                     :x:                                                                                     |                      :white_check_mark:                      |
|     **linuxX64**      | :white_check_mark: | :white_check_mark: |                                                                             :white_check_mark:                                                                              |                      :white_check_mark:                      |
|     **mingwX64**      | :white_check_mark: | :white_check_mark: |                                                                             :white_check_mark:                                                                              |                      :white_check_mark:                      |
|     **macosX64**      | :white_check_mark: | :white_check_mark: |                                                                             :white_check_mark:                                                                              |                      :white_check_mark:                      |
|    **macosArm64**     | :white_check_mark: | :white_check_mark: |                                                                             :white_check_mark:                                                                              |                      :white_check_mark:                      |
|     **iosArm64**      | :white_check_mark: | :white_check_mark: |                                                                                     :x:                                                                                     |                      :white_check_mark:                      |
|      **iosX64**       | :white_check_mark: | :white_check_mark: |                                                                                     :x:                                                                                     |                      :white_check_mark:                      |
| **iosSimulatorArm64** | :white_check_mark: | :white_check_mark: |                                                                                     :x:                                                                                     |                      :white_check_mark:                      |
|        **JVM**        | :white_check_mark: | :white_check_mark: | linuxX64 - :white_check_mark:<br/>mingwX64 - :white_check_mark:<br/>macosX64 - :white_check_mark:<br/>macosArm64 - :white_check_mark:<br/>androidArm64 - :white_check_mark: | androidArm32 - :x:<br/>androidX86 - :x:<br/>androidX64 - :x: |

For Kotlin/JS, we use the new [IR](https://kotlinlang.org/docs/js-ir-compiler.html).
We also use the [new memory model](https://blog.jetbrains.com/kotlin/2021/08/try-the-new-kotlin-native-memory-manager-development-preview/)
in Kotlin/Native. Keep this in mind when using Multik in your multiplatform projects.

**Note**:
* on ubuntu 18.04 and older `multik-openblas` doesn't work due to older versions of _**glibc**_.
* `multik-openblas` for desktop targets (_linuxX64_, _mingwX64_, _macosX64_, _macosArm64_) is experimental and unstable.
We will improve stability and perfomance as _Kotlin/Native_ evolves.
* JVM target `multik-openblas` for Android only supports **arm64-v8a** processors.

## Quickstart

Visit [Multik documentation](https://kotlin.github.io/multik) for a detailed feature overview.

#### Creating arrays

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
/* [1, 2, 3] */

val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
/*
[[1.5, 2.1, 3.0],
[4.0, 5.0, 6.0]]
*/

val c = mk.ndarray(mk[mk[mk[1.5f, 2f, 3f], mk[4f, 5f, 6f]], mk[mk[3f, 2f, 1f], mk[4f, 5f, 6f]]])
/*
[[[1.5, 2.0, 3.0],
[4.0, 5.0, 6.0]],

[[3.0, 2.0, 1.0],
[4.0, 5.0, 6.0]]]
*/


mk.zeros<Double>(3, 4) // create an array of zeros
/*
[[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]
*/
mk.ndarray<Float, D2>(setOf(30f, 2f, 13f, 12f), intArrayOf(2, 2)) // create an array from a collection
/*
[[30.0, 2.0],
[13.0, 12.0]]
*/
val d = mk.ndarray(doubleArrayOf(1.0, 1.3, 3.0, 4.0, 9.5, 5.0), 2, 3) // create an array of shape(2, 3) from a primitive array
/*
[[1.0, 1.3, 3.0],
[4.0, 9.5, 5.0]]
*/
mk.d3array(2, 2, 3) { it * it } // create an array of 3 dimension
/*
[[[0, 1, 4],
[9, 16, 25]],

[[36, 49, 64],
[81, 100, 121]]]
*/

mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
/*
[[0.0+(0.0)i, 0.0+(1.0)i, 0.0+(2.0)i],
[1.0+(0.0)i, 1.0+(1.0)i, 1.0+(2.0)i],
[2.0+(0.0)i, 2.0+(1.0)i, 2.0+(2.0)i]]
 */

mk.arange<Long>(10, 25, 5) // creare an array with elements in the interval [10, 25) with step 5
/* [10, 15, 20] */

mk.linspace<Double>(0, 2, 9) // create an array of 9 elements in the interval [0, 2]
/* [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] */

val e = mk.identity<Double>(3) // create an identity array of shape (3, 3)
/*
[[1.0, 0.0, 0.0],
[0.0, 1.0, 0.0],
[0.0, 0.0, 1.0]]
*/

val diag = mk.diagonal(mk[2, 4, 8]) // create a diagonal array
/*
[[2, 0, 0],
[0, 4, 0],
[0, 0, 8]]
 */
```

#### Array properties
```kotlin
a.shape // Array dimensions
a.size // Size of array
a.dim // object Dimension
a.dim.d // number of array dimensions
a.dtype // Data type of array elements
```

#### Arithmetic operations
```kotlin
val f = b - d // subtraction
/*
[[0.5, 0.8, 0.0],
[0.0, -4.5, 1.0]]
*/

d + f // addition
/*
[[1.5, 2.1, 3.0],
[4.0, 5.0, 6.0]]
*/

b / d // division
/*
[[1.5, 1.6153846153846154, 1.0],
[1.0, 0.5263157894736842, 1.2]]
*/

f * d // multiplication
/*
[[0.5, 1.04, 0.0],
[0.0, -42.75, 5.0]]
*/
```

#### Array mathematics

See documentation for other methods of
[mathematics](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.math/index.html),
[linear algebra](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.linalg/index.html),
[statistics](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.stat/index.html).

```kotlin
a.sin() // element-wise sin, equivalent to mk.math.sin(a)
a.cos() // element-wise cos, equivalent to mk.math.cos(a)
b.log() // element-wise natural logarithm, equivalent to mk.math.log(b)
b.exp() // element-wise exp, equivalent to mk.math.exp(b)
d dot e // dot product, equivalent to mk.linalg.dot(d, e)
```

#### Aggregate functions
```kotlin
mk.math.sum(c) // array-wise sum
mk.math.min(c) // array-wise minimum elements
mk.math.maxD3(c, axis=0) // maximum value of an array along axis 0
mk.math.cumSum(b, axis=1) // cumulative sum of the elements
mk.stat.mean(a) // mean
mk.stat.median(b) // meadian
```

#### Copying arrays
```kotlin
val f = a.copy() // create a copy of the array and its data
val h = b.deepCopy() // create a copy of the array and copy the meaningful data
```

#### Operations of Iterable
```kotlin
c.filter { it < 3 } // select all elements less than 3
b.map { (it * it).toInt() } // return squares
c.groupNDArrayBy { it % 2 } // group elements by condition
c.sorted() // sort elements
```

#### Indexing/Slicing/Iterating
```kotlin
a[2] // select the element at the 2 index
b[1, 2] // select the element at row 1 column 2
b[1] // select row 1 
b[0..1, 1] // select elements at rows 0 to 1 in column 1
b[0, 0..2..1] // select elements at row 0 in columns 0 to 2 with step 1

for (el in b) {
    print("$el, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0, 
}

// for n-dimensional
val q = b.asDNArray()
for (index in q.multiIndices) {
    print("${q[index]}, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0, 
}
```

#### Inplace

```kotlin
val a = mk.linspace<Float>(0, 1, 10)
/*
a = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 
0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0]
*/
val b = mk.linspace<Float>(8, 9, 10)
/*
b = [8.0, 8.11111111111111, 8.222222222222221, 8.333333333333334, 8.444444444444445, 8.555555555555555,
8.666666666666666, 8.777777777777779, 8.88888888888889, 9.0]
*/

a.inplace { 
    math { 
        (this - b) * b
         abs()
    }
}
// a = [64.0, 64.88888, 65.77778, 66.66666, 67.55556, 68.44444, 69.333336, 70.22222, 71.111115, 72.0]
```

## Building
To build the entire project, you need to set up an environment for building `multik-openblas`:
* JDK 1.8 or higher
* _JAVA_HOME_ environment - to search for jni files
* Compilers _gcc_, _g++_, _gfortran_ version 8 or higher.
It is important that they are of the same version.

Run `./gradlew assemble` to build all modules.
If you don't need to build `multik-openblas`,
just disable the `cmake_build` task and build the module you need.

## Contributing
There is an opportunity to contribute to the project:
1. Implement [math](multik-core/src/main/kotlin/org/jetbrains/kotlinx/multik/api/math/Math.kt),
[linalg](multik-core/src/main/kotlin/org/jetbrains/kotlinx/multik/api/linalg/LinAlg.kt),
[stat](multik-core/src/main/kotlin/org/jetbrains/kotlinx/multik/api/Statistics.kt) interfaces.
2. Create your own engine successor from [Engine](multik-core/src/main/kotlin/org/jetbrains/kotlinx/multik/api/Engine.kt), for example - [JvmEngine](multik-kotlin/src/main/kotlin/org/jetbrains/kotlinx/multik/jvm/JvmEngine.kt).
3. Use [mk.addEngine](https://github.com/devcrocod/multik/blob/972b18cfd2952abd811fabf34461d238e55c5587/multik-core/src/main/kotlin/org/jetbrains/multik/api/Multik.kt#L23) and [mk.setEngine](https://github.com/devcrocod/multik/blob/972b18cfd2952abd811fabf34461d238e55c5587/multik-core/src/main/kotlin/org/jetbrains/multik/api/Multik.kt#L27)
to use your implementation.
