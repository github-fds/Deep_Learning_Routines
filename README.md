# DLR: Deep Learning Routine
'***DLR (Deep Learning Routines)***' as a part of DPU (Deep Learning Processing
Unit) is a collection of high-level synthesizable C/C++ routines for deep learning
inference network.

All contents are provided as it is WITHOUT ANY WARRANTY and NO TECHNICAL SUPPORT
will be provided for problems that might arise.

## Table of contents

<details><summary>Click to expand table of contents</summary>

1. [Overview](#overview)<br>
   1.1 [Note](#note)<br>
   1.2 [Prerequisites](#prerequisites)<br>
   1.3 [Conventions](#conventions)<br>
   1.4 [How to build](#build)<br>
   1.5 [How to use](#use)
2. [Deep-Learning Routines](#DLR)<br>
   2.1 [Activation](#activation)<br>
   2.1.1. [Hyperbolic tangent](#activation_tanh)<br>
   2.1.2. [Leaky ReLU](#activation_leaky_relu)<br>
   2.1.3. [ReLU](#activation_relu)<br>
   2.1.4. [Sigmoid](#activation_sigmoid)<br>
   2.2 [Concatenation](#concatenation)<br>
   2.2.1 [Concat2d](#concatenation_concat2d)<br>
   2.3 [Convolution](#convolution)<br>
   2.3.1 [Convolution2d](#convolution_convolution2d)<br>
   2.4 [Deconvolution](#deconvolution)<br>
   2.4.1 [Deconvolution2d](#deconvolution_deconvolution2d)<br>
   2.5 [Linear (Fully connected)](#linear)<br>
   2.6 [Normalization](#normalization)<br>
   2.6.1 [Batch normalization](#normalization_batch_normalization)<br>
   2.7 [Pooling](#pooling)<br>
   2.7.1 [Average pooling](#pooling_average)<br>
   2.7.2 [Max pooling](#pooling_max)
3. [Projects](#project2)<br>
   3.1 [Project: LeNet-5](#project_lenet5)<br>
   3.2 [Project: Tiny YOLO-V2 for VOC](#project_lenet5)
4. [Other things](#other_things)<br>
   4.1 [Acknowledgment](#acknowledgment)<br>
   4.2 [Authors and contributors](#authors_and_contributors)<br>
   4.3 [License](#license)<br>
   4.4 [Revision history](#revision_history)

</details>

---
# 1. Overview<a name="overview"></a>
'DLR (Deep Learning Routines)' as a part of DPU (Deep Learning Processing
Unit) is a collection of high-level synthesizable C/C++ routines for deep learning
inference network.

As shown in the picture below, DLR routines are verified in the context of C/C++, Python, PyTorch, and FPGA.

| ![Overall framework](doc/images/dlr_framework-simple.png) |
|:---:|
| *Overall framework* |

DLR routines contain followings (some are under development and more will be added):
* Convolution layer (strict convolution )
* Activation layer (ReLU, Leaky ReLU, hyperbolic tangent, sigmoid)
* Pooling layer (max, average)
* Fully connected layer (linear layer)
* De-convolution layer (transposed convolution layer)
* Batch normalization layer
* Concatenation layer

The routines have following highlights:
* Fully synthesizable C code
* Highly parameterized to be adopted wide range of usages
* C, Python, PyTorch, Verilog test-benches
* FPGA verified using Future Design Systems’ CON-FMC

### 1.1 Note<a name="note"></a>
It should be noted that the routines are not optimized to get a higher
performance since the routines are for hardware implementation not for
computation. In addition to this the routines are only for inference not for training.

> Version 1.4 uses 'dlr' namespace.

### 1.2 Prerequisites<a name="prerequisites"></a>
This program requires followings.
* GNU GCC: C compiler
* PyTorch / Conda / Python
* (Optional for hardware verification and implementation)
  * HDL simulator: Xilinx Xsim
  * High level synthesis: Xilinx Vivado HLS

### 1.3 Conventions<a name="conventions"></a>
This program uses following conventions.
* DLR routines are defined within ***dlr*** namespace for C++.
* Each DLR routine has C++ and C routines, where templates are used for C++.
* Python wrappers follow C++ name.
* PyTorch wrappers follow PyTorch *torch.nn.functional* name.

### 1.4 How to build<a name="build"></a>
Simple do as follows and '*include*' and '*lib*' directories will be ready to use under 'v1.4' directory.

```sh
$ cd v1.4/src
$ make clean
$ make
$ make install
```

### 1.5 How to use<a name="use"></a>
You should specify where the DLR header files resiede using "-I" option for g++ compiler,
in which $(DLR_HOME) is the path to the deep-learning routines such as /home/user/Deep_Learning_Routines/v1.4.
> $ g++ .... -I$(DLR_HOME)/include ....

You should specify where the DLR libaray resides using "-L" and "-l" options for g++ linker.
> $ g++ .... -L$(DLR_HOME)/lib -ldlr ....

Refer to the 
<a href="https://github.com/adki/DLR_Projects" target="_blank">DLR Projects</a>.
for more details.

---
## 2. Deep-Learning Routines<a name="DLR"></a>

### 2.1 Activation<a name="activation"></a>

<details><summary>Click to expand this section</summary>

#### 2.1.1 Hyperbolic tangent<a name="activation_tanh"></a>

#### 2.1.2 Leaky ReLU<a name="activation_leaky_relu"></a>

#### 2.1.3 ReLU<a name="activation_relu"></a>

#### 2.1. 4 Sigmoid<a name="activation_sigmoid"></a>

</details>

### 2.2 Concatenation<a name="concatenation"></a>

<details><summary>Click to expand this section</summary>

#### 2.2.1 Concat2d<a name="concatenation_concat2d"></a>

| ![2D concatenation](doc/images/concat.png) |
|:---:|
| *Concatenation* |

</details>

### 2.3 Convolution<a name="convolution"></a>

<details><summary>Click to expand this section</summary>

#### 2.3.1 Convolution2d<a name="convolution_convolution2d"></a>
'Convolution2d()' applies a 2D convolution over an input data composed of multiple planes (i.e., channels).

| ![2D convolution](doc/images/convolution2d.png) |
|:---:|
| *COnvolutioin* |

#### 1.1. C++ routine<a name="convolution2d_cpp"></a>
C++ routines uses template to support user specific data type.

<details><summary>Click to expand C++ code</summary>

```cpp
template<class TYPE=float>
void Convolution2d
(           TYPE     *out_data    // out_channel x out_size x out_size
    , const TYPE     *in_data     // in_channel x in_size x in_size
    , const TYPE     *kernel      // out_channel x in_channel x kernel_size x kernel_size
    , const TYPE     *bias        // out_channel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding=0
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0   // check rigorously when 1
    , const int       verbose=0 // verbose level
    #endif
)
```

</details>

#### 1.2. C routine<a name="convolution2d_c"></a>
C routines are build by overloading template of corresponding C++ routine.

<details><summary>Click to expand C code</summary>

```c
extern void Convolution2dInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const int      *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const int      *bias        // bias per kernel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // number of biases, it should be the same as out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride      // stride default 1
    , const uint8_t   padding     // padding default 0
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);
extern void Convolution2dDouble( ); // arguments are the same as Concat2dFloat, but data type is 'double' instead of 'float'
extern void Convolution2dInt( ); // arguments are the same as Concat2dFloat, but data type is 'double' instead of 'int'
```

</details>

#### 1.3. Python wrapper<a name="convolution2d_python"></a>
‘Convolution2d()’ wrapper gets NumPy arguments for array and ‘out_data’
carries calculated result. It returns ‘True’ on success or ‘False’ on failure.

<details><summary>Click to expand Python code</summary>

```python
Convolution2d( out_data    # out_channel x out_size x out_size
             , in_data     # in_channel x in_size x in_size
             , kernel      # out_channel x in_channel x kernel_size x kernel_size
             , bias=None   # out_channel
             , stride=1
             , padding=0
             , rigor=False
             , verbose=False)
```

</details>

#### 1.4. PyTorch wrapper<a name="convolution2d_pytorch"></a>
'conv2d()' PyTorch wrapper gets PyTorch tensor arguments for array and
returns calculated result. It calls Python wrapper after converting PyTorch tensor
to NumPy array.

<details><summary>Click to expand PyTorch code</summary>

```python
conv2d( input     # in_minibatch x in_channel x in_size x in_size
      , weight    # out_channel  x in_channel x kernel_size x kernel_size
      , bias=None # out_channel
      , stride=1
      , padding=0
      , dilation=1
      , groups=1
      , rigor=False
      , verbose=False)
```

</details>

</details>

### 2.4 Deconvolution<a name="deconvolution"></a>

<details><summary>Click to expand this section</summary>

#### 2.4.1 Deconvolution2d<a name="deconvolution_deconvolution2d"></a>

| ![2D deconvolution](doc/images/deconvolution.png) |
|:---:|
| *Deconvolution (Transposed convolution)* |

</details>

### 2.5 Linear (Fully connected)<a name="linear"></a>
Linear transfomation on the input data

y = x * W <sup>T</sup> + b

<details><summary>Click to expand this section</summary>

| ![Linear](doc/images/linear1d.png) |
|:---:|
| *Linear one-dimensional* |

| ![LinearNd](doc/images/linearNd.png) |
|:---:|
| *Linear N-dimensional* |

</details>

### 2.6 Normalization<a name="normalization"></a>

<details><summary>Click to expand this section</summary>

#### 2.6.1 Batch normalization<a name="normalization_batch_normalization"></a>

| ![Batch normalization equation](doc/images/batchnorm-equation.png) |
|:---:|
| *Batch normaization equation* |
| E[x]: mean, Var[x]: variance, &gamma;: scaling factor, &beta;: bias, &epsilon;: numerical stability |
| ![Batch normalization](doc/images/batchnorm.png) |
|:---:|
| *Batch normaization* |

</details>

### 2.7 Pooling<a name="pooling"></a>

<details><summary>Click to expand this section</summary>

#### 2.7.1 Average pooling<a name="pooling_average"></a>

#### 2.7.2 Max pooling<a name="pooling_max"></a>

</details>

---
## 9. Projects<a name="projects"></a>

### 9.1 Project: LeNet-5 for MNIST<a name="project_lenet5"></a>
LeNet-5 is a popular convolutional
neural network architecture for handwritten and
machine-printed character recognition.

Details of this project can be found from 
<a href="https://github.com/adki/DLR_Projects" target="_blank">DLR Projects: LeNet-5</a>.

| ![LeNet-5 project on FPGA](doc/images/lenet5-on-fpga.png) |
|:---:|
| *LeNet-5 project on FPGA* |

<details><summary>Click to expand this project</summary>

### 1. LeNet-5 network
Following picture shows network structure and its data size.

| ![LeNet-5 network](doc/images/lenet5-network.png) |
|:---:|
| *LeNet-5 network* |

Following table shows a summary of the network and its parameters.

| ![LeNet-5 network details](doc/images/lenet5-network-details.png) |
|:---:|
| *LeNet-5 network details* |

### 2. Design flow
The picture below shows an overall design flow from model training to FPGA implementation.

| ![LeNet-5 design flow](doc/images/lenet5-design-flow-all.png) |
|:---:|
| *LeNet-5 design flow* |

</details>

### 9.2 Project: Tiny YOLO-V2 for VOC<a name="project_lenet5"></a>
YOLO is a popular convolutional
neural network architecture for object detection.

Details of this project can be found from 
<a href="https://github.com/adki/DLR_Projects" target="_blank">DLR Projects: Tiny YOLO-V2</a>.

| ![Tiny YOLO-V2 project on FPGA](doc/images/yolov2tiny-on-fpga.png) |
|:---:|
| *Tiny YOLO-V2 project on FPGA* |

<details><summary>Click to expand this project</summary>

</details>

---
### 4. Other things <a name="other_things"></a>

#### 4.1 Acknowledgment<a name="acknowledgment"></a>
This work was supported partially by KARI (Korea Aerospace Research
Institute) under the “Development of deep learning hardware accelerator for
microsatellite ” (Contract 2019110A35C-00).<br>
Some part of this work was carried out partially by Handong Global University
2020 Summer Internship program.<br>
Some part of this work was supported partially by ETRI (Electronics and
Telecommunications Research Institute) under the “Development and training of
FPGA-based AI semiconductor design environment ” (Contract EA20202206).

#### 4.2 Authors and contributors<a name="authors_and_contributors"></a>
* **[Ando Ki]** - *Initial work* - <a href="http://www.future-ds.com" target="_blank">Future Design Systems</a>
* **Seongwon Seo** - <a href="http://www.future-ds.com" target="_blank">Future Design Systems</a>
* **Kwangsub Jeon** - <a href="http://www.future-ds.com" target="_blank">Future Design Systems</a>
* **Chae Eon Lim** - <a href="http://www.future-ds.com" target="_blank">Future Design Systems</a>

#### 4.3 License<a name="license"></a>
DLR (Deep Learning Routines) and its associated materials are licensed with
the 2-clause BSD license to make the program and library useful in open and
closed source projects independent of their licensing scheme.
Each contributor holds copyright over their respective contribution.

<details><summary>The 2-Clause BSD License</summary>
Copyright 2020-2021 Future Design Systems (http:://www.future-ds.com)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
</details>

#### 4.4 Revision history<a name="revision_history"></a>
* 2021.09.04: v1.4 uses namespace 'dlr'
* 2020.11.12: Released v1.3
* 2020.03.10: Started by Ando Ki (adki at future-ds.com)
