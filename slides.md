---
title: ROCm with ROCr, right?
author: Peter Steinbach
origin: Scionics Computer Innovation GmbH
email: steinbach@scionics.de
date: March 30, 2017
---


# Before I start


## Scionics Who?

[columns,class="row vertical-align"]

[column,class="col-xs-6"]

![](img/scionics_main_logo.png)  
[Scionics Computer Innovation GmbH](scionics.de)

[/column]

[column,class="col-xs-6"]

- founded in 2000, Dresden (Germany)
- service provider to the [Max Planck Institute of Molecular Cell Biology and Genetics](mpi-cbg.de)  

    - scientific computing facility
    - IT infrastructure
    - public relations

[/column]


[/columns]



## Why parallel2017?


[columns,class="row"]

[column,class="col-xs-4"]

<center>
Nvidia Tesla  
![](img/nvidia_tesla_p100_cropped_x400.png)
</center>

[/column]

[column,class="col-xs-4"]

<center>
AMD FirePro  
![](img/FirePro_S9300x2_x400.png)
</center>

[/column]

[column,class="col-xs-4"]

<center>
Intel MIC  
![](img/xeon_phi_x400.jpg)
</center>

[/column]

[/columns]


<center>

**What should our clients choose?**

</center>


## Why I present?

<center>

<video width="1400" poster="video/Celegans_lateral_one_view_versus_deconvolved.png" controls loop>
<source src="video/Celegans_lateral_one_view_versus_deconvolved.webm" type='video/webm; codecs="vp8.0, vorbis"'>
<source src="video/Celegans_lateral_one_view_versus_deconvolved.mp4" type='video/mp4'>
<p>Movie does not work! Sorry!</p>
</video>


*Accelerating [our clients' scientific algorithms](http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html) on [GPUs](https://github.com/psteinb/gtc2015.git)  
(multi-GB dataset, a lot of FFTs)*

</center>


## This Talk != Advertisement 

<center>
**AMD provided test hardware and that's it!**
</center>


- Our company is by no means financially tied to AMD nor any of it's resellers.

- whatever I find missing or not working, I'll report it here
(use the [issue tracker](https://github.com/psteinb/parallel2017/issues) of this talk to correct me)



## This Talk is


<center>
![](img/opensource-550x475.png)  

**[github.com/psteinb/parallel2017](https://github.com/psteinb/parallel2017)**
</center>


## Outline

<div style="font-size : 1.5em">

<center>
1. ROCm

2. Porting Code from CUDA

3. HC
</center>

</div>


# ROCm

## Radeon Open Compute Platform 

[columns,class="row"]

[column,class="col-xs-4"]

[![](img/rocm_logo_x400.png)](http://gpuopen.com/compute-product/rocm/)  

[/column]

[column,class="col-xs-8"]

[/column]

- very young:  
*April 25th, 2016*, version 1.0
- 3 main components:

    - [ROCm](http://gpuopen.com/compute-product/rocm/) Linux kernel driver 
    - [ROCr](https://github.com/RadeonOpenCompute/ROCR-Runtime) runtime & library stack
    - [HCC](https://github.com/RadeonOpenCompute/hcc) compiler based on LLVM  

<center>
**Open Source!**
</center>

[/columns]

## ROCm kernel driver

[columns,class="row"]

[column,class="col-xs-4"]

[![](img/amd-gcn-2.0-hawaii_x400.png)](http://www.amd.com/en-us/innovations/software-technologies/gcn)  

[/column]

[column,class="col-xs-8"]

[/column]

<center>

- large memory single allocation   
(>32GB in one pointer)
- peer-to-peer Multi-GPU
- peer-to-peer with RDMA
- systems management API and tooling

</center>

[/columns]


## ROCr runtime

[columns,class="row-fluid"]

[column,class="col-xs-4"]

<center>
[![](img/hsa_logo_x400.jpg)](http://www.hsafoundation.com/)  
</center>

[/column]


[column,class="col-xs-8"]

[/column]

- AMD's implementation of HSA runtime  
(+ extensions for multi-GPU)
- user mode queues
- flat memory addressing
- atomic memory transactions & signals
- process concurrency & preemption
- device discovery

[/columns]


## Heterogenous Compute Compiler

[columns,class="row"]

[column,class="col-xs-4"]

<center>
[![](img/llvm-dragon_x400.png)](http://www.hsafoundation.com/)  
</center>

[/column]

[column,class="col-xs-8"]

[/column]

- [hcc](https://github.com/RadeonOpenCompute/hcc) compiler for supported APIs
- LLVM native GCN ISA code generation
- offline compilation support
- standardized loader and code object format
- GCN ISA assembler and disassembler
- HIP, HC and OpenCL programming interface

[/columns]


# Porting Code from CUDA


## Prologue

<center>

![ [UoB-HPC/GPU-STREAM](https://github.com/UoB-HPC/GPU-STREAM) ](img/github_gpu_stream.png){ width=90% }

</center>

## [UoB-HPC/GPU-STREAM](https://github.com/UoB-HPC/GPU-STREAM)

```
/* add   */ c[:]    = a[:]
/* mul   */ b[:]    = scalar*b[:]
/* copy  */ c[:]    = a[:] + b[:]
/* triad */ a[:]    = b[:] + scalar*c[:] 
/* dot   */ scalar  = dot(a[:],b[:])
```
  

<center>

- benchmark of various programming paradigms:  
  OpenMP3, OpenMP4, CUDA, Kokkos, Raja, OpenCL, ...
- for now *nix only

</center>

## Hipify

[columns, class="row"]

[column,class="col-xs-4"]

<center>

![](fig/hipify-pipeline.svg){ width=65% }

</center>

[/column]


[column,class="col-xs-8"]

- Convert CUDA to portable C++, `hipify`
- C++ kernel language ( C++11/14/17 features )
- C runtime API
- same performance as native CUDA

. . . 

&nbsp;

- supports *most commonly* used parts of CUDA:  
  streams, events, memory (de-)allocation, profiling

- produced apps have full tool support:
    - CUDA: nvcc, nvprof, nvvp
    - ROCM: hcc, rocm-prof, codexl

[/column]
  
[/columns]


## [CUDA Example](https://github.com/UoB-HPC/GPU-STREAM/blob/master/CUDAStream.cu#L149)


```
__global__ void add_kernel(const T * a, 
                           const T * b, 
                           T * c){
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];}

void CUDAStream<T>::add(){
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();  //..
  }
```

## [Hip`ified Example](https://github.com/UoB-HPC/GPU-STREAM/blob/master/HIPStream.cpp#L152)

```
__global__ void add_kernel(hipLaunchParm lp, 
                           const T * a, 
                           const T * b, 
                           T * c){
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i] + b[i];
}

void HIPStream<T>::add(){
  hipLaunchKernel(HIP_KERNEL_NAME(add_kernel), 
                  dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, 
                  d_a, d_b, d_c);  check_error();  //...
}
```

## HIP summary

- very interesting tool to get started with production or legacy code

- still low-level CUDA programming

- HIP evosystem available as well: hipBlas, hipFFT, hipRNG, MIOpen (machine learning library)


# Heterogenous Compute

## HC

- C++ parallel runtime and API
- superset of C++AMP
- adds device specific instrinsics (wavefront shuffle, bit extraction, atomics)
- compiled with `hcc`
- very similar to [thrust](http://thrust.github.io/), [boost.compute](https://github.com/boostorg/com), [sycl](https://www.khronos.org/sycl)
- current C++17 STL implementatipn wraps around this


## HC API

<center>
![](fig/hc_api_nutshell.svg){ width=80% }
</center>
