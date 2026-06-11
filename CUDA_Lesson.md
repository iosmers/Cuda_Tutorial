# 海外 GPU 体系结构与并行计算课程索引

> 最后更新：2026-06-11  
> 目标：整理美国和其他海外大学中与 **GPU 体系结构、CUDA/GPGPU、并行计算、异构计算、高性能计算** 相关的课程/公开课/研究计算中心短训，方便系统学习 GPU 硬件与编程。  
> 筛选原则：优先选择大学官方课程页、院系课程目录、研究计算中心培训页；部分课程不是纯 CUDA 课，但包含 GPU 架构、CUDA、OpenCL、OpenACC、HIP、并行性能优化等核心内容。

---

## 1. 怎么读这份清单？

课程大致分成四类：

| 类型 | 适合你学什么 | 代表课程 |
|------|--------------|----------|
| **GPU/CUDA 专门课** | CUDA 编程模型、GPU memory hierarchy、kernel 优化、profiling | Caltech CS179、UIUC ECE408、Northwestern COMP_SCI 368/468、Oxford CUDA course |
| **并行计算系统课** | SIMD、多核、GPU、分布式、性能建模，建立完整并行计算视角 | Stanford CS149、CMU 15-418、Berkeley CS267、ANU COMP4300 |
| **GPU 架构课** | SM/CU、warp/wavefront、scheduler、cache、memory controller、GPU 编译器 | Georgia Tech CS7295、UCR EE/CS 217、Heidelberg GPU Computing |
| **HPC/科研计算短训** | 面向科研代码迁移到 GPU，通常有 hands-on lab | Oxford、Cambridge、Sheffield、Cornell CVW、TAMU HPRC、Toronto SciNet |

自学时建议不要按学校名盲目刷。更实用的顺序是：

```text
CUDA 入门
  -> GPU memory hierarchy 和性能分析
  -> 并行算法模式：reduction / scan / stencil / histogram / GEMM
  -> GPU 架构：SM / warp / cache / memory controller / Tensor Core
  -> 多 GPU、异构计算、HPC 应用
```

---

## 2. 美国大学与研究计算中心课程

| # | 学校/机构 | 课程/资源 | 类型 | 重点内容 | 适合阶段 | 链接 |
|---|-----------|-----------|------|----------|----------|------|
| 1 | California Institute of Technology | CS 179: GPU Programming | GPU/CUDA 专门课 | CUDA 编程、GPU 架构、并行算法、性能优化、项目 | 入门到进阶 | [课程页](https://courses.cms.caltech.edu/cs179/) |
| 2 | Stanford University | CS149: Parallel Computing | 并行计算系统课 | 并行硬件/软件、GPU architecture and CUDA programming、data-parallel thinking | 入门到系统化 | [课程页](https://cs149.stanford.edu/) |
| 3 | Carnegie Mellon University | 15-418/15-618: Parallel Computer Architecture and Programming | 并行计算系统课 | 多核、GPU、CUDA、并行编程模型、性能优化 | 系统化进阶 | [课程页](https://www.cs.cmu.edu/~418/) |
| 4 | University of Illinois Urbana-Champaign | ECE 408 / CS 483 / CSE 408: Applied Parallel Programming | GPU/CUDA 专门课 | CUDA、并行算法模式、GPU memory、CNN/GEMM/scan/stencil 等应用 | 入门到进阶 | [官方目录](https://ece.illinois.edu/academics/courses/ece408-120181) / [公开课程站](https://lumetta.web.engr.illinois.edu/408-Sum24/) |
| 5 | Georgia Institute of Technology | CS 7295: GPU Hardware and Software | GPU 架构与软件 | CUDA、GPU 架构、优化、编译器、硬件论文阅读 | 进阶 | [OMSCS 课程页](https://omscs.gatech.edu/cs-7295-gpu-hardware-and-software) |
| 6 | Northwestern University | COMP_SCI 368/468: Programming Massively Parallel Processors with CUDA | GPU/CUDA 专门课 | CUDA、GPU 上的软件开发与优化、massively parallel processors | 入门到进阶 | [课程描述](https://www.mccormick.northwestern.edu/computer-science/academics/courses/descriptions/368-468.html) |
| 7 | University of California, Berkeley | CS C267: Applications of Parallel Computers | HPC/并行计算课 | 并行算法、GPU、云平台、MPI/OpenMP、科学计算应用 | 系统化 | [课程目录](https://www2.eecs.berkeley.edu/Courses/CSC267/) |
| 8 | Johns Hopkins University | 605.617: Introduction to GPU Programming | GPU 编程课 | CUDA、OpenCL、GPU 编程基础、数据分析/搜索等并行任务 | 入门 | [课程页](https://ep.jhu.edu/courses/605617-introduction-to-gpu-programming/) |
| 9 | University of California, Riverside | EE/CS 217: GPU Architecture and Parallel Programming | GPU 架构与并行编程 | CUDA、GPU memory/threading model、OpenCL、数据并行模式 | 入门到进阶 | [课程页](https://vsclab.ece.ucr.edu/courses/2013/08/02/eecs-217-winter-2014-gpu-architecture-and-parallel-programming) |
| 10 | Stony Brook University | CSE 392/591: GPU Programming | GPU 编程课 | 并行编程基础、GPU 架构、CUDA、Programming Massively Parallel Processors | 入门 | [课程页](https://www3.cs.stonybrook.edu/~mueller/teaching/cse591_GPU/) |
| 11 | University of Florida | CIS 6930: GPU Parallel Architecture and Programming | GPU 架构与编程 | CUDA threads/block/grid、CUDA memory、OpenCL、Fermi 架构、warp scheduling | 进阶 | [课程大纲](https://www.cise.ufl.edu/~peir/CIS6930sp12/syllabus.html) |
| 12 | University of Georgia | CUDA C Programming on GPUs for High Performance Computing | GPU/CUDA 课程 | CUDA C、GPU architecture、threads、performance issues、floating point | 入门 | [课程目录](https://bulletin.uga.edu/Course/Details/26532) |
| 13 | Purdue University | CGT 62000: Graphics Processing Unit Computing | GPU 计算课 | GPU architecture、CUDA programming model、OpenCL programming model | 入门到进阶 | [课程目录](https://catalog.purdue.edu/preview_course_nopop.php?catoid=10&coid=104250) |
| 14 | Binghamton University | CS 580J: GPU Architecture & CUDA Programming | GPU 架构与 CUDA | GPU architecture、CUDA fundamentals、HPC on parallel hardware | 入门到进阶 | [课程目录](https://catalog.binghamton.edu/preview_course_nopop.php?catoid=5&coid=65265) |
| 15 | Milwaukee School of Engineering | CSC 5241: GPU Programming | GPU 编程课 | CUDA model/libraries、profiling、optimization、GPU architecture | 入门到进阶 | [课程目录](https://catalog.msoe.edu/preview_course.php?catoid=41&coid=44804) |
| 16 | University of Illinois Chicago | MCS 572: Introduction to Supercomputing | HPC 课程 | MPI/OpenMP、GPU、CUDA、Tensor Cores、PyCUDA/Julia CUDA 等 | 入门到系统化 | [课程页](https://homepages.math.uic.edu/~jan/mcs572/index.html) |
| 17 | Cornell University | Cornell Virtual Workshop: Understanding GPU Architecture | 公开训练 | GPU 架构、CPU/GPU 对比、GPGPU 程序构造、NVIDIA GPU memory/compute components | 入门 | [CVW 路线](https://cvw.cac.cornell.edu/gpu-architecture) |
| 18 | Texas A&M University | HPRC GPU Programming | HPC 短训 | CUDA fundamentals、GPU architecture、kernel、memory management、性能优化 | 入门 | [培训页](https://hprc.tamu.edu/training/intro_cuda.html) |
| 19 | University of Texas at Austin / Oden Institute | CUDA Programming on NVIDIA GPUs | 密集短课 | CUDA hands-on、GPU 应用开发、面向研究人员和研究生 | 入门到进阶 | [Oden 新闻页](https://oden.utexas.edu/news-and-events/news/cuda-nvidia-gpu-programming-course/) / [2026 课程页](https://people.maths.ox.ac.uk/~gilesm/cuda/index_Oden26.html) |
| 20 | University of Illinois Urbana-Champaign | Heterogeneous Parallel Programming | MOOC/异构并行课 | CUDA/OpenCL、OpenACC、MPI、GPU-based heterogeneous systems | 入门到系统化 | [Wen-mei Hwu 页面](https://ece.illinois.edu/about/directory/emeritus/w-hwu) |
| 21 | University of Illinois Urbana-Champaign | Introduction to Parallel Programming with CUDA | CUDA 短训 | CUDA parallel programming、parallelism forms、hardware limits、efficient data structures | 入门 | [活动页](https://newfrontiers.illinois.edu/news-and-events/introduction-to-parallel-programming-with-cuda/) |

---

## 3. 美国以外大学与研究计算中心课程

| # | 学校/机构 | 国家/地区 | 课程/资源 | 类型 | 重点内容 | 适合阶段 | 链接 |
|---|-----------|-----------|-----------|------|----------|----------|------|
| 22 | University of Oxford / Oxford e-Research Centre | 英国 | CUDA Programming on NVIDIA GPUs | GPU/CUDA hands-on | CUDA 编程、GPU 应用开发、lectures + practicals | 入门到进阶 | [OeRC 页面](https://oerc.ox.ac.uk/education/cuda-programming-on-nvidia-gpus) / [Mike Giles 课程页](https://people.maths.ox.ac.uk/~gilesm/cuda/) |
| 23 | University of Cambridge | 英国 | High Performance Computing: Programming GPU using CUDA | HPC 短训 | CUDA 语言、GPU programming 入门 | 入门 | [培训页](https://www.training.cam.ac.uk/ucs/course/ucs-hpc-gpu-cuda) |
| 24 | University of Sheffield | 英国 | COM4521/COM6521: Parallel Computing with Graphical Processing Units | GPU/CUDA 模块 | NVIDIA CUDA、GPU hardware-aware optimization、并行计算 | 入门到进阶 | [公开教学页](https://rse.shef.ac.uk/training/com4521/) |
| 25 | University of Birmingham | 英国 | NVIDIA Fundamentals of Accelerated Computing with Modern CUDA C++ | GPU/CUDA workshop | CUDA C++、core libraries、memory migration、GPU-accelerated algorithms | 入门 | [培训页](https://www.birmingham.ac.uk/research/arc/bear/training/nvidia-fundamentals-of-accelerated-computing-with-modern-cuda-c) |
| 26 | ARCHER2 / EPCC training ecosystem | 英国 | GPU Programming with CUDA | HPC 短训 | CPU/GPU 架构差异、kernel execution、memory management、shared memory、性能问题 | 入门 | [课程页](https://www.archer2.ac.uk/training/courses/220627-gpu-cuda/) |
| 27 | Australian National University | 澳大利亚 | COMP4300/8300: Parallel Systems | 并行系统课 | GPU architecture、CUDA programming/execution model、memory hierarchy、streams | 系统化 | [资源页](https://comp.anu.edu.au/courses/comp4300/resources/) / [GPU 讲义](https://comp.anu.edu.au/courses/comp4300/assets/lectures/Lecture_GPUs_partI_4up.pdf) |
| 28 | University of Toronto / SciNet | 加拿大 | HPC133: Introduction to GPU Programming / Programming GPUs with CUDA | HPC 短训 | GPU 科学计算、CUDA/框架介绍、hands-on examples | 入门 | [Intro GPU](https://education.scinet.utoronto.ca/course/view.php?id=1266) / [CUDA workshop](https://education.scinet.utoronto.ca/course/view.php?id=1013) |
| 29 | Technical University of Munich | 德国 | Practical Course: GPU Programming in Computer Vision | 应用型 GPU/CUDA | NVIDIA CUDA、并行化基础 CV 算法、CUDA/C++ project | 入门到进阶 | [课程页](https://cvg.cit.tum.de/teaching/ss2018/gpucourse_ss2018) |
| 30 | Saarland University | 德国 | GPU Programming | GPU/CUDA 课程 | CUDA、parallel hardware architectures、GPU efficient algorithms、项目 | 入门到进阶 | [课程页](https://graphics.cg.uni-saarland.de/courses/gpu-2020/index.html) |
| 31 | University of Mannheim | 德国 | GPU Programming | GPU 编程课 | GPU programming、课程作业/练习、英文授课 | 入门到进阶 | [课程页](https://www.wim.uni-mannheim.de/moerkotte/teaching/courses/gpu-programming/) |
| 32 | Heidelberg University | 德国 | GPU Computing: Architecture and Programming | GPU 架构与编程 | GPU internal architecture、CUDA、shared memory optimization、multi-GPU、advanced architecture | 进阶 | [课程页](https://hawaii.ziti.uni-heidelberg.de/teaching/gpu/) |
| 33 | University of Freiburg | 德国 | GPU Programming Course | GPU/CUDA 应用课 | CUDA framework、parallel GPU programming、computer vision algorithms | 入门 | [课程页](https://lmb.informatik.uni-freiburg.de/lectures/gpu_course/) |
| 34 | Heidelberg University / ARI | 德国 | Introduction to GPU Accelerated Computing | GPU/CUDA 入门 | CUDA C、数值加速计算、GPGPU examples | 入门 | [课程页](https://wwwstaff.ari.uni-heidelberg.de/spurzem/lehre/WS24/cuda/cuda.php.en) |
| 35 | ETH Zurich | 瑞士 | Solving PDEs in Parallel on GPUs with Julia | GPU 科学计算课 | GPU 架构、CUDA.jl、Julia GPU、PDE 并行求解 | 入门到进阶 | [课程主页](https://pde-on-gpu.vaw.ethz.ch/) |
| 36 | ETH Zurich | 瑞士 | Heterogeneous Systems Seminar | 异构系统研讨课 | GPUs、FPGAs、ASICs、heterogeneous memory/systems、论文研讨 | 进阶 | [课程页](https://systems.ethz.ch/education/courses/2022-spring/heterogeneous-systems-seminar.html) |
| 37 | EPFL | 瑞士 | GPUs: Introduction to CUDA / Architecture and Programming lectures | GPU 架构讲义 | GPU architecture、parallelism model、CUDA programming、memory allocation/synchronization | 入门 | [Introduction to CUDA](https://graphsearch.epfl.ch/en/lecture/0_mwg0po2n) / [Architecture lecture](https://graphsearch.epfl.ch/en/lecture/0_uiolg26t) |
| 38 | University of Hong Kong | 中国香港 | SDST4013 / Applied HPC and Parallel Programming | HPC 课程 | MPI、OpenMP、CUDA programming、GPU acceleration | 入门到系统化 | [SDST4013](https://saasweb.hku.hk/courses/202526/sdst4013.html) / [APAI4013](https://webapp.science.hku.hk/sr4/servlet/enquiry?Type=Course&course_code=APAI4013) |
| 39 | Nanyang Technological University | 新加坡 | Graduate course info: parallel computing topics | HPC/并行课 | multithreaded programming、GPU computing、C++ threads、OpenMP、CUDA、MPI | 入门到系统化 | [课程信息页](https://www.ntu.edu.sg/spms/about-us/mathematics/grad/course-info) |
| 40 | National University of Singapore | 新加坡 | Solving Problems with Thousands of CPUs / GPU workshop | GPU/CUDA workshop | GPU architecture、CUDA programming model、NVIDIA GPU examples | 入门 | [Workshop 页面](https://www.comp.nus.edu.sg/~sws/2018/Solving%20Problems%20with%20Thousands%20of%20-CPUs-2018.htm) |
| 41 | Johannes Gutenberg University Mainz | 德国 | Accelerated Computing with GPUs | GPU 加速计算课 | GPU accelerated computing、理论基础、应用和编程技术 | 入门到进阶 | [教学页](https://www.hpc.informatik.uni-mainz.de/teaching/) |
| 42 | Paderborn University / HPC.NRW | 德国 | GPU Computing at HPC.NRW | HPC 短训 | CUDA programming、GPU code tuning、HPC system practice | 入门到进阶 | [活动页](https://events.uni-paderborn.de/e/gpu2022) |

---

## 4. 按学习目标选课

### 4.1 想从零开始学 CUDA

优先看：

1. [Caltech CS179](https://courses.cms.caltech.edu/cs179/)
2. [Oxford CUDA Programming on NVIDIA GPUs](https://people.maths.ox.ac.uk/~gilesm/cuda/)
3. [Cornell Virtual Workshop: Understanding GPU Architecture](https://cvw.cac.cornell.edu/gpu-architecture)
4. [Texas A&M HPRC GPU Programming](https://hprc.tamu.edu/training/intro_cuda.html)
5. [University of Toronto SciNet HPC133](https://education.scinet.utoronto.ca/course/view.php?id=1266)

这些课程/短训的共同点是：不会假设你已经懂 GPU，通常从 CPU/GPU 差异、kernel launch、memory copy、thread/block/grid 开始。

### 4.2 想系统理解并行计算，不只学 CUDA 语法

优先看：

1. [Stanford CS149](https://cs149.stanford.edu/)
2. [CMU 15-418/15-618](https://www.cs.cmu.edu/~418/)
3. [UC Berkeley CS267](https://www2.eecs.berkeley.edu/Courses/CSC267/)
4. [ANU COMP4300/8300](https://comp.anu.edu.au/courses/comp4300/resources/)
5. [UIUC ECE408/CS483](https://ece.illinois.edu/academics/courses/ece408-120181)

这些课程适合把 CUDA 放进更大的背景里：SIMD、shared memory、多核、MPI、OpenMP、分布式系统、性能建模。

### 4.3 想深入 GPU 架构和性能优化

优先看：

1. [Georgia Tech CS7295: GPU Hardware and Software](https://omscs.gatech.edu/cs-7295-gpu-hardware-and-software)
2. [UCR EE/CS 217](https://vsclab.ece.ucr.edu/courses/2013/08/02/eecs-217-winter-2014-gpu-architecture-and-parallel-programming)
3. [Heidelberg GPU Computing: Architecture and Programming](https://hawaii.ziti.uni-heidelberg.de/teaching/gpu/)
4. [Northwestern COMP_SCI 368/468](https://www.mccormick.northwestern.edu/computer-science/academics/courses/descriptions/368-468.html)
5. [University of Florida CIS 6930](https://www.cise.ufl.edu/~peir/CIS6930sp12/syllabus.html)

重点关注：

- warp / wavefront 调度
- shared memory bank conflict
- occupancy 与 register pressure
- L1/L2/HBM 层级
- memory coalescing
- Tensor Core / matrix instruction
- profiling 与 roofline 分析

### 4.4 想做科学计算/HPC 的 GPU 迁移

优先看：

1. [Oxford CUDA course](https://oerc.ox.ac.uk/education/cuda-programming-on-nvidia-gpus)
2. [ARCHER2 GPU Programming with CUDA](https://www.archer2.ac.uk/training/courses/220627-gpu-cuda/)
3. [ETH Solving PDEs in Parallel on GPUs with Julia](https://pde-on-gpu.vaw.ethz.ch/)
4. [UC Berkeley CS267](https://www2.eecs.berkeley.edu/Courses/CSC267/)
5. [UIC MCS 572](https://homepages.math.uic.edu/~jan/mcs572/index.html)

这些更贴近真实科研代码：PDE、stencil、linear algebra、MPI+GPU、集群环境、profiling 和性能迁移。

---

## 5. 推荐自学路线

如果目标是学习 GPU 硬件和 CUDA 编程，建议用下面的顺序组合课程：

### 路线 A：CUDA 工程入门

```text
Cornell GPU Architecture
  -> Caltech CS179
  -> UIUC ECE408
  -> Northwestern COMP_SCI 368/468
  -> 自己实现 reduction / scan / matmul / convolution
```

适合目标：能独立写 CUDA kernel，并能做基本性能优化。

### 路线 B：并行计算系统路线

```text
Stanford CS149
  -> CMU 15-418
  -> UC Berkeley CS267
  -> Georgia Tech CS7295
```

适合目标：不只会 CUDA，还理解 CPU 多核、GPU、分布式、编译器和硬件权衡。

### 路线 C：GPU 架构深入路线

```text
Cornell GPU Architecture
  -> UCR EE/CS 217
  -> Georgia Tech CS7295
  -> Heidelberg GPU Computing
  -> 阅读 NVIDIA/AMD 架构白皮书和 Nsight Compute 指标
```

适合目标：能从硬件角度解释 kernel 为什么快/慢。

### 路线 D：科研/HPC 应用路线

```text
Oxford CUDA course
  -> ARCHER2 GPU Programming with CUDA
  -> ETH PDEs on GPUs with Julia
  -> Berkeley CS267
  -> 在自己的 PDE / stencil / linear algebra 代码里做 GPU porting
```

适合目标：把已有 CPU 科学计算代码迁移到 GPU/集群。

---

## 6. 选课时重点看什么？

| 判断项 | 为什么重要 |
|--------|------------|
| 是否有作业/实验 | GPU 编程必须写代码；只看讲义很难建立性能直觉 |
| 是否讲 memory hierarchy | register/shared/L1/L2/HBM 是 CUDA 优化核心 |
| 是否讲 profiling | 没有 Nsight Compute/Nsight Systems 或类似工具，优化容易靠猜 |
| 是否覆盖并行算法模式 | reduction、scan、stencil、histogram、GEMM 是 CUDA 基础套路 |
| 是否覆盖多 GPU/通信 | 深度学习训练和 HPC 都离不开 NCCL、MPI、NVLink/InfiniBand |
| 是否讲硬件架构 | 想深入性能必须理解 warp scheduler、occupancy、coalescing、cache |
| 是否有公开材料 | 自学优先选择 lecture slides、assignments、recordings 公开的课程 |

---

## 7. 我的优先推荐

如果只挑 8 门/套资源，优先顺序如下：

1. **Caltech CS179**：CUDA/GPU 入门非常直接，适合动手。
2. **Stanford CS149**：并行计算系统视角强，讲 GPU 但不局限于 GPU。
3. **CMU 15-418/15-618**：和 Stanford CS149 类似，适合建立系统观。
4. **UIUC ECE408/CS483**：Programming Massively Parallel Processors 风格，CUDA 算法模式扎实。
5. **Georgia Tech CS7295**：GPU hardware + software，适合深入架构。
6. **Oxford CUDA Programming on NVIDIA GPUs**：面向科研人员的 CUDA hands-on，很实用。
7. **Cornell Virtual Workshop GPU Architecture**：短小，适合作为硬件术语预习。
8. **Heidelberg GPU Computing: Architecture and Programming**：课程标题和内容都非常贴近“GPU 体系结构 + CUDA 编程”。

---

## 8. 和本仓库已有笔记的对应关系

| 本仓库主题 | 建议配套课程 |
|------------|--------------|
| CUDA 入门、grid/block/thread | Caltech CS179、Oxford CUDA course、TAMU HPRC |
| memory hierarchy、shared memory、bank conflict | UIUC ECE408、Northwestern COMP_SCI 368/468、Cornell CVW |
| coalesced access、profiling、性能优化 | Georgia Tech CS7295、Heidelberg GPU Computing、UCR EE/CS 217 |
| GPU 硬件拆解、SM/warp/HBM | Cornell CVW、Georgia Tech CS7295、Stanford CS149 |
| HPC/科学计算 GPU 迁移 | Berkeley CS267、ARCHER2、ETH PDE on GPUs、Oxford CUDA course |
| 多 GPU 和集群 | Berkeley CS267、ETH Heterogeneous Systems、ANU COMP4300 |

---

## 9. YouTube 视频/播放列表资源

> 这一节收集 YouTube 上讲 CUDA、GPU 架构、GPU 并行编程、Nsight 性能分析、Triton/GPU kernel 的视频和播放列表。  
> 类型里标注了 **播放列表** 或 **单视频**。自学时优先看播放列表；遇到具体问题时再补单视频。

### 9.1 优先推荐的 YouTube 资源

| # | 频道/博主 | 视频/播放列表 | 类型 | 适合看什么 | 链接 |
|---|-----------|---------------|------|------------|------|
| 1 | NVIDIA Developer | CUDA Trainings & Updates | 播放列表 | CUDA 官方培训、工具链、新特性 | [YouTube](https://www.youtube.com/playlist?list=PL5B692fm6--vScfBaxgY89IRWFzDt0Khm) |
| 2 | NVIDIA Developer | Boost CUDA Development with Nsight Developer Tools | 播放列表 | Nsight Compute/System 性能分析 | [YouTube](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR) |
| 3 | NVIDIA Developer | Getting Started with CUDA and Parallel Programming | 单视频 | CUDA 官方入门、并行编程概念 | [YouTube](https://www.youtube.com/watch?v=GmNkYayuaA4) |
| 4 | NVIDIA Developer | Coding on NVIDIA GPUs with CUDA C | 单视频 | CUDA C 编码流程 | [YouTube](https://www.youtube.com/watch?v=JKHmXbCH7vE) |
| 5 | NVIDIA Developer | Accelerating Applications with Parallel Algorithms | 单视频 | 并行算法和 CUDA C++ | [YouTube](https://www.youtube.com/watch?v=Sdjn9FOkhnA) |
| 6 | NVIDIA Developer | Implementing New Algorithm with CUDA Kernels | 单视频 | 自定义 CUDA kernel 设计 | [YouTube](https://www.youtube.com/watch?v=kTWoGCSugB4) |
| 7 | NVIDIA Developer | Asynchrony and CUDA Streams | 单视频 | CUDA streams、异步执行 | [YouTube](https://www.youtube.com/watch?v=pyW9St8uM8w) |
| 8 | NVIDIA Developer | Understanding NVIDIA GPU Hardware as a CUDA C Programmer | 单视频 | 从 CUDA C 程序员视角看 NVIDIA GPU 硬件 | [YouTube](https://www.youtube.com/watch?v=1Goq8Yc3dfo) |
| 9 | NVIDIA Developer | Deep Dive: How to Use cuTile Python | 单视频 | cuTile、tile 编程模型 | [YouTube](https://www.youtube.com/watch?v=YFrP03KuMZ8) |
| 10 | NVIDIA Developer | Intro to NVIDIA Nsight Compute | 单视频 | Nsight Compute 入门 | [YouTube](https://www.youtube.com/watch?v=Iuy_RAvguBM) |
| 11 | NVIDIA Developer | Intro to NVIDIA Nsight Systems | 单视频 | Nsight Systems 时间线分析 | [YouTube](https://www.youtube.com/watch?v=dUDGO66IadU) |
| 12 | NVIDIA Developer | SOL Analysis with NVIDIA Nsight Compute | 单视频 | Speed of Light 分析 | [YouTube](https://www.youtube.com/watch?v=uHN5fpfu8As) |
| 13 | NVIDIA Developer | Memory Analysis with NVIDIA Nsight Compute | 单视频 | 显存、cache、访存性能分析 | [YouTube](https://www.youtube.com/watch?v=GCkdiHk6fUY) |
| 14 | NVIDIA Developer | Guided Analysis with Nsight Compute | 单视频 | 用 Nsight Compute 定位瓶颈 | [YouTube](https://www.youtube.com/watch?v=04dJ-aePYpE) |
| 15 | NVIDIA Developer | CUDA Tutorials: CUDA Compatibility | 单视频 | CUDA 版本兼容、驱动/toolkit 关系 | [YouTube](https://www.youtube.com/watch?v=H3AQnlpxk0c) |
| 16 | GTC / Stephen Jones | How CUDA Programming Works | 单视频 | CUDA 编程模型底层机制 | [YouTube](https://www.youtube.com/watch?v=QQceTDjA4f4) |
| 17 | Creel | CUDA Tutorials | 播放列表 | 经典 CUDA 入门系列 | [YouTube](https://www.youtube.com/playlist?list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK) |
| 18 | Creel | NVIDIA CUDA Tutorial 1: Introduction | 单视频 | CUDA 基本概念 | [YouTube](https://www.youtube.com/watch?v=m0nhePeHwFs) |
| 19 | Creel | NVIDIA CUDA Tutorial 5: Memory Overview | 单视频 | CUDA memory overview | [YouTube](https://www.youtube.com/watch?v=RY2_8wB2QY4) |
| 20 | Creel | NVIDIA CUDA Tutorial 8: Intro to Shared Memory | 单视频 | shared memory 入门 | [YouTube](https://www.youtube.com/watch?v=upGoZ00MlfI) |
| 21 | Creel | NVIDIA CUDA Tutorial 9: Bank Conflicts | 单视频 | shared memory bank conflict | [YouTube](https://www.youtube.com/watch?v=CZgM3DEBplE) |
| 22 | Creel | NVIDIA CUDA Tutorial 10: Blocking with Shared Memory | 单视频 | shared memory blocking/tiling | [YouTube](https://www.youtube.com/watch?v=oJK5CDlkwCs) |
| 23 | Udacity | Intro to Parallel Programming | 播放列表 | CS344 CUDA/GPU 并行编程完整视频 | [YouTube](https://www.youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2) |
| 24 | Udacity | Introduction to Parallel Programming | 单视频 | GPU/CUDA 并行编程导论 | [YouTube](https://www.youtube.com/watch?v=zb49vDrOxgA) |
| 25 | Udacity | Intro to the Class - Intro to Parallel Programming | 单视频 | CS344 课程导入 | [YouTube](https://www.youtube.com/watch?v=F620ommtjqk) |
| 26 | Udacity | A CUDA Program - Intro to Parallel Programming | 单视频 | CUDA 程序结构 | [YouTube](https://www.youtube.com/watch?v=dJ_D7JjOe8s) |
| 27 | Udacity | CUDA Program Diagram - Intro to Parallel Programming | 单视频 | CUDA 程序执行图 | [YouTube](https://www.youtube.com/watch?v=lQVV5JCd74I) |
| 28 | Udacity | Starting the CUDA project - Intro to Parallel Programming | 单视频 | CUDA 项目实践起步 | [YouTube](https://www.youtube.com/watch?v=1qh3SSl10tU) |
| 29 | CoffeeBeforeArch / Nick | CUDA Crash Course | 播放列表 | CUDA crash course，覆盖 vector add、matmul、reduction、convolution | [YouTube](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU) |
| 30 | CoffeeBeforeArch / Nick | From Scratch | 播放列表 | 从零写 CUDA vector add、matrix multiplication、tiled matmul | [YouTube](https://www.youtube.com/playlist?list=PLxNPSjHT5qvvwoy6KXzUbLaF5A8NdJvuo) |
| 31 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: Introduction | 单视频 | GPU 架构基础 | [YouTube](https://www.youtube.com/watch?v=4Pi424VJgcE) |
| 32 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: Programming Model Part 1 | 单视频 | GPU programming model | [YouTube](https://www.youtube.com/watch?v=5PcQIw1JFcA) |
| 33 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: Programming Model Part 2 | 单视频 | GPU 编程模型进阶 | [YouTube](https://www.youtube.com/watch?v=OccbJTcyPLQ) |
| 34 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: SIMT Core Part 2 | 单视频 | SIMT core | [YouTube](https://www.youtube.com/watch?v=2oUON2-SDHk) |
| 35 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: SIMT Core Part 3 | 单视频 | SIMT core 细节 | [YouTube](https://www.youtube.com/watch?v=BstLxr-1c-I) |
| 36 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: SIMT Core Part 4 | 单视频 | SIMT core 细节 | [YouTube](https://www.youtube.com/watch?v=1MOgOISHbX0) |
| 37 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: SIMT Core Part 5 | 单视频 | SIMT core 细节 | [YouTube](https://www.youtube.com/watch?v=Ut7WscyIoac) |
| 38 | CoffeeBeforeArch / Nick | Fundamentals of GPU Architecture: Warp Compaction | 单视频 | warp divergence/compaction 思路 | [YouTube](https://www.youtube.com/watch?v=p4Ef30Vza1o) |
| 39 | CoffeeBeforeArch / Nick | CUDA Crash Course: Vector Addition | 单视频 | 第一个 CUDA kernel | [YouTube](https://www.youtube.com/watch?v=2NgpYFdsduY) |
| 40 | CoffeeBeforeArch / Nick | CUDA Crash Course: Unified Memory Vector Add | 单视频 | Unified Memory 入门 | [YouTube](https://www.youtube.com/watch?v=84iwCupHW14) |
| 41 | CoffeeBeforeArch / Nick | CUDA Crash Course: Matrix Multiplication | 单视频 | CUDA 矩阵乘基础 | [YouTube](https://www.youtube.com/watch?v=XEOc4HCf_pQ) |
| 42 | CoffeeBeforeArch / Nick | CUDA Crash Course: Cache Tiled Matrix Multiplication | 单视频 | tiled matmul、cache/shared 思路 | [YouTube](https://www.youtube.com/watch?v=3xfyiWhtvZw) |
| 43 | CoffeeBeforeArch / Nick | CUDA Crash Course: Why Coalescing Matters | 单视频 | coalesced memory access | [YouTube](https://www.youtube.com/watch?v=_qSP455IekE) |
| 44 | CoffeeBeforeArch / Nick | CUDA Crash Course: Sum Reduction Part 3 | 单视频 | reduction 与 bank conflict 优化 | [YouTube](https://www.youtube.com/watch?v=iHeze1VdxYA) |
| 45 | CoffeeBeforeArch / Nick | Shared Memory Atomics and Dynamic Allocation in CUDA | 单视频 | shared memory atomics、动态 shared memory | [YouTube](https://www.youtube.com/watch?v=Bwv5J7dHYjU) |
| 46 | CoffeeBeforeArch / Nick | CUDA Crash Course: 1-D Convolution with Constant Memory | 单视频 | constant memory、1D convolution | [YouTube](https://www.youtube.com/watch?v=n7vtr2hCzoc) |
| 47 | CoffeeBeforeArch / Nick | CUDA Crash Course: GPU Performance Optimizations Part 1 | 单视频 | CUDA 性能优化思路 | [YouTube](https://www.youtube.com/watch?v=b8ESCws3_1s) |
| 48 | CoffeeBeforeArch / Nick | From Scratch: Matrix Multiplication in CUDA | 单视频 | 从零实现 matmul | [YouTube](https://www.youtube.com/watch?v=DpEgZe2bbU0) |
| 49 | CoffeeBeforeArch / Nick | From Scratch: Cache Tiled Matrix Multiplication in CUDA | 单视频 | tiled matmul 从零实现 | [YouTube](https://www.youtube.com/watch?v=ga2ML1uGr5o) |
| 50 | CoffeeBeforeArch / Nick | GPU Microbenchmarking: Inline PTX | 单视频 | inline PTX、微基准 | [YouTube](https://www.youtube.com/watch?v=iL0cRnjN2OE) |

### 9.2 进阶补充：GPU MODE、Triton、课程录播与科普

| # | 频道/博主 | 视频/播放列表 | 类型 | 适合看什么 | 链接 |
|---|-----------|---------------|------|------------|------|
| 51 | GPU MODE | cuda mode | 播放列表 | CUDA Mode/GPU Mode 系列课 | [YouTube](https://www.youtube.com/playlist?list=PLVEjdmwEDkgW0uEWNt4olLk-bnSueZARH) |
| 52 | GPU MODE | GPU mode lectures | 播放列表 | CUDA、Triton、NCCL、Tensor Core、kernel 优化 | [YouTube](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL) |
| 53 | GPU MODE | Lecture 2 Ch1-3 PMPP book | 单视频 | PMPP 前几章导读 | [YouTube](https://www.youtube.com/watch?v=NQ-0D5Ti2dc) |
| 54 | GPU MODE | Lecture 3: Getting Started With CUDA for Python Programmers | 单视频 | Python 程序员视角入门 CUDA | [YouTube](https://www.youtube.com/watch?v=4sgKnKbR-WE) |
| 55 | GPU MODE | Lecture 4 Compute and Memory Basics | 单视频 | 计算/内存基础、roofline 思维 | [YouTube](https://www.youtube.com/watch?v=lTmYrKwjSOU) |
| 56 | GPU MODE | Lecture 8: CUDA Performance Checklist | 单视频 | CUDA 性能检查清单 | [YouTube](https://www.youtube.com/watch?v=SGhfUhlowB4) |
| 57 | GPU MODE | Lecture 9 Reductions | 单视频 | reduction 优化 | [YouTube](https://www.youtube.com/watch?v=09wntC6BT5o) |
| 58 | GPU MODE | Lecture 14: Practitioners Guide to Triton | 单视频 | Triton 实战指南 | [YouTube](https://www.youtube.com/watch?v=DdTsX6DQk24) |
| 59 | GPU MODE | Lecture 16: On Hands Profiling | 单视频 | profiler 实战 | [YouTube](https://www.youtube.com/watch?v=SKV6kDk1s94) |
| 60 | GPU MODE | Lecture 17: NCCL | 单视频 | NCCL、多 GPU 通信 | [YouTube](https://www.youtube.com/watch?v=T22e3fgit-A) |
| 61 | GPU MODE | Lecture 23: Tensor Cores | 单视频 | Tensor Core 概念与用法 | [YouTube](https://www.youtube.com/watch?v=hQ9GPnV0-50) |
| 62 | GPU MODE | Lecture 40: CUDA Docs for Humans | 单视频 | 如何读 CUDA 文档 | [YouTube](https://www.youtube.com/watch?v=qmpGv72qPCE) |
| 63 | GPU MODE | Lecture 50: A learning journey CUDA, Triton, Flash Attention | 单视频 | CUDA/Triton/FlashAttention 学习路线 | [YouTube](https://www.youtube.com/watch?v=4jQTb6sRGLg) |
| 64 | GPU MODE | Bonus Lecture: CUDA C++ llm.cpp | 单视频 | LLM 推理中的 CUDA C++ | [YouTube](https://www.youtube.com/watch?v=WiB_3Csfj_Q) |
| 65 | Stanford Online | CS149 Lecture 7: GPU architecture and CUDA Programming | 单视频 | Stanford 并行计算课中的 GPU/CUDA | [YouTube](https://www.youtube.com/watch?v=qQTDF0CBoxE) |
| 66 | Programming Massively Parallel Processors | AUB Spring 2021 El Hajj | 播放列表 | PMPP 课程录播 | [YouTube](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) |
| 67 | Programming Massively Parallel Processors | Lecture 01 - Introduction | 单视频 | PMPP 课程导论 | [YouTube](https://www.youtube.com/watch?v=4pkbXmE4POc) |
| 68 | Programming Massively Parallel Processors | Lecture 03 - Multidimensional Grids and Data | 单视频 | 多维 grid/data 映射 | [YouTube](https://www.youtube.com/watch?v=c8dehGOB8mQ) |
| 69 | Programming Massively Parallel Processors | Lecture 04 - GPU Architecture | 单视频 | GPU architecture | [YouTube](https://www.youtube.com/watch?v=pBQJAwogMoE) |
| 70 | Programming Massively Parallel Processors | Lecture 05 - Memory and Tiling | 单视频 | memory and tiling | [YouTube](https://www.youtube.com/watch?v=31ZyYkoClT4) |
| 71 | Programming Massively Parallel Processors | Lecture 08 - Convolution | 单视频 | convolution pattern | [YouTube](https://www.youtube.com/watch?v=xEVyTZG1wlk) |
| 72 | Programming Massively Parallel Processors | Lecture 09 - Stencil | 单视频 | stencil pattern | [YouTube](https://www.youtube.com/watch?v=NOoSyDCVRU0) |
| 73 | Programming Massively Parallel Processors | Scan (Brent Kung) - Lecture 12 | 单视频 | parallel scan | [YouTube](https://www.youtube.com/watch?v=CcwdWP44aFE) |
| 74 | Argonne Meetings, Webinars, and Lectures | An Intro to GPU Architecture and Programming Models | 单视频 | Tim Warburton 的 GPU 架构与编程模型讲解 | [YouTube](https://www.youtube.com/watch?v=uvVy3CqpVbM) |
| 75 | Peter Messmer / cscsch | CUDA Part A: GPU Architecture Overview and CUDA Basics | 单视频 | CUDA 架构概览和基础 | [YouTube](https://www.youtube.com/watch?v=nRSxp5ZKwhQ) |
| 76 | Peter Messmer / cscsch | CUDA Part F: Kernel Optimizations: Shared Memory Accesses | 单视频 | shared memory 访问优化 | [YouTube](https://www.youtube.com/watch?v=qOCUQoF_-MM) |
| 77 | HPC Education | CUDA Programming | 播放列表 | CUDA lecture series | [YouTube](https://www.youtube.com/playlist?list=PL3xCBlatwrsXHv_AbfzpBSs9OjYWsevHw) |
| 78 | HPC Education | GPU Programming | 播放列表 | GPU programming lecture series | [YouTube](https://www.youtube.com/playlist?list=PL3xCBlatwrsXCGW4SfEoLzKiMSUCE7S_X) |
| 79 | HPC4AI | GPU Programming - Åbo Akademi University | 播放列表 | 大学 GPU programming 课程录播 | [YouTube](https://www.youtube.com/playlist?list=PL6tGegLTCzMn4CexvqZyKZHhZqfJD4FXz) |
| 80 | CMPS 297S/396AA | GPU Computing - Spring 2021 | 播放列表 | GPU Computing 课程录播 | [YouTube](https://www.youtube.com/playlist?list=PLZrjSW9GrEZE07f8gLECc0tfJwNDm75RW) |
| 81 | Simon Oz | GPU Programming | 播放列表 | GPU 编程动画讲解 | [YouTube](https://www.youtube.com/playlist?list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j) |
| 82 | Simon Oz | Introduction - GPU Programming Episode 0 | 单视频 | GPU programming 导论 | [YouTube](https://www.youtube.com/watch?v=c8mQYGbT310) |
| 83 | Simon Oz | CPU vs GPU - GPU Programming Episode 1 | 单视频 | CPU/GPU 对比 | [YouTube](https://www.youtube.com/watch?v=D8HS-H-MIe0) |
| 84 | Simon Oz | Modern GPU Architecture | 单视频 | 现代 GPU 架构 | [YouTube](https://www.youtube.com/watch?v=whPSD8sdx-0) |
| 85 | Simon Oz | Performance Characteristics | 单视频 | GPU 性能特征 | [YouTube](https://www.youtube.com/watch?v=3GlIV2hERzo) |
| 86 | Simon Oz | Occupancy | 单视频 | occupancy 概念 | [YouTube](https://www.youtube.com/watch?v=lQN_Tew4snI) |
| 87 | nickcorn93 | Tutorial: CUDA programming in Python with numba and cupy | 单视频 | Numba/CuPy 写 GPU 代码 | [YouTube](https://www.youtube.com/watch?v=9bBsvpg-Xlk) |
| 88 | Anaconda, Inc. | Writing CUDA kernels in Python with Numba | 单视频 | Python/Numba CUDA kernel | [YouTube](https://www.youtube.com/watch?v=E_1nC_fx6TU) |
| 89 | freeCodeCamp.org | CUDA Programming Course - High-Performance Computing with GPUs | 单视频 | 长课：CUDA/HPC/GPU 架构 | [YouTube](https://www.youtube.com/watch?v=86FAWCzIe_4) |
| 90 | Sasha Rush | GPU Puzzles: Let's Play | 单视频 | GPU Puzzles 互动式 CUDA 学习 | [YouTube](https://www.youtube.com/watch?v=K4T-YwsOxrM) |
| 91 | Branch Education | How do Graphics Cards Work? Exploring GPU Architecture | 单视频 | GPU 硬件架构科普 | [YouTube](https://www.youtube.com/watch?v=h9Z4oGN89MU) |
| 92 | Fireship | Nvidia CUDA in 100 Seconds | 单视频 | CUDA 快速科普 | [YouTube](https://www.youtube.com/watch?v=pPStdjuYzSI) |
| 93 | Computerphile | What is CUDA? | 单视频 | CUDA 概念科普 | [YouTube](https://www.youtube.com/watch?v=K9anz4aB0S0) |
| 94 | Computerphile | CPU vs GPU | 单视频 | CPU/GPU 差异 | [YouTube](https://www.youtube.com/watch?v=_cyVDoyI6NE) |
| 95 | Tom Nurkkala | CUDA Hardware | 单视频 | CUDA hardware 解释 | [YouTube](https://www.youtube.com/watch?v=kUqkOAU84bA) |
| 96 | Tom Nurkkala | Intro to GPU Programming | 单视频 | GPU programming 入门 | [YouTube](https://www.youtube.com/watch?v=G-EimI4q-TQ) |
| 97 | Zipped | C++ CUDA Tutorial: Theory & Setup | 单视频 | C++ CUDA 环境与理论 | [YouTube](https://www.youtube.com/watch?v=IuxJO0HOcH0) |
| 98 | Zachary Huang | Give Me 30 min, I'll Make CUDA Click Forever | 单视频 | CUDA 快速建立直觉 | [YouTube](https://www.youtube.com/watch?v=xewKxorikwE) |
| 99 | Low Level | Writing Code That Runs FAST on a GPU | 单视频 | GPU 上写快代码的直觉 | [YouTube](https://www.youtube.com/watch?v=8sDg-lD1fZQ) |
| 100 | eisfrosch | The Chaotic State of GPU Programming | 单视频 | CUDA/OpenCL/Triton 等 GPU 编程生态对比 | [YouTube](https://www.youtube.com/watch?v=9-DiGrnz8l8) |
| 101 | Tushar Gautam | 2678x Faster with CUDA C: Simple Matrix Multiplication on a GPU | 单视频 | CUDA C 矩阵乘入门 | [YouTube](https://www.youtube.com/watch?v=oQT7IC0x254) |
| 102 | Tushar Gautam | 4.5x Faster CUDA C with just Two Variable Changes | 单视频 | CUDA 矩阵乘微优化 | [YouTube](https://www.youtube.com/watch?v=QmKNE3viwIE) |
| 103 | achal | Intro to Parallel Reduction | 单视频 | CUDA reduction 概念 | [YouTube](https://www.youtube.com/watch?v=zL28uOxoXj8) |
| 104 | achal | CUDA Programming: Parallel Reduction | 单视频 | reduction CUDA 实现 | [YouTube](https://www.youtube.com/watch?v=YSLJOVinit0) |
| 105 | achal | CUDA Programming: Parallel Scan (Kogge-Stone) | 单视频 | parallel scan CUDA 实现 | [YouTube](https://www.youtube.com/watch?v=5hczlBr4yMo) |
| 106 | Aviraj Bevli | Stencil computation pattern in GPU programming CUDA | 单视频 | stencil 模式 | [YouTube](https://www.youtube.com/watch?v=XvPqBXp4Pg0) |
| 107 | TheJDen | Triton GPU Programming From Scratch - Tutorial | 单视频 | Triton 从零入门 | [YouTube](https://www.youtube.com/watch?v=LMU0LrX1f88) |
| 108 | GPU MODE | Optimizing Linear Attention in Triton | 单视频 | Triton 优化 attention | [YouTube](https://www.youtube.com/watch?v=j4zJbrNE9lA) |
| 109 | InfoWorld | GPU-accelerated Python with CuPy and Numba's CUDA | 单视频 | CuPy/Numba GPU Python | [YouTube](https://www.youtube.com/watch?v=wa0EmEq5Otw) |
| 110 | Molly Rocket | Zen, CUDA, and Tensor Cores - Part 1 | 单视频 | CUDA 与 Tensor Core 思路 | [YouTube](https://www.youtube.com/watch?v=uBtuMsAY7J8) |

### 9.3 推荐观看顺序

如果你只想先看一条主线，不建议从 100 多个资源里随机挑。可以按这个顺序：

```text
1. Branch Education / Computerphile / Fireship：先建立 GPU 和 CUDA 的直觉
2. Udacity CS344 或 Caltech/CS149 对应 YouTube 课：建立并行编程模型
3. CoffeeBeforeArch CUDA Crash Course：写 vector add、matmul、reduction、convolution
4. NVIDIA Developer CUDA + Nsight：学官方工具链和 profiler
5. GPU MODE：补 PyTorch/CUDA/Triton/NCCL/Tensor Core 现代生态
6. PMPP lectures：系统学习并行算法模式
```

---

## 10. 备注

- “课程是否仍在开设”会随学期变化；本表优先记录截至 2026-06-11 可访问的官方页面或公开资料。
- 有些课程是正式学分课，有些是 university HPC center 的短训；自学价值不完全取决于是否是学分课，而取决于是否有公开讲义、实验和作业。
- GPU 生态更新很快。CUDA 语法基础相对稳定，但 Tensor Core、TMA、异步拷贝、多 GPU 通信、compiler stack 相关内容需要结合最新 NVIDIA/AMD 文档补充。
