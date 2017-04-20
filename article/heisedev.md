# House of ROC

Moderne Programmierung von Grafikkarten (GPU) für wissenschalftliches Rechnen findet heute vorranging in C und C++ statt. Der Primus in diesem Feld ist Nvidias CUDA and das dazu gehörige Software-Ökosystem aus performanten und spezialisierten C-Bibliotheken (`cuFFT`, `cuBLAS`, `cuRAND`, `cuDNN`, `cuSPARSE` uvm.), der C-Laufzeitumgebung (`libcuda`, `libcudart` uvm.), C++-Abstraktionen (`thrust` und `cub`) sowie Entwicklungswerkzeugen (bspw. dem Debugger `cuda-gdb` oder der Profiling-Suite NVVP), damit Entwickler einen einfachen und erfolgreichen Einstieg in die Programmierung von Nvidia Hardware haben. Dieser Einstieg verspricht nicht nur eine Beschleunigung der Anwendung eines Entwicklers sondern auch höhere Verkaufszahlen der Hardware von Big Green, denn CUDA-Applikationen und -Bibliotheken sind in erster Linie nur auf Nvidia-Hardware lauffähig.
Die offen standardisierte Alternative dazu, OpenCL, fristet leider seit Jahren ein Schatten dasein. Trotz mehrerer Anstrengungen einer Renaissance (OpenCL 2.1 im Jahr 2015, OpenCL 2.2 im Jahr 2016) und der semantischen Nähe zu CUDA, hat dieses Paradigma wenig flächendeckende Anwendung erfahren, wenn man es zu CUDA vergleicht. Hauptgrund dafür ist u.a. die fehlende Unterstützung von Werkzeugen auf allen Platformen (CPU und GPU basiert), die ,just-in-time''-Übersetzung der GPU-Kernel zur Laufzeit und die Notwendigung von maßgeschneiderten Implementierungen für jede unterstützte Architektur.

Aus dieser Situation heraus hat AMD in den vergangenen Monaten eine Open-Source Platform zur Ausführung, Analyse und Kompilierung von Software auf AMD-GPU-Hardware geschaffen. Innerhalb der Initiative Radeon Open Compute (ROC) ist ein vollständiges und reichhaltiges Ökosystem von Treibern, Laufzeitumgebungen, Compiler-Infrastruktur und Analyse-Werkzeugen unter dem Namen [ROCm](http://gpuopen.com/compute-product/rocm/) (*R*adeon *O*pen *C*ompute Platfor*m*) entstanden. Als Betriebssystem wird exklusiv Linux unterstützt, da der Fokus des Projekts auf dem serverbasierten Cloud-, Deep-Learning- und HPC-Markt liegt. 

Das Herzstück der ROCm bildet der [Kerneltreiber](https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver), _ROCk_. Der Treiber wird aus AMD-spefizische Paketquellen im laufenden Betrieb installiert und bei einem Reboot aktiviert. Diese Kernelerweiterung unterstützt auf der Hostseite Intels Xeon E3 und E5 Chips der Haswell-Generation oder neuer, sowie alle Core i3/i5/i7 Chips der Generation `v3` und aufwärts. In kommenden Versionen der ROCm-Platform werden auch AMDs hauseigenen Naples- und Ryzen-Kerne unterstützt sowie Caviums Thunder X ARM-Chipsatz. ROCm arbeitet zum Zeitpunkt der Fertigstellung des Artikels mit AMDs Fiji und Polaris-GPu-Karten zusammen, also der 3. und 4. Generation der "Graphics Core Next"-Architektur (GCN). Das Treibermodul unterstützt außerdem RDMA, den Multi-GPU-Betrieb und exportiert eine Systemmanagement-API für die Anbindung von Monitoring-Werkzeugen u.ä.

Auf der Kernelinfrastruktur baut die _ROCr_-[Laufzeitumgebung](https://github.com/RadeonOpenCompute/ROCR-Runtime) auf. _ROCr_ implementiert nicht nur die Laufzeitspezifikation der HSA Foundation, sondern beinhaltet auch diverse Zusätze für den Multi-GPU-Betrieb - gerade für Deep-Learning-Anwendungen und anspruchsvolle Simulationen ein wichtiges Feature. Desweiteren werden APIs zur "Device Discovery", Nebenläufigkeit von CPU- und GPU-Prozessen und GPU-Multitasking, atomare Speichertransaktionen und Signale sowie "User Space"-Warteschlangen und flache Speicheraddressierung.

Die Hauptinteraktion eines Anwendungsentwicklers mit der ROCm-Platform ist aber schließlich der mitgelieferte Kompiler `hcc` (heterogenous compute compiler). Dieser basiert auf der LLVM/Clang-Compiler-Infrastruktur und bietet Front-Ends zur Verabeitung von Quelltext in OpenCL, HIP und HC (zu beiden später mehr) sowie zukünftig OpenMP4 zur Programmierung der GPU. Der OpenCL-Support ist dabei erst kürzlich und auf Drängen der Community in Version 1.4 hinzugekommen. Im Back-End generiert der Kompiler native GCN-ISA-Instruktionen, die auf der GPU ausgeführt werden. Die ROCm bietet auch hierfür Werkzeuge zur weiteren Erforschung des Codes (assembler, disassembler) sowie ein offenes Code-Objekt-Format. 

Für den erfahrenen GPU-Programmierer sind OpenCL und OpenMP Standardansätze zur Bechreibung von Parallelität auf Multi- und Many-Core-Architekturen. HIP und HC sind jedoch Eigenentwicklungen aus dem Haus AMD. [HIP](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP) ist eine CUDA-nahe Sprache, die es Entwicklern leicht machen soll, CUDA-Anwendungen oder -Bibliotheken in eine Form zu überführen, die auf AMD- und Nvidia-Hardware lauffähig ist. HIP selbst bietet C++ in Kernel-Konstrukten an (es wird der Standard 2011/2014 und teilweise 2017 unterstützt). Die Laufzeitprogrammierung auf der CPU geschieht wie bei CUDA mittels einer C API.
HIP unterstützt hierbei die meist genutzten CUDA-Feature wie Speicher-Allokationen auf dem _Device_, streams zur Verwaltung von CPU-GPU-Nebenläufigkeit, Ereignisse zur Synchronisation von CPU-GPU-Nebenläufigkeit sowie die Profiling-API. Die HIP-Kompilate erfreuen sich 100% Tool-Support aller Profiler und Debugger aus dem CUDA- wie auch dem ROCm-Universum. 

Zur Illustration soll an dieser Stelle der open-source [BabelStream](https://github.com/UoB-HPC/BabelStream)-Benchmark der Universität Bristol dienen. Dieser Benchmark in C++11 baut auf dem klassischen HPC-Speicherbandbreiten-Benchmark STREAM von John McAlpin auf. Er implemiert die vier Vektor-Operationen (`copy`, `mul`tiply, `add`, `triad`) sowie das Skalarprodukt (`dot`):

```
c[:]    = a[:]					/* copy  */ 
b[:]    = scalar*b[:]			/* mul   */ 
c[:]    = a[:] + b[:]			/* add   */ 
a[:]    = b[:] + scalar*c[:] 	/* triad */ 
scalar  = dot(a[:],b[:])		/* dot   */ 
```

Der Benchmark selbst führt diese Operationen wiederholt auf synthetisch gefüllten Feldern aus, deren Größe durch den Nutzer konfigurierbar ist. Die Messung der Laufzeit dieses Ensembles bietet nicht nur eine direkte Messung der Speicherbandbreite auf der GPU, sondern auch der Effizienz eines Programmierparadigmas und der benutzten Compiler-Infrastruktur.

Der `add`-Kernel in der CUDA-Implementierung sieht hierfür so aus:

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

BabelStream templatisiert alle Benchmark-Instanzen, um den benutzten Datentyp von einfach-genauen zu doppelt-genauen Fließkommazahlen ohne zusätzlichen Code zu wechseln. Zeile 8 in Listing 2 zeigt den Aufruf des Device-Kernels in vereinfachter Form. Die drei Zeiger `d_a, d_b, d_c` sind hierbei allozierte Felder auf der GPU mit der Anzahl `array_size` an Elementen. CUDA verlangt für jeden Kernel-Aufruf eine doppelte virtuelle Partitionierung des Indexraumes aller zu bearbeitenden Elemente. In diesem Fall ist der Indexraum eindimensional - `array_size/TBSIZE` und `TBSIZE` sind ganzzahlige Skalare - der in `array_size/TBSIZE` Grid-Netze und `TBSIZE` Thread-Blöcke zerlegt wird. Diese Notation umgeht das explizite Angeben einer for-Schleife über den Indexraum. Es erzwingt aber das manuelle Zusammensetzen des Indexes wie in Zeile 4 zu sehen ist. Dabei sind `blockDim`, `blockIdx` und `threadIdx` C-Strukturen, die von der CUDA-Laufzeit-API zur Verfügung gestellt werden.

Quellcode wie in Listing 2 kann nun mit dem Kommandozeilen-Werkzeug `hipify` in der ROCm-Platform nach HIP übersetzt werden.

```
__global__ void add_kernel(hipLaunchParm lp, 
                           const T * a, const T * b, 
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

Wie man sieht ist Listing 3 strukturell identisch zur Listing 2. Es wurden Umbenennungen der Laufzeit-structs vorgenommen (`hipBlockDim_x`, `hipBlockIdx_x`, `hipThreadIdx_x`) Der Aufruf und die Signature des Kernels selbst wurde durch Boilerplate-Argumente ergänzt, um vergleichbare Funktionalität zu CUDA zu erreichen und dennoch Code zu generieren, der von `nvcc` und `hcc` verarbeitet werden kann.

HIP ist damit ein interessantes und ernst zu nehmendens Werkzeug zur Konvertierung von Legacy-CUDA-Anwendungen, das es AMD ermöglichen soll, schnell auf AMD-GPUs lauffähige Projekte zu gewinnen. HIP-Code kann zudem mit Ökosystem an spezialisierten Bibliotheken interagieren, wie bspw. [hipBlas](https://bitbucket.org/multicoreware/hcblas), [hipFFT](https://bitbucket.org/multicoreware/hcFFT), [hipRNG](https://bitbucket.org/multicoreware/hcrng).

Einen etwas innovatieveren Anstrich hat die Sprache `hc`. Hier sind Laufzeit- und Device-Kernel-Umgebung in C++ geschrieben. Der Kern der Sprache basiert auf dem offenen C++AMP 1.2 Standard innerhalb des `hc`-Namensraum. Dazu sind Erweiterungen gekommen und die Möglichkeit in Host- und Device-seitig C++14 zu benutzen. Die Struktur der Sprache ist ähnlich der von [thrust](http://thrust.github.io/), [boost.compute](https://github.com/boostorg/com) oder [sycl](https://www.khronos.org/sycl). 

![](../fig/hc_api_nutshell_inv.pdf)



