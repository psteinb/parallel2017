# House of ROC

Moderne Programmierung von Grafikkarten (GPU) für wissenschaftliches Rechnen findet heute vorrangig in C++ und Fortran (und teilweise in C) statt. Der Primus in diesem Feld ist Nvidias CUDA und das dazu gehörige Software-Ökosystem aus performanten und spezialisierten C-Bibliotheken (`cuFFT`, `cuBLAS`, `cuRAND`, `cuDNN`, `cuSPARSE` uvm.), der C-Laufzeitumgebung (`libcuda`, `libcudart` uvm.), C++-Abstraktionen (`thrust` und `cub`) sowie aus Entwicklungswerkzeugen (beispielsweise dem Debugger `cuda-gdb` oder der Profiling-Suite NVVP).  Die Entwickler erhalten in Verbindung mit einem breiten Spektrum an meist frei zugänglichen Lernressourcen wie Webinaren oder Online-Kursen einen vergleichsweise komfortablen Einstieg in die Programmierung von Nvidia GPUs. Dieser Einstieg verspricht nicht nur eine Beschleunigung der Anwendung eines Entwicklers sondern auch höhere Verkaufszahlen der Hardware von Big Green, denn CUDA-Applikationen und -Bibliotheken sind in erster Linie nur auf Nvidia-Hardware lauffähig.
Die offen standardisierte Alternative dazu, OpenCL, fristet leider seit Jahren ein Schattendasein. Trotz mehrerer Anstrengungen einer Renaissance (OpenCL 2.1 im Jahr 2015, OpenCL 2.2 im Jahr 2016) und der semantischen Nähe zu CUDA, hat dieses Paradigma wenig flächendeckende Anwendung erfahren verglichen mit CUDA. Gründe dafür sind unter anderem ein fehlendes einheitliches Software-Ökosystem für GPU- und CPU- basierte Plattformen, wirtschaftliche Interessen der Hersteller (welche hauseigene Technologien den Vorzug geben), die schiere Hardware-Komplexität im Wechselspiel zum low-level Character von OpenCL, die heterogenen Möglichkeiten der Kompilation von OpenCL-Kerneln uvm.

Aus dieser Situation heraus hat AMD in den vergangenen Monaten eine Open-Source Plattform zur Ausführung, Analyse und Kompilierung von Software auf AMD-GPU-Hardware geschaffen. Innerhalb der AMD-Initiative Radeon Open Compute (ROC) ist ein vollständiges und reichhaltiges Ökosystem von Treibern, Laufzeitumgebungen, Compiler-Infrastruktur und Analyse-Werkzeugen unter dem Namen [ROCm](http://gpuopen.com/compute-product/rocm/) (*R*adeon *O*pen *C*ompute Platfor*m*) entstanden. Als Betriebssystem wird aktuell exklusiv Linux unterstützt, da der Fokus des Projekts auf dem serverbasierten Cloud-, Deep-Learning- und HPC-Markt liegt.

Das Herzstück der ROCm bildet der [Kerneltreiber](https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver) _ROCk_. Der Treiber wird aus AMD-spezifischen Paketquellen im laufenden Betrieb installiert und bei einem Reboot aktiviert. Diese Kernelerweiterung unterstützt auf der Hostseite Intels Xeon E3 und E5 Chips der Haswell-Generation oder neuer, sowie alle Core i3/i5/i7 Chips der 3. Generation und aufwärts. In kommenden Versionen der ROCm-Plattform werden auch AMDs hauseigene Naples- und Ryzen-Architekturen unterstützt sowie Caviums Thunder X ARM-Chipsatz. ROCm arbeitet zum Zeitpunkt der Fertigstellung des Artikels mit AMDs Fiji und Polaris-GPU-Karten zusammen, also der 3. und 4. Generation der "Graphics Core Next"-Architektur (GCN). Das Treibermodul unterstützt außerdem den Multi-GPU-Betrieb sowie Remote Direct Memory Access (RDMA) und exportiert zusätzlich eine Systemmanagement-API für bspw. die Anbindung von Monitoring-Werkzeugen.

Auf der Kernelinfrastruktur baut die _ROCr_-[Laufzeitumgebung](https://github.com/RadeonOpenCompute/ROCR-Runtime) auf. _ROCr_ implementiert nicht nur die Laufzeitspezifikation der Heterogeneous System Architecture (HSA) Foundation, sondern beinhaltet auch diverse Zusätze für den Multi-GPU-Betrieb - gerade für Deep-Learning-Anwendungen und anspruchsvolle Simulationen ein wichtiges Feature. Des Weiteren werden APIs zur "Device Discovery", Nebenläufigkeit von CPU- und GPU-Prozessen und GPU-Multitasking, atomare Speichertransaktionen und Signale sowie "User Space"-Warteschlangen und flache Speicheradressierung angeboten.

Die Hauptinteraktion eines Anwendungsentwicklers mit der ROCm-Plattform ist aber schließlich der mitgelieferte Compiler `hcc` (heterogenous compute compiler). Dieser basiert auf der LLVM/Clang-Compiler-Infrastruktur und bietet Front-Ends zur Verabeitung von Quelltext in OpenCL, HIP und HC (zu beiden später mehr) sowie zukünftig OpenMP4 zur Programmierung der GPU. Der OpenCL-Support ist dabei erst kürzlich und auf Drängen der Community in ROCm-Version 1.4 hinzugekommen. Im Back-End generiert dieser Compiler native GCN-ISA-Instruktionen, die auf der GPU ausgeführt werden. Die ROCm bietet auch hierfür Werkzeuge zur weiteren Erforschung des Codes (assembler, disassembler) sowie ein offenes Code-Objekt-Format ("hsaco").

Für den erfahrenen GPU-Programmierer liefern OpenCL und OpenMP Standardansätze zur Bechreibung von Parallelität auf Multi- und Many-Core-Architekturen abseits von CUDA. Heterogeneous-compute Interface for Portability (HIP) und HC sind jedoch Eigenentwicklungen aus dem Haus AMD. [HIP](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP) ist eine CUDA-nahe Sprache und soll Entwicklern (gegebenenfalls mithilfe von dedizierten Werkzeugen) die Überführung von CUDA-Anwendungen und CUDA-Bibliotheken in eine auf AMD- und Nvidia-Hardware lauffähige Form erleichtern. HIP selbst bietet C++ in Kernel-Konstrukten an. Der Standard 2011/2014 und teilweise 2017 werden unterstützt. Die Laufzeitprogrammierung auf der CPU geschieht wie bei CUDA mittels einer C API, welche auch in C++ nutzbar ist.
HIP unterstützt hierbei die meist genutzten CUDA-Features wie Speicher-Management auf dem _Device_, _Streams_ zur Verwaltung von CPU-GPU-Nebenläufigkeit inklusive Ereignisse zur Synchronisation sowie die Profiling-API. Die HIP-Kompilate erfreuen sich 100% Tool-Support aller Profiler und Debugger aus dem CUDA- wie auch dem ROCm-Universum.

Zur Illustration der weiteren Diskussion soll an dieser Stelle der open-source [BabelStream](https://github.com/UoB-HPC/BabelStream)-Benchmark der Universität Bristol dienen. Dieser Benchmark in C++11 baut auf dem klassischen HPC-Speicherbandbreiten-Benchmark STREAM von John McAlpin auf. BabelStream implementiert die vier Vektor-Operationen (`copy`, `mul`tiply, `add`, `triad`) sowie das Skalarprodukt (`dot`):

```
c[:]    = a[:]					/* copy  */
b[:]    = scalar*b[:]			/* mul   */
c[:]    = a[:] + b[:]			/* add   */
a[:]    = b[:] + scalar*c[:]    /* triad */
scalar  = dot(a[:],b[:])		/* dot   */
```

Der Benchmark selbst führt diese Operationen wiederholt auf synthetisch gefüllten Feldern aus, deren Größe durch den Nutzer konfigurierbar ist. Die Messung der Laufzeit dieser Operationen bietet nicht nur eine direkte Messung der Speicherbandbreite auf der GPU, sondern auch der Effizienz eines Programmierparadigmas und der benutzten Compiler-Infrastruktur.

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

Wie in Listing 2 sichtbar, templatisiert BabelStream alle Benchmark-Instanzen, um den benutzten Datentyp von einfach-genauen zu doppelt-genauen Fließkommazahlen ohne zusätzlichen Code auszutauschen. Zeile 8 in Listing 2 zeigt den Aufruf des Device-Kernels in vereinfachter Form. Die drei Zeiger `d_a, d_b, d_c` sind hierbei allozierte Felder auf der GPU mit der Anzahl `array_size` an Elementen. CUDA verlangt für jeden Kernel-Aufruf eine doppelte, virtuelle Partitionierung des Indexraumes aller zu bearbeitenden Elemente. In diesem Fall ist der Indexraum eindimensional - `array_size/TBSIZE` und `TBSIZE` sind ganzzahlige Skalare - der in `array_size/TBSIZE` Blöcke und `TBSIZE` Threads pro Block zerlegt wird. Diese Notation umgeht das explizite Angeben einer for-Schleife über den Indexraum. Es erzwingt aber das manuelle Zusammensetzen des Indexes, wie in Zeile 4 zu sehen ist. Dabei sind `blockDim`, `blockIdx` und `threadIdx` C-Strukturen, die von der CUDA-Laufzeit-API zur Verfügung gestellt werden.

Quellcodes wie in Listing 2 können nun mit dem Kommandozeilen-Werkzeug `hipify` in der ROCm-Plattform nach HIP übersetzt werden.

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

Wie man sieht, ist Listing 3 strukturell identisch zur Listing 2. Es wurden Umbenennungen der Laufzeit-structs vorgenommen (`hipBlockDim_x`, `hipBlockIdx_x`, `hipThreadIdx_x`). Der Aufruf und die Signatur des Kernels selbst wurden durch HIP-spezifische Boilerplate-Argumente ergänzt, um vergleichbare Funktionalität zu CUDA zu erreichen und dennoch Code zu generieren, der von `nvcc` und `hcc` verarbeitet werden kann.

HIP ist damit ein interessantes und ernst zu nehmendes Werkzeug zur Konvertierung von Legacy-CUDA-Anwendungen, das soll AMD ermöglichen, schnell auf AMD-GPUs lauffähige Projekte zu gewinnen. HIP-Code kann zudem mit einem Ökosystem an spezialisierten Bibliotheken interagieren, wie beispielsweise [hipBlas](https://bitbucket.org/multicoreware/hcblas), [hipFFT](https://bitbucket.org/multicoreware/hcFFT), [hipRNG](https://bitbucket.org/multicoreware/hcrng).

Einen etwas innovativeren Anstrich hat die C++ API `hc`. Hier sind Laufzeit- und Device-Kernel-Umgebung in C++ benutzbar. Der Kern der Sprache basiert auf dem offenen C++AMP 1.2 Standard innerhalb des `hc`-Namensraumes. Dazu sind Erweiterungen bspw. für den asynchronen Memorytransfer im selben Namensraum gekommen und die Möglichkeit in Host- und Device-seitigem Quelltext C++14 zu benutzen. Die Struktur der API ist ähnlich der von [thrust](http://thrust.github.io/), [boost.compute](https://github.com/boostorg/com) oder [sycl](https://www.khronos.org/sycl).

![Abb. 1. Struktur der `hc` API in ROCm in Bezug zum heutigen Hardwaremodell einer CPU plus dedizierter GPU ](hc_api_nutshell_inv.pdf)

Während in Abbildung 1 die host-seitigen Strukturen durch Container, Algorithmen und Funktionen der C++-Sprachen und -standardbibliothek abgebildet werden, gibt es eine API zur Arbeit mit GPU-Speicherbereichen (`hc::array` und `hc::array_view`), zum Transfer von Daten von und zur GPU (`hc::copy`, `hc::async_copy`) sowie Funktionen zur Durchführung von Berechnungen und anderen Operationen auf dem Device (`hc::parallel_for_each`). Im Gegensatz zu aktuellen low-level GPU-Sprachen wie CUDA, wird auf eine Kernel-Syntax bzw. das Grid-Threadblock-Dispatchment ganz verzichtet. Optimierungen, welche die Ausführung von `hc::parallel_for_each` auf der Hardware zur Laufzeit betreffen, werden vom Compiler bzw. der Laufzeitumgebung durchgeführt.

In Anlehnung an obigen BabelStream-Code in CUDA, gestaltet sich die Implementierung des Add-Kernels in `hc` wie folgt:

```
template <class T>
void HCStream<T>::add()
{
    hc::array_view<T,1> view_a(this->d_a);
    hc::array_view<T,1> view_b(this->d_b);
    hc::array_view<T,1> view_c(this->d_c);

    hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> i) [[hc]] {
                                  view_c[i] = view_a[i]+view_b[i];
                                  });
}
```

Die bereits allozierten Speicherbereiche auf der GPU `d_a`, `d_b` und `d_c` werden in der `HCStream`-Klasse durch Instanzen vom Typ `hc::array` repräsentiert. Zur vereinfachten Handhabung im folgenden Lambda-Aufruf werden Referenzen auf diese Felder in Objekte vom Typ `hc::array_view` gekapselt. Dies ermöglicht die Übergabe _per-value_ an die Lambda-Funktion später (hier zu rein illustrativen Zwecken benutzt). Die Funktion `hc::parallel_for_each` wird aber nicht nur mit Funktionalität versorgt, sondern auch mit einer Definition des Indexraums welcher bearbeitet werden soll. In diesem Fall ist das ein eindimensionaler Index im rechts-offenen Intervall `[0,array_size)`, dessen Dimensionalität zur Compilezeit fest stehen muss. Dementsprechend muss die Signatur der Lambda-Funktion ebenfalls dieser Dimensionalität folgen und nimmt ein von der Laufzeitbibliothek zur Verfügung gestelltes `hc::index<1>`-Objekt als Parameter. Dieses wird schlussendlich benutzt, um die Operationen auf den GPU-Feldern `d_a`, `d_b` und `d_c` zu platzieren.

![Abbildung 2 Vergleich der Bandbreiten des Add-Kernels für verschiedene Feldgrößen, GPU-Hardware und Sprachparadigmen.](gpu_stream_lim_add_with_nvidia_bw.pdf)

Abbildung 2 zeigt, mit welcher Güte die Implementierungen aller drei Programmierparadigmen auf der ROCm von einer gemeinsamen Compiler-Infrastruktur profitieren. Die Benchmarks mittels `hc`, `hip` und `OpenCL` liegen über das gesamte Spektrum von Feldgrößen gleichauf - bis auf stochastische Schwankungen. Ein beeindruckender Fakt nebenbei: die Speicherbrandbreite einer Fiji R9 Nano (veröffentlicht 2015) ergibt sich doppelt so hoch als die einer Nvidia GeForce GTX 1080 (veröffentlicht 2016). Der Grund hierfür liegt in der Speicherarchitektur. Die AMD-Karte benutzt High Bandwidth Memory der ersten Generation, wobei die Nvidia-Karte GDDR5 DRAM benutzt. Recht deutlich fällt der Vorsprung einer Nvidia Tesla P100 durch ihren High Bandwidth Memory der zweiten Generation gegenüber der Fiji Nano und der GeForce-Karte aus.

Zuletzt möchte ich noch ein kleines Juwel in der `hc` API vorstellen, das aus der Sicht des GPU-Entwicklers eine besondere Erwähnung verdient. Die Funktionen `hc::parallel_for_each` sowie `hc::async_copy` geben als Rückgabewert ein Objekt vom Typ `hc::completion_future` zurück. Damit wären auch Konstrukte für Daten- und Aufgabenabhängigkeiten denkbar, die ähnlich zu dem für C++2020 angedachten Concurrency TS sind:

```
std::vector<float> payload (/*pick a number if not 42*/);
hc::array<float,1> d_payload(payload.size());

hc::completion_future when_done = hc::async_copy(payload.begin(),
                                                 payload.end(),
                                                 d_payload);
when_done.then(call_kernel_functor); //continuation function!
```

Dies eröffnet technologisch vielerlei Möglichkeiten, um asynchrone Operationen im Wechselspiel CPU-GPU (Task-Parallelität) zu implementieren und damit die Fähigkeiten einer heterogenen Hardware des 21. Jahrhunderts in vielen Szenarien mit minimalem Code auszureizen. Damit wären auch Konstrukte zum Ausdruck von Daten- sowie Aufgabenabhängigkeiten ähnliche des Concurrency TS, welcher für C++2020 diskutiert wird, denkbar:

```
std::vector<hc::completion_future> streams(n);
for(hc::completion_future when_done : streams){

    when_done = hc::async_copy(payload_begin_itr,
                               payload_end_itr,
                               d_payload_view);
    when_done.then(parallel_for_each(/*do magic*/))
             .then(hc::async_copy(d_payload_view,result_begin_itr));
}

hc::when_all(streams);
```

In obigem Pseudocode, werden `n` Berechnungsschritte inkl. Datentransfer zu und von der GPU an ein `completion_future` gebunden. Die Laufzeitumgebung erhält damit die Freiheit die Operationen derart auszuführen, so dass maximale Bandbreite und Latenz erreicht wird. Der Schritt `hc::when_all(streams)` dient als Synchronisations-Barriere. Man sieht an diesem Beispiel, welches Potential an Ausdrucksstärke in `hc` steckt.

Zusammenfassend kann man feststellen, dass die ROCm-Plattform ein junges und ambitioniertes Projekt ist. Dieser Softwarestack aus dem Haus AMD baut von Keller bis Dach auf Open-Source auf (Kerneltreiber, Laufzeitumgebung, Compiler, Bibliotheken). `hc` kann zu einer ausdrucksstarken Sprache avancieren, welche den Boiler-Plate-Anteil am Quelltext in realen Projekten drastisch reduzieren kann. Es wird sich zeigen, wie stabil die `hc` API ist und inwiefern sie sich neben C++17 für Device-Operationen behaupten kann. AMD plant verschiedene Algorithmen der parallelen STL in C++17 auch auf die GPU zu bringen.

Es lohnt sich also, das ROCm-Projekt im Blick zu behalten, auch wenn in Version 1.4 die elementaren Dinge wie Dokumentation und Werkzeuge wie der Profiler und Debugger noch nicht ganz reif für den Projekteinsatz sind. Die ROCm-Infrastruktur und die AMD-Hardware scheinen aber in den Startlöchern zu stehen, um eine Aufholjagd mit CUDA zu bestreiten.

Peter Steinbach ist Scientific Software Engineer bei [Scionics Computer Innovation GmbH](www.scionics.de) in Dresden und arbeitet in dieser Rolle als interner Dienstleister und Berater für das [Max-Planck Institut für molekulare Zellbiologie und Genetik](www.mpi-cbg.de) vor allem im Bereich HPC, GPU-Programmierung und Performance-Engineering. Scionics ist Mitglied im [GPU Center of Excellence](www.gcoe-dresden.de) unter Schirmherrenschaft der [TU Dresden](www.tu-dresden.de), in welchem sich lokale Institute und Industriepartner der effizienten Nutzung von Grafikkarten zum Zwecke der Visualisierung und performanter Berechnungen austauschen. Ich möchte mich bei Matthias Werner (ZiH, TU Dresden) und Robert Haase (Scionics) für das wertvolle Feedback zu diesem Artikel herzlich bedanken.
