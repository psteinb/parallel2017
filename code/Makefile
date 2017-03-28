SRC_FILES=$(wildcard *cpp)
DST_FILES=$(SRC_FILES:%.cpp=%)

HCC ?= hcc

HC_CXXFLAGS ?= -hc -stdlib=libc++ -I/opt/rocm/hcc-lc/include
HC_LDFLAGS  ?= -g -fstandalone-debug -hc -L/opt/rocm/hcc-lc/lib -Wl,--rpath=/opt/rocm/hcc-lc/lib -lc++ -lc++abi -ldl -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive -pthreads


all : $(DST_FILES)

% : %.cpp
	$(HCC) $(HC_CXXFLAGS)  $^ $(HC_LDFLAGS) -o $@
