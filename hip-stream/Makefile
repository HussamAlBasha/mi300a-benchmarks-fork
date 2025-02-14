EXECUTABLE = ./stream
all: $(EXECUTABLE)

.PHONY: test

OBJECTS = stream.o mpi.o

CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O2 -DNDEBUG

LDFLAGS = -I$(ROCM_PATH)/include/ -L$(ROCM_PATH)/lib  -lamdhip64 -lhsa-runtime64

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

%.o: %.hip
	hipcc $(HIPCC_FLAGS) -c $^ -o $@

%.o: %.cpp
	mpic++ $(CXXFLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	mpic++ $^ $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build
