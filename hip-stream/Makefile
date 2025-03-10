EXE = ./stream
all: $(EXE) $(EXE)-device $(EXE)-constant $(EXE)-managed

.PHONY: test

OBJECTS = stream.o mpi.o

CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O2 -DNDEBUG -Wall

ifdef VERBOSE
	HIPCC_FLAGS += -DVERBOSE=1
endif

$(EXE): LDFLAGS = -I$(ROCM_PATH)/include/ -L$(ROCM_PATH)/lib  -lamdhip64 -lhsa-runtime64

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

stream-device:	HIPCC_FLAGS += -DDEVICE_ARRAY
stream-constant:	HIPCC_FLAGS += -DCONSTANT_ARRAY
stream-managed:	HIPCC_FLAGS += -DMANAGED_ARRAY

stream.o: HIPCC_FLAGS += -c
stream.o stream-device stream-constant stream-managed: stream.hip
	hipcc $(HIPCC_FLAGS) $^ -o $@

%.o: %.cpp
	mpic++ $(CXXFLAGS) -c $^ -o $@

CC = hipcc
$(EXE): CC = mpic++
$(EXE): mpi.o

$(EXE)%: stream%.o
#mpic++ $^ $(LDFLAGS) -o $@

test: $(EXE)
	$(EXE)

clean:
	rm -rf $(EXE) $(OBJECTS) build
