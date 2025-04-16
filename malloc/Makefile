TARGET = bench

all: $(TARGET)

$(TARGET): bench.cpp
	hipcc -o $@ -O2 $<
