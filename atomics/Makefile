TARGET = atomic_test atomic_test-unsafe

all: $(TARGET)

atomic_test-unsafe: EXTRA = -munsafe-fp-atomics
$(TARGET): atomic_test.hip
	hipcc -o $@ -O2 $(EXTRA) $^
