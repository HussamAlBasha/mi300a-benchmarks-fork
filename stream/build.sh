hipcc -o mstream -O2 -g -fopenmp ../mstream.cpp
hipcc -o mstream-managed -O2 -g -fopenmp ../mstream.cpp -DSTATIC_MANAGED
