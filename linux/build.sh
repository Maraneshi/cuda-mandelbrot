FILES="../*.cpp ../*.c ../*.cu"
FLAGS="-std=c++11 -O3 -arch=sm_50 -res-usage -lineinfo -use_fast_math"
FOLDERS="-I../include -L../lib"
LIBS="-lGL -lglut -lGLU"
TARGET="../bin/mandelbrot"

mkdir -p ../bin
nvcc $FILES $FOLDERS $FLAGS $LIBS -o $TARGET
