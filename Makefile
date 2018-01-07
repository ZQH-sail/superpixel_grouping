all:
	g++ -O2 -std=c++11 src/main.cpp `pkg-config --libs --cflags opencv` -o run
