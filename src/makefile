all:
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp/pearl.cpp -o cpp/pearl`python3-config --extension-suffix`

clean:
	rm cpp/*.so
