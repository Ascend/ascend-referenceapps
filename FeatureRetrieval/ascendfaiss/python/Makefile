FAISS_INSTALL_PATH=/usr/local
FAISS_HEADER_PATH=$(FAISS_INSTALL_PATH)/include

ASCEND_MODULE_NAME=swig_ascendfaiss
ASCEND_FAISS_PATH=..

-include $(ASCEND_FAISS_PATH)/makefile.inc

# do not auto remove intermediate files
.SECONDARY:

all: build

%.cpp: %.swig $(ASCEND_FAISS_PATH)/libascendfaiss.a
	$(SWIG) -python -c++ -Doverride= -module $(ASCEND_MODULE_NAME) \
				-I$(FAISS_HEADER_PATH) -I$(ASCEND_FAISS_PATH) $(SWIGFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) $(PYTHONCFLAGS) \
               -I$(FAISS_HEADER_PATH) -I$(ASCEND_FAISS_PATH) -c $< -o $@

_%.so: %.o
	$(CXX) $(SHAREDFLAGS) $(LDFLAGS) -o $@ $^ -L$(ASCEND_FAISS_PATH) -lascendfaiss $(LIBS)

build: _$(ASCEND_MODULE_NAME).so ascendfaiss.py
	$(PYTHON) setup.py build

install: build
	$(PYTHON) setup.py install

clean:
	rm -f *.o *.cpp $(ASCEND_MODULE_NAME).py _*.so
	rm -rf build/

.PHONY: all build clean
