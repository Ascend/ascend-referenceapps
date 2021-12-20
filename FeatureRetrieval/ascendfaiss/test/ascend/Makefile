-include ../../makefile.inc

TESTS_SRC = TestRpc.cpp TestAscendIndexFlat.cpp TestAscendIndexIVFPQ.cpp \
	TestAscendIndexIVFSQ.cpp TestAscendIndexPreTransform.cpp \
	TestAscendIndexSQ.cpp TestAscendIndexIVFFlat.cpp \
	TestAscendIndexInt8Flat.cpp TestAscendIndexInt8IVFFlat.cpp

TESTS_OBJ = $(TESTS_SRC:.cpp=.o)
TESTS_BIN = $(TESTS_OBJ:.o=)

run: $(TESTS_BIN)
	for t in $(TESTS_BIN); do ./$$t || exit; done

$(TESTS_OBJ): %.o: %.cpp gtest
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ -c $< -Igtest/include -I../../ -I.

$(TESTS_BIN): %: %.o ../../libascendfaiss.a gtest/make/gtest_host.a
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o $(TESTS_BIN)

.PHONY: clean run

