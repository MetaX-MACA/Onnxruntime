CXX = mxcc
MXCCFLAGS     := -x maca
MACA_PATH = /opt/maca

CFLAGES += -DUSE_MACA

INCLUDE_PATH = -I/opt/maca-ai/onnxruntime-maca/include -I/opt/maca/include
LIBRARY_PATH = -L/opt/maca-ai/onnxruntime-maca/lib

SRCS = custom_op_library.cc
OBJS = $(SRCS:.cpp=.o)
MACA_TARGET = libcustom_op_library_maca.so
CPU_TARGET = libcustom_op_library_cpu.so

LIBS = -lmxkw -lmcruntime -lonnxruntime

all: $(MACA_TARGET) $(CPU_TARGET)

$(MACA_TARGET): $(OBJS)
	$(CXX) $(INCLUDE_PATH) $(MXCCFLAGS) $(CFLAGES) -shared -o $@ $^ $(LIBRARY_PATH) --maca-path=$(MACA_PATH) -fPIC

$(CPU_TARGET): $(OBJS)
	$(CXX) $(INCLUDE_PATH) $(MXCCFLAGS) -shared -o $@ $^ $(LIBRARY_PATH) --maca-path=$(MACA_PATH) -fPIC
