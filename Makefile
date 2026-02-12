LLAMA_DIR = llama.cpp
LLAMA_BUILD_DIR = $(LLAMA_DIR)/build

CFLAGS = -Wall -Wextra -O3 -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LDFLAGS = -L$(LLAMA_BUILD_DIR)/src -L$(LLAMA_BUILD_DIR)/ggml/src \
		  -lpthread -lm -ldl -lstdc++ -g \
		  -lllama -lggml -lggml-cpu -lggml-base


# -Wl,-rpath,$(shell pwd)/$(LLAMA_BUILD_DIR)/bin \

prompt: prompt.c
	$(CC) $(CFLAGS) prompt.c -o prompt $(LDFLAGS)

llama:
	mkdir llama.cpp/build && \
		cd llama.cpp/build && \
		cmake ../ -DBUILD_SHARED_LIBS=OFF && \
		make -j8

clean:
	-rm -f prompt
	cd llama.cpp/build && make clean
	-rm -Rf llama.cpp/build

docker:
	docker build -t promptd .
	docker run -it promptd bash
