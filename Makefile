MAKEFLAGS += -j4
MEX_ASSURE="cc docker wget"

include makext.mk

LLAMA_DIR = llama.cpp

CFLAGS = -Wall -Wextra -O3 -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LDFLAGS = -L$(LLAMA_DIR)/build/src -L$(LLAMA_DIR)/build/ggml/src \
		  -lpthread -lm -ldl -lstdc++ -g \
		  -lllama -lggml -lggml-cpu -lggml-base

help: .help

build/prompt: prompt.c vectordb.c models.h # Build prompt binary for testing
	$(CC) $(CFLAGS) prompt.c vectordb.c -o prompt $(LDFLAGS)

build/context: context.c vectordb.c models.h # Build context binary for testing
	$(CC) $(CFLAGS) context.c vectordb.c -o context $(LDFLAGS)

build/llama.cpp: .assure # Build llama.cpp libraries
	mkdir $(LLAMA_DIR)/build && \
		cd $(LLAMA_DIR)/build && \
		cmake ../ -DBUILD_SHARED_LIBS=OFF && \
		make -j8

run/fetch-models: .assure # Fetch GGUF models
	-mkdir -p models
	cd models && wget -nc -i ../models.txt

run/docker: .assure # Runs prompt in Docker container
	docker build -t promptd .
	docker run -it promptd

run/clean: # Cleans up all the build artefacts
	-rm -f prompt
	cd $(LLAMA_DIR)/build && make clean
	-rm -Rf $(LLAMA_DIR)/build
