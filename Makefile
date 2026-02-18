MAKEFLAGS += -j4
MEX_ASSURE="cc docker wget xxd"

include makext.mk

LLAMA_DIR = llama.cpp

CFLAGS = -Wall -Wextra -O3 -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LDFLAGS = -L$(LLAMA_DIR)/build/src -L$(LLAMA_DIR)/build/ggml/src \
		  -lpthread -lm -ldl -lstdc++ -g \
		  -lllama -lggml -lggml-cpu -lggml-base

help: .help

build/npc: run/system-prompt npc.c vectordb.c models.h # Build npc binary for testing
	$(CC) $(CFLAGS) npc.c vectordb.c -o npc $(LDFLAGS)

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

run/docker: .assure # Runs npc in Docker container
	docker build -t npcd .
	docker run -it npcd

run/system-prompt: .assure # Generate C style header
	xxd -i system_prompt.txt > system_prompt.h

run/clean: # Cleans up all the build artefacts
	-rm -f npc
	cd $(LLAMA_DIR)/build && make clean
	-rm -Rf $(LLAMA_DIR)/build
