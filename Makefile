MAKEFLAGS += -j4
MEX_ASSURE="cc docker wget xxd"

include makext.mk

LLAMA_DIR = llama.cpp

CFLAGS = -Wall -Wextra -O3 -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LDFLAGS = -L$(LLAMA_DIR)/build/src -L$(LLAMA_DIR)/build/ggml/src \
		  -lpthread -lm -ldl -lstdc++ -g \
		  -lllama -lggml -lggml-cpu -lggml-base

PROMPT_TXT := $(wildcard prompts/*.txt)
PROMPT_HEADERS := $(PROMPT_TXT:.txt=.h)

MAP_TXT := $(wildcard maps/*.txt)
MAP_HEADERS := $(MAP_TXT:.txt=.h)

CORPUS_TXT := $(wildcard corpus/*.txt)
CORPUS_VDB := $(CORPUS_TXT:.txt=.vdb)

help: .help

build/llama.cpp: .assure # Build llama.cpp libraries
	mkdir $(LLAMA_DIR)/build && \
		cd $(LLAMA_DIR)/build && \
		cmake ../ -DBUILD_SHARED_LIBS=OFF && \
		make -j8

build/context: context.c vectordb.c models.h # Build context binary for testing
	$(CC) $(CFLAGS) context.c vectordb.c -o context $(LDFLAGS)

build/npc: build/prompts npc.c vectordb.c models.h # Build npc binary for testing
	$(CC) $(CFLAGS) npc.c vectordb.c -o npc $(LDFLAGS)

build/game: build/prompts build/maps game.c vectordb.c models.h # Build npc binary for testing
	$(CC) $(CFLAGS) game.c vectordb.c -o game $(LDFLAGS)

build/prompts: $(PROMPT_HEADERS) # Generate prompts in C style header

build/maps: $(MAP_HEADERS) # Generate maps in  C style header

build/corpus: $(CORPUS_VDB) # Build vector DBs for all corpuses

run/fetch-models: .assure # Fetch GGUF models
	-mkdir -p models
	cd models && wget -nc -i ../models.txt

run/docker: .assure # Runs npc in Docker container
	docker build -t npcd .
	docker run -it npcd

run/clean: # Cleans up all the build artefacts
	-rm -f npc
	cd $(LLAMA_DIR)/build && make clean
	-rm -Rf $(LLAMA_DIR)/build

prompts/%.h: prompts/%.txt .assure
	xxd -i $< > $@

maps/%.h: maps/%.txt .assure
	xxd -i $< > $@

corpus/%.vdb: corpus/%.txt build/context
	./context -m qwen3 -i $< -o $@
