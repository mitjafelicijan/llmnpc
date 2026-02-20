An experiment using tiny LLMs as NPCs that could be embedded into the game.

Embed models into the game, build a simple vector database from text, embed 
prompts, retrieve top‑k by cosine similarity, and feed context into tiny 
CPU LLMs for NPC interactions.

**No external API calls.** Everything is local, directly using GGUF models 
and [llama.cpp](https://github.com/ggml-org/llama.cpp) for inference.

https://github.com/user-attachments/assets/863b75eb-0da7-4235-8112-f00bc82d81f6

> [!NOTE]
> This project is just for fun, to see how LLMs would fare as NPCs. Because of
> the non-deterministic nature of LLMs, the results vary and are often quite
> funny. A lot of tweaking would be needed to make this really useful in real
> games, but not impossible.

Goals of the experiment:

- Have LLM be run only on CPU, this is why small LLMs have been chosen in this
  experiment, so they can be used in other games.
- To produce a simple C library that can be reused elsewhere.
- Test existing small and tiny LLMs and provide some useful results on how they
  behave.

## Getting started

1. Build dependencies and binaries:
   ```bash
   make build/llama.cpp
   make run/fetch-models
   make build/context
   make build/game
   ```

2. Build a vector context database:
   ```bash
   build/corpus
   ```

3. Run the game:
   ```bash
   ./game -m phi-4-mini-instruct -e qwen3
   ```

## Building

### Prerequisites

- C compiler (gcc/clang)
- CMake
- Docker (optional, for containerized use of binaries)

### Build Steps

1. Build llama.cpp libraries:
   ```bash
   make build/llama.cpp
   ```

2. Download models:
   ```bash
   make run/fetch-models
   ```

3. Build binaries:
   ```bash
   make build/context
   make build/prompts
   make build/npc
   make build/game
   ```

## Usage

### Build a vector context database

`context` reads a text file (one document per line), embeds each line, and
produces a binary vector database file. For best results, use a dedicated
embedding model (for example, `qwen3`) even if you generate answers with a
different model.

```bash
./context -m qwen3 -i corpus/map1_keldor.txt -o corpus/map1_keldor.vdb
```

### Run an NPC query with retrieved context

`npc` loads a vector database, embeds the prompt, selects the top 5 matching
lines by cosine similarity, and runs the NPC system prompt against that context.
You can pass a separate embedding model with `-e`/`--embed-model`.

```bash
./npc -m phi-4-mini-instruct -e qwen3 -p "Who is Keldor?" -c corpus/map1_keldor.vdb
./npc -m qwen3 -e qwen3 -p "What does Keldor believe about the marsh lights?" -c corpus/map1_keldor.vdb
```

### Run the game

The game uses the same models and retrieval pipeline, with short NPC replies.

```bash
./game -m phi-4-mini-instruct -e qwen3
```

### context options

| Flag | Description |
|------|-------------|
| `-m, --model` | Embedding model to use (default: first model in config) |
| `-i, --in` | Input context text file (required) |
| `-o, --out` | Output vector database file (required) |
| `-l, --list` | List available models |
| `-v, --verbose` | Enable llama.cpp logging |
| `-h, --help` | Show help message |

### npc options

| Flag | Description |
|------|-------------|
| `-m, --model` | Model to use (required) |
| `-e, --embed-model` | Embedding model to use (optional) |
| `-p, --prompt` | Prompt text (required) |
| `-c, --context` | Context vector database file (.vdb) (required) |
| `-l, --list` | List available models |
| `-v, --verbose` | Enable llama.cpp logging |
| `-h, --help` | Show help message |

### game options

| Flag | Description |
|------|-------------|
| `-m, --model` | Model to use (default: first model in config) |
| `-e, --embed-model` | Embedding model to use (optional) |
| `-v, --verbose` | Enable llama.cpp logging |
| `-h, --help` | Show help message |

## Models

Configure models in `models.h`. The default model is the first entry in the
`models` array; each entry points at a local GGUF file under `models/`.

## Vector database format

`context` produces a binary file with a fixed header and a contiguous list of
documents. The header includes a magic value, version, embedding size, maximum
text length, and document count. Each document stores the original text (fixed
size `VDB_MAX_TEXT`) and its embedding (`VDB_EMBED_SIZE`).

## Docker

```bash
make run/docker
```

Builds a Docker image and runs an interactive shell with the binaries and
models under `/app/`.

## Cleaning

```bash
make run/clean
```

## Reading material

- https://www.tinyllm.org/
- https://en.wikipedia.org/wiki/Cosine_similarity
