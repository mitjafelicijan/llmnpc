An experiment using tiny LLMs as NPCs that could be embedded into the game.

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
   ```

## Usage

### Build a vector context database

`context` reads a text file (one document per line), embeds each line, and
produces a binary vector database file.

```bash
./context -i corpus/lotr.txt -o corpus/lotr.vdb
./context -m flan-t5-small -i corpus/lotr.txt -o corpus/lotr.vdb
```

### Run an NPC query with retrieved context

`npc` loads a vector database, embeds the prompt, selects the top 3 matching
lines by cosine similarity, and runs the NPC system prompt against that context.

```bash
./npc -m flan-t5-small -p "Who is Gandalf?" -c corpus/lotr.vdb
./npc -m flan-t5-small -p "Who is Frodo?" -c corpus/lotr.vdb
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
| `-p, --prompt` | Prompt text (required) |
| `-c, --context` | Context vector database file (.vdb) (required) |
| `-l, --list` | List available models |
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
