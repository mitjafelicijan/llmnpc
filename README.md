# llmnpc

Command-line tooling for NPC-focused LLM experiments with lightweight context
retrieval, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

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
   make build/npc
   ```

## Usage

### Build a vector context database

`context` reads a text file (one document per line), embeds each line, and
produces a binary vector database file.

```bash
./context -i context.txt -o context.vdb
./context -m flan-t5-small -i context.txt -o context.vdb
```

### Run an NPC query with retrieved context

`npc` loads a vector database, embeds the prompt, selects the top 3 matching
lines by cosine similarity, and runs the NPC system prompt against that context.

```bash
./npc -m flan-t5-small -p "Who is Gandalf?" -c context.vdb
./npc -m flan-t5-small -p "Who is Frodo?" -c context.vdb
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
