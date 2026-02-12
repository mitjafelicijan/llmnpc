# llmnpc

A command-line LLM inference tool powered by
[llama.cpp](https://github.com/ggerganov/llama.cpp) for testing how/if NPC's
could use LLM's.

## Building

### Prerequisites

- C compiler (gcc/clang)
- CMake
- Docker (optional, for containerized use of binaries)

### Build Steps

1. Build llama.cpp libraries:
   ```bash
   make llamacpp
   ```

2. Download models
   ```bash
   make fetchmodels
   ```

3. Build the prompt binary:
   ```bash
   make prompt
   ```

## Usage

```bash
./prompt -p "Your prompt here"
./prompt -m flan-t5-small -p "What is machine learning?"
```

### Options

| Flag | Description |
|------|-------------|
| `-m, --model` | Model to use (default: first model in config) |
| `-p, --prompt` | Prompt text (required) |
| `-h, --help` | Show help message |

## Models

Configure models in `models.h`. The default model is `flan-t5-small`, expecting a GGUF file at `models/flan-t5-small.F16.gguf`.

## Docker

```bash
make docker
```

This builds a Docker image and drops you into a shell with the prompt binary and models available at `/app/`.

## Cleaning

```bash
make clean
```

## Reading material

- https://www.tinyllm.org/
