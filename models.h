#ifndef MODELS_H
#define MODELS_H

#include "llama.h"
#include <stddef.h>
#include <string.h>

typedef struct {
    const char *name;
    const char *filepath;
    int n_gpu_layers;
    bool use_mmap;
    int n_ctx;
    int n_batch;
    bool embeddings;
    float temperature;
    float min_p;
    uint32_t seed;
} model_config;

model_config models[] = {
    {
        .name = "flan-t5-small",
        .filepath = "models/flan-t5-small.F16.gguf",
        .n_gpu_layers = 0,
        .use_mmap = false,
        .n_ctx = 512,
        .n_batch = 512,
        .embeddings = false,
        .temperature = 0.8f,
        .min_p = 0.05f,
        .seed = LLAMA_DEFAULT_SEED,
    },
    {
        .name = "phi-4-mini-instruct",
        .filepath = "models/Phi-4-mini-instruct.Q2_K.gguf",
        .n_gpu_layers = 0,
        .use_mmap = false,
        .n_ctx = 131072,
        .n_batch = 4096,
        .embeddings = false,
        .temperature = 0.8f,
        .min_p = 0.05f,
        .seed = LLAMA_DEFAULT_SEED,
    },
    {
        .name = "tinyllama-1",
        .filepath = "models/TinyLlama-1.1B-intermediate-step-1431k-3T-Q2_K.gguf",
        .n_gpu_layers = 0,
        .use_mmap = false,
        .n_ctx = 2048,
        .n_batch = 4096,
        .embeddings = false,
        .temperature = 0.8f,
        .min_p = 0.05f,
        .seed = LLAMA_DEFAULT_SEED,
    },
};

const model_config *get_model_by_name(const char *name) {
    for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); i++) {
        if (models[i].name != NULL && strcmp(models[i].name, name) == 0) {
            return &models[i];
        }
    }
    return NULL;
}

#endif
