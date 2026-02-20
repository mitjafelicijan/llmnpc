#ifndef MODELS_H
#define MODELS_H

#include "llama.h"
#include <stddef.h>
#include <string.h>

typedef enum {
	PROMPT_STYLE_PLAIN = 0,
	PROMPT_STYLE_CHAT = 1,
	PROMPT_STYLE_T5 = 2,
} PromptStyle;

typedef struct {
	const char *name;
	const char *filepath;
	const char *embed_model_name;
	int n_gpu_layers;
	bool use_mmap;
	int n_ctx;
	int n_batch;
	bool embeddings;
	int n_predict;
	float temperature;
	float min_p;
	int top_k;
	float top_p;
	int repeat_last_n;
	float repeat_penalty;
	float freq_penalty;
	float presence_penalty;
	uint32_t seed;
	PromptStyle prompt_style;
} ModelConfig;

ModelConfig models[] = {
	{
		.name = "qwen3",
		.filepath = "models/Qwen3-0.6B-UD-Q6_K_XL.gguf",
		.embed_model_name = "qwen3",
		.n_gpu_layers = 0,
		.use_mmap = false,
		.n_ctx = 2048,
		.n_batch = 4096,
		.embeddings = false,
		.n_predict = 128,
		.temperature = 0.6f,
		.min_p = 0.05f,
		.top_k = 40,
		.top_p = 0.9f,
		.repeat_last_n = 64,
		.repeat_penalty = 1.1f,
		.freq_penalty = 0.0f,
		.presence_penalty = 0.0f,
		.seed = LLAMA_DEFAULT_SEED,
		.prompt_style = PROMPT_STYLE_CHAT,
	},
	{
		.name = "tinyllama-1.1b",
		.filepath = "models/tinyllama-1.1b.gguf",
		.embed_model_name = "qwen3",
		.n_gpu_layers = 0,
		.use_mmap = false,
		.n_ctx = 2048,
		.n_batch = 4096,
		.embeddings = false,
		.n_predict = 128,
		.temperature = 0.7f,
		.min_p = 0.05f,
		.top_k = 40,
		.top_p = 0.9f,
		.repeat_last_n = 64,
		.repeat_penalty = 1.1f,
		.freq_penalty = 0.0f,
		.presence_penalty = 0.0f,
		.seed = LLAMA_DEFAULT_SEED,
		.prompt_style = PROMPT_STYLE_PLAIN,
	},
	{
		.name = "tinyllama-1",
		.filepath = "models/TinyLlama-1.1B-intermediate-step-1431k-3T-Q2_K.gguf",
		.embed_model_name = "qwen3",
		.n_gpu_layers = 0,
		.use_mmap = false,
		.n_ctx = 2048,
		.n_batch = 4096,
		.embeddings = false,
		.n_predict = 128,
		.temperature = 0.7f,
		.min_p = 0.05f,
		.top_k = 40,
		.top_p = 0.9f,
		.repeat_last_n = 64,
		.repeat_penalty = 1.1f,
		.freq_penalty = 0.0f,
		.presence_penalty = 0.0f,
		.seed = LLAMA_DEFAULT_SEED,
		.prompt_style = PROMPT_STYLE_PLAIN,
	},
	{
		.name = "flan-t5-small",
		.filepath = "models/flan-t5-small.F16.gguf",
		.embed_model_name = "qwen3",
		.n_gpu_layers = 0,
		.use_mmap = false,
		.n_ctx = 512,
		.n_batch = 512,
		.embeddings = false,
		.n_predict = 128,
		.temperature = 0.2f,
		.min_p = 0.05f,
		.top_k = 40,
		.top_p = 0.9f,
		.repeat_last_n = 64,
		.repeat_penalty = 1.1f,
		.freq_penalty = 0.0f,
		.presence_penalty = 0.0f,
		.seed = LLAMA_DEFAULT_SEED,
		.prompt_style = PROMPT_STYLE_T5,
	},
	{
		.name = "phi-4-mini-instruct",
		.filepath = "models/Phi-4-mini-instruct.Q2_K.gguf",
		.embed_model_name = "qwen3",
		.n_gpu_layers = 0,
		.use_mmap = false,
		.n_ctx = 4096,
		.n_batch = 4096,
		.embeddings = false,
		.n_predict = 128,
		.temperature = 0.6f,
		.min_p = 0.05f,
		.top_k = 40,
		.top_p = 0.9f,
		.repeat_last_n = 64,
		.repeat_penalty = 1.1f,
		.freq_penalty = 0.0f,
		.presence_penalty = 0.0f,
		.seed = LLAMA_DEFAULT_SEED,
		.prompt_style = PROMPT_STYLE_CHAT,
	},
};

const ModelConfig *get_model_by_name(const char *name) {
	for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); i++) {
		if (models[i].name != NULL && strcmp(models[i].name, name) == 0) {
			return &models[i];
		}
	}
	return NULL;
}

#endif
