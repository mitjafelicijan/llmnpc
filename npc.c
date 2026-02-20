#include "llama.h"
#include "vectordb.h"
#include "models.h"

#define NONSTD_IMPLEMENTATION
#include "nonstd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "prompts/lotr.h"

static void llama_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
	(void)level;
	(void)user_data;
	(void)text;
}

static void list_available_models() {
	printf("Model list:\n");
	ModelConfig model;
	static_foreach(ModelConfig, model, models) {
		printf(" - %s [ctx: %d, temp: %f]\n", model.name, model.n_ctx, model.temperature);
	}
}

static void show_help(const char *prog) {
	printf("Usage: %s [OPTIONS]\n", prog);
	printf("Options:\n");
	printf("  -m, --model <name>    Specify model to use (default: first model)\n");
	printf("  -e, --embed-model <name> Specify model to use for embeddings\n");
	printf("  -p, --prompt <text>   Specify prompt text (default: \"What is 2+2?\")\n");
	printf("  -c, --context <file>  Specify vector database file (.vdb)\n");
	printf("  -l, --list            Lists all available models\n");
	printf("  -v, --verbose         Enable verbose logging\n");
	printf("  -h, --help            Show this help message\n");
}

static int has_vdb_extension(const char *path) {
	size_t len = strlen(path);
	const char *ext = ".vdb";
	size_t ext_len = strlen(ext);
	if (len < ext_len) {
		return 0;
	}
	return strcmp(path + (len - ext_len), ext) == 0;
}

static void append_prompt_context(stringb *sb, const char *context, const char *question) {
	sb_append_cstr(sb, "Context:\n");
	if (context && context[0] != '\0') {
		sb_append_cstr(sb, context);
	}
	sb_append_cstr(sb, "\nQuestion:\n");
	sb_append_cstr(sb, question ? question : "");
}

static char *build_prompt(const ModelConfig *cfg, const char *system, const char *context,
	const char *question) {
	stringb full = {0};
	sb_init(&full, 0);

	switch (cfg->prompt_style) {
		case PROMPT_STYLE_T5:
			sb_append_cstr(&full, "instruction: ");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\nquestion: ");
			sb_append_cstr(&full, question ? question : "");
			sb_append_cstr(&full, "\ncontext:\n");
			if (context && context[0] != '\0') {
				sb_append_cstr(&full, context);
			}
			sb_append_cstr(&full, "\nanswer:");
			break;
		case PROMPT_STYLE_CHAT:
			sb_append_cstr(&full, "System:\n");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\nUser:\n");
			append_prompt_context(&full, context, question);
			sb_append_cstr(&full, "\nAssistant:");
			break;
		case PROMPT_STYLE_PLAIN:
		default:
			sb_append_cstr(&full, "System:\n");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\n");
			append_prompt_context(&full, context, question);
			sb_append_cstr(&full, "\nAnswer:");
			break;
	}

	return full.data;
}

static int execute_prompt_with_context(const ModelConfig *cfg, const char *prompt,
	const char *context, int n_predict) {
	if (cfg == NULL) {
		log_message(stderr, LOG_ERROR, "Model config is missing");
		return 1;
	}

	char *system_prefix = (char *)malloc(prompts_lotr_txt_len + 1);
	if (system_prefix == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to allocate system prompt");
		return 1;
	}
	memcpy(system_prefix, prompts_lotr_txt, prompts_lotr_txt_len);
	system_prefix[prompts_lotr_txt_len] = '\0';

	ggml_backend_load_all();

	struct llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = cfg->n_gpu_layers;
	model_params.use_mmap = cfg->use_mmap;

	struct llama_model *model = llama_model_load_from_file(cfg->filepath, model_params);
	if (model == NULL) {
		log_message(stderr, LOG_ERROR, "Unable to load model from %s", cfg->filepath);
		return 1;
	}

	const struct llama_vocab *vocab = llama_model_get_vocab(model);

	const char *system_text = system_prefix;
	if (strncmp(system_prefix, "System:", 7) == 0) {
		system_text = system_prefix + 7;
		while (*system_text == ' ' || *system_text == '\n' || *system_text == '\r') {
			system_text++;
		}
	}

	char *full_prompt = build_prompt(cfg, system_text, context, prompt);
	if (full_prompt == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to build prompt");
		free(system_prefix);
		llama_model_free(model);
		return 1;
	}

	int n_prompt = -llama_tokenize(vocab, full_prompt, strlen(full_prompt), NULL, 0, true, true);
	llama_token *prompt_tokens = (llama_token *)malloc((size_t)n_prompt * sizeof(llama_token));
	if (prompt_tokens == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to allocate prompt tokens");
		free(full_prompt);
		free(system_prefix);
		llama_model_free(model);
		return 1;
	}
	if (llama_tokenize(vocab, full_prompt, strlen(full_prompt), prompt_tokens, n_prompt, true, true) < 0) {
		log_message(stderr, LOG_ERROR, "Failed to tokenize prompt");
		free(full_prompt);
		free(prompt_tokens);
		free(system_prefix);
		llama_model_free(model);
		return 1;
	}

	struct llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = cfg->n_ctx;
	ctx_params.n_batch = cfg->n_batch;
	ctx_params.embeddings = cfg->embeddings;

	struct llama_context *ctx = llama_init_from_model(model, ctx_params);
	if (ctx == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to create llama_context");
		free(full_prompt);
		free(prompt_tokens);
		free(system_prefix);
		llama_model_free(model);
		return 1;
	}

	struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
	struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
	if (cfg->top_k > 0) {
		llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cfg->top_k));
	}
	if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg->top_p, 1));
	}
	if (cfg->min_p > 0.0f) {
		llama_sampler_chain_add(smpl, llama_sampler_init_min_p(cfg->min_p, 1));
	}
	llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
		cfg->repeat_last_n,
		cfg->repeat_penalty,
		cfg->freq_penalty,
		cfg->presence_penalty));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg->temperature));
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(cfg->seed));

	struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);

	if (llama_model_has_encoder(model)) {
		if (llama_encode(ctx, batch)) {
			log_message(stderr, LOG_ERROR, "Failed to encode prompt");
			llama_sampler_free(smpl);
			free(full_prompt);
			free(prompt_tokens);
			free(system_prefix);
			llama_free(ctx);
			llama_model_free(model);
			return 1;
		}

		llama_token decoder_start = llama_model_decoder_start_token(model);
		if (decoder_start == LLAMA_TOKEN_NULL) {
			decoder_start = llama_vocab_bos(vocab);
		}
		batch = llama_batch_get_one(&decoder_start, 1);
	}

	printf(">> Prompt: %s\n", prompt);
	printf(">> Response: ");
	fflush(stdout);

	int n_pos = 0;
	llama_token new_token_id;
	size_t out_cap = 256;
	size_t out_len = 0;
	char *out = (char *)malloc(out_cap);
	if (out == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to allocate output buffer");
		free(full_prompt);
		free(prompt_tokens);
		free(system_prefix);
		llama_sampler_free(smpl);
		llama_free(ctx);
		llama_model_free(model);
		return 1;
	}
	out[0] = '\0';

	while (n_pos + batch.n_tokens < n_prompt + n_predict) {
		if (llama_decode(ctx, batch)) {
			log_message(stderr, LOG_ERROR, "Failed to decode");
			break;
		}

		n_pos += batch.n_tokens;
		new_token_id = llama_sampler_sample(smpl, ctx, -1);
		if (llama_vocab_is_eog(vocab, new_token_id)) {
			break;
		}

		char buf[128];
		int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
		if (n < 0) {
			log_message(stderr, LOG_ERROR, "Failed to convert token to piece");
			break;
		}
		if (out_len == 0 && n > 0 && buf[0] == '\n') {
			batch = llama_batch_get_one(&new_token_id, 1);
			continue;
		}
		if (out_len + (size_t)n + 1 > out_cap) {
			while (out_len + (size_t)n + 1 > out_cap) {
				out_cap *= 2;
			}
			char *next = (char *)realloc(out, out_cap);
			if (next == NULL) {
				log_message(stderr, LOG_ERROR, "Failed to grow output buffer");
				break;
			}
			out = next;
		}
		memcpy(out + out_len, buf, (size_t)n);
		out_len += (size_t)n;
		out[out_len] = '\0';

		batch = llama_batch_get_one(&new_token_id, 1);
	}

	printf("%s\n", out);

	free(full_prompt);
	free(prompt_tokens);
	free(system_prefix);
	free(out);

	llama_sampler_free(smpl);
	llama_free(ctx);
	llama_model_free(model);

	return 0;
}

int main(int argc, char **argv) {
	set_log_level(LOG_DEBUG);

	const char *model_name = NULL;
	const char *prompt = NULL;
	const char *context_file = NULL;
	int verbose = 0;
	const char *embed_model_name = NULL;

	int n_predict = 0;

	static struct option long_options[] = {
		{"model", required_argument, 0, 'm'},
		{"prompt", required_argument, 0, 'p'},
		{"context", required_argument, 0, 'c'},
		{"embed-model", required_argument, 0, 'e'},
		{"list", no_argument, 0, 'l'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "m:p:c:e:lvh", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'm':
				model_name = optarg;
				break;
			case 'p':
				prompt = optarg;
				break;
			case 'c':
				context_file = optarg;
				break;
			case 'e':
				embed_model_name = optarg;
				break;
			case 'v':
				verbose = 1;
				break;
			case 'l':
				list_available_models();
				return 0;
			case 'h':
				show_help(argv[0]);
				return 0;
			default:
				fprintf(stderr, "Usage: %s [-m model] [-p prompt] [-h]\n", argv[0]);
				return 1;
		}
	}

	if (verbose == 0) {
		llama_log_set(llama_log_callback, NULL);
	}

	if (prompt == NULL) {
		log_message(stderr, LOG_ERROR, "Prompt must be provided. Exiting...");
		return 1;
	}

	if (model_name == NULL) {
		log_message(stderr, LOG_ERROR, "Model must be provided. Exiting...");
		return 1;
	}

	if (context_file == NULL) {
		log_message(stderr, LOG_ERROR, "Context .vdb file must be provided. Exiting...");
		return 1;
	}

	if (!has_vdb_extension(context_file)) {
		log_message(stderr, LOG_ERROR, "Context file must be a .vdb vector database");
		return 1;
	}

	llama_backend_init();

	const ModelConfig *cfg = NULL;
	if (model_name != NULL) {
		cfg = get_model_by_name(model_name);
		if (cfg == NULL) {
			log_message(stderr, LOG_ERROR, "Unknown model '%s'", model_name);
			llama_backend_free();
			return 1;
		}
	} else {
		cfg = &models[0];
	}

	const ModelConfig *embed_cfg = NULL;
	if (embed_model_name != NULL) {
		embed_cfg = get_model_by_name(embed_model_name);
		if (embed_cfg == NULL) {
			log_message(stderr, LOG_ERROR, "Unknown embedding model '%s'", embed_model_name);
			llama_backend_free();
			return 1;
		}
	} else if (cfg->embed_model_name != NULL) {
		embed_cfg = get_model_by_name(cfg->embed_model_name);
	}
	if (embed_cfg == NULL) {
		embed_cfg = cfg;
	}

	if (n_predict <= 0) {
		n_predict = cfg->n_predict > 0 ? cfg->n_predict : 128;
	}

	struct llama_model_params embed_params = llama_model_default_params();
	embed_params.n_gpu_layers = embed_cfg->n_gpu_layers;
	embed_params.use_mmap = embed_cfg->use_mmap;
	struct llama_model *model = llama_model_load_from_file(embed_cfg->filepath, embed_params);
	if (model == NULL) {
		log_message(stderr, LOG_ERROR, "Unable to load embedding model");
		llama_backend_free();
		return 1;
	}

	struct llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx = embed_cfg->n_ctx;
	cparams.n_batch = embed_cfg->n_batch;
	cparams.embeddings = true;

	struct llama_context *embed_ctx = llama_init_from_model(model, cparams);
	if (embed_ctx == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to create embedding context");
		llama_model_free(model);
		llama_backend_free();
		return 1;
	}

	VectorDB db = {};
	vdb_init(&db, embed_ctx);
	VectorDBErrorCode vdb_rc = vdb_load(&db, context_file);
	if (vdb_rc != VDB_SUCCESS) {
		log_message(stderr, LOG_ERROR, "Failed to load vector database %s: %s", context_file, vdb_error(vdb_rc));
		llama_free(embed_ctx);
		llama_model_free(model);
		llama_backend_free();
		return 1;
	}

	float query[VDB_EMBED_SIZE];
	int results[5];
	for (int i = 0; i < 5; i++) {
		results[i] = -1;
	}

	vdb_embed_query(&db, prompt, query);
	vdb_search(&db, query, 5, results);

	size_t context_cap = 1024;
	size_t context_len = 0;
	char *context = (char *)malloc(context_cap);
	if (context == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to allocate context buffer");
		llama_free(embed_ctx);
		llama_model_free(model);
		llama_backend_free();
		return 1;
	}
	context[0] = '\0';

	for (int i = 0; i < 5; i++) {
		if (results[i] < 0) {
			continue;
		}
		const char *text = db.docs[results[i]].text;
		char header[32];
		int header_len = snprintf(header, sizeof(header), "Snippet %d:\n", i + 1);
		size_t text_len = strlen(text);
		size_t need = context_len + (size_t)header_len + text_len + 2;
		if (need > context_cap) {
			while (need > context_cap) {
				context_cap *= 2;
			}
			char *next = (char *)realloc(context, context_cap);
			if (next == NULL) {
				log_message(stderr, LOG_ERROR, "Failed to grow context buffer");
				free(context);
				llama_free(embed_ctx);
				llama_model_free(model);
				llama_backend_free();
				return 1;
			}
			context = next;
		}
		if (header_len > 0) {
			memcpy(context + context_len, header, (size_t)header_len);
			context_len += (size_t)header_len;
		}
		memcpy(context + context_len, text, text_len);
		context_len += text_len;
		context[context_len++] = '\n';
		context[context_len] = '\0';
	}

	llama_free(embed_ctx);
	llama_model_free(model);

	int rc = execute_prompt_with_context(cfg, prompt, context, n_predict);
	free(context);
	llama_backend_free();
	return rc;
}
