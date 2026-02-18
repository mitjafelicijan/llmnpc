#include "llama.h"
#include "vectordb.h"
#include "models.h"

#define NONSTD_IMPLEMENTATION
#include "nonstd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

static void llama_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
	(void)level;
	(void)user_data;
	(void)text;
}

void list_available_models() {
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
	printf("  -i, --in <file>       Specify input context file\n");
	printf("  -o, --out <file>      Specify output vector database file\n");
	printf("  -l, --list            Lists all available models\n");
	printf("  -v, --verbose         Enable verbose logging\n");
	printf("  -h, --help            Show this help message\n");
}

int main(int argc, char **argv) {
	set_log_level(LOG_DEBUG);

	const char *model_name = NULL;
	const char *in_file = NULL;
	const char *out_file = NULL;
	int list_models = 0;
	int verbose = 0;

	static struct option long_options[] = {
		{"model", required_argument, 0, 'm'},
		{"in", required_argument, 0, 'i'},
		{"out", required_argument, 0, 'o'},
		{"list", no_argument, 0, 'l'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "m:i:o:lvh", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'm':
				model_name = optarg;
				break;
			case 'i':
				in_file = optarg;
				break;
			case 'o':
				out_file = optarg;
				break;
			case 'l':
				list_models = 1;
				break;
			case 'v':
				verbose = 1;
				break;
			case 'h':
				show_help(argv[0]);
				return 0;
			default:
				fprintf(stderr, "Usage: %s [-m model] [-i file] [-o file] [-lvh]\n", argv[0]);
				return 1;
		}
	}

	if (verbose == 0) {
		llama_log_set(llama_log_callback, NULL);
	}

	if (list_models == 1) {
		list_available_models();
		return 0;
	}

	if (in_file == NULL) {
		log_message(stderr, LOG_ERROR, "Input context file must be provided. Exiting...");
		return 1;
	}

	if (out_file == NULL) {
		log_message(stderr, LOG_ERROR, "Output vector context file must be provided. Exiting...");
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

	struct llama_model *model = llama_model_load_from_file(cfg->filepath, llama_model_default_params());
	if (model == NULL) {
		log_message(stderr, LOG_ERROR, "Unable to load embedding model");
		llama_backend_free();
		return 1;
	}

	struct llama_context_params cparams = llama_context_default_params();
	cparams.embeddings = true;

	struct llama_context *embed_ctx = llama_init_from_model(model, cparams);
	if (embed_ctx == NULL) {
		log_message(stderr, LOG_ERROR, "Failed to create embedding context");
		llama_model_free(model);
		llama_backend_free();
		return 1;
	}

	FILE *context_fp = fopen(in_file, "r");
	if (context_fp == NULL) {
		log_message(stderr, LOG_ERROR, "Unable to open context file %s", in_file);
		return 1;
	}

	VectorDB db;
	vdb_init(&db, embed_ctx);

	char line[1024];
	while (fgets(line, sizeof(line), context_fp) != NULL) {
		size_t len = strlen(line);
		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
			line[len - 1] = '\0';
			len--;
		}
		if (len == 0) {
			continue;
		}
		vdb_add_document(&db, line);
	}

	if (vdb_save(&db, out_file) > 0) {
		log_message(stderr, LOG_ERROR, "Something went wrong saving file %s", out_file);
		fclose(context_fp);
		return 1;
	}

	log_message(stdout, LOG_INFO, "Context vector database file %s successfully written", out_file);
	fclose(context_fp);
	return 0;
}
