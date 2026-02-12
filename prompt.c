#include "llama.h"
#include "models.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

static void show_help(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:\n");
    printf("  -m, --model <name>    Specify model to use (default: first model)\n");
    printf("  -p, --prompt <text>   Specify prompt text (default: \"What is 2+2?\")\n");
    printf("  -h, --help            Show this help message\n");
}

int main(int argc, char **argv) {
    const char *model_name = NULL;
    const char *prompt = NULL;
    
    int n_predict = 64;

    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"prompt", required_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:p:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                model_name = optarg;
                break;
            case 'p':
                prompt = optarg;
                break;
            case 'h':
                show_help(argv[0]);
                return 0;
            default:
                fprintf(stderr, "Usage: %s [-m model] [-p prompt] [-h]\n", argv[0]);
                return 1;
        }
    }

    if (prompt == NULL) {
		printf("Prompt must be provided. Exiting...");
		return 1;
    }

    const model_config *cfg = NULL;
    if (model_name != NULL) {
        cfg = get_model_by_name(model_name);
        if (cfg == NULL) {
            fprintf(stderr, "Error: unknown model '%s'\n", model_name);
            return 1;
        }
    } else {
        cfg = &models[0];
    }

    ggml_backend_load_all();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg->n_gpu_layers;
    model_params.use_mmap = cfg->use_mmap;

    struct llama_model *model = llama_model_load_from_file(cfg->filepath, model_params);
    if (model == NULL) {
        fprintf(stderr, "Error: unable to load model from %s\n", cfg->filepath);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
    llama_token *prompt_tokens = (llama_token *)malloc(n_prompt * sizeof(llama_token));
    if (llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens, n_prompt, true, true) < 0) {
        fprintf(stderr, "Error: failed to tokenize the prompt\n");
        free(prompt_tokens);
        llama_model_free(model);
        return 1;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg->n_ctx;
    ctx_params.n_batch = cfg->n_batch;
    ctx_params.embeddings = cfg->embeddings;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "Error: failed to create the llama_context\n");
        free(prompt_tokens);
        llama_model_free(model);
        return 1;
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg->temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(cfg->min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(cfg->seed));

    struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "Error: failed to encode prompt\n");
            return 1;
        }

        llama_token decoder_start = llama_model_decoder_start_token(model);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    printf("Prompt: %s\n", prompt);
    printf("Response: ");
    fflush(stdout);

    int n_pos = 0;
    llama_token new_token_id;

    while (n_pos + batch.n_tokens < n_prompt + n_predict) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Error: failed to decode\n");
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
            fprintf(stderr, "Error: failed to convert token to piece\n");
            break;
        }
        printf("%.*s", n, buf);
        fflush(stdout);

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    printf("\n");

    free(prompt_tokens);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
