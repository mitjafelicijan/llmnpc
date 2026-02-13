#include "llama.h"
#include "vectordb.h"
#include "models.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <ctype.h>

#define MAX_TOKENS 512
#define MAX_TOKEN_LEN 32

static const char *refusal_text = "I don't have that information.";

static void llama_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    (void)text;
}

static int is_stopword(const char *token, size_t len) {
    static const char *stopwords[] = {
        "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on",
        "for", "with", "and", "or", "not", "if", "then", "else", "from", "by",
        "as", "at", "it", "its", "this", "that", "these", "those", "who", "what",
        "when", "where", "why", "how", "which", "about", "into", "over", "under",
        "be", "been", "being", "do", "does", "did", "but", "so", "than"
    };
    for (size_t i = 0; i < sizeof(stopwords) / sizeof(stopwords[0]); i++) {
        if (strlen(stopwords[i]) == len && strncmp(stopwords[i], token, len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int token_exists(char tokens[MAX_TOKENS][MAX_TOKEN_LEN], int count, const char *token) {
    for (int i = 0; i < count; i++) {
        if (strcmp(tokens[i], token) == 0) {
            return 1;
        }
    }
    return 0;
}

static int collect_tokens(const char *text, char tokens[MAX_TOKENS][MAX_TOKEN_LEN]) {
    int count = 0;
    char buf[MAX_TOKEN_LEN];
    int len = 0;
    for (const unsigned char *p = (const unsigned char *)text; ; p++) {
        if (isalnum(*p)) {
            if (len < MAX_TOKEN_LEN - 1) {
                buf[len++] = (char)tolower(*p);
            }
        } else {
            if (len > 0) {
                buf[len] = '\0';
                if (len >= 4 && !is_stopword(buf, (size_t)len)) {
                    if (!token_exists(tokens, count, buf) && count < MAX_TOKENS) {
                        strncpy(tokens[count], buf, MAX_TOKEN_LEN - 1);
                        tokens[count][MAX_TOKEN_LEN - 1] = '\0';
                        count++;
                    }
                }
                len = 0;
            }
            if (*p == '\0') {
                break;
            }
        }
    }
    return count;
}

static int has_overlap(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int token_count = collect_tokens(b, tokens);
    if (token_count == 0) {
        return 0;
    }
    char buf[MAX_TOKEN_LEN];
    int len = 0;
    for (const unsigned char *p = (const unsigned char *)a; ; p++) {
        if (isalnum(*p)) {
            if (len < MAX_TOKEN_LEN - 1) {
                buf[len++] = (char)tolower(*p);
            }
        } else {
            if (len > 0) {
                buf[len] = '\0';
                if (len >= 4 && !is_stopword(buf, (size_t)len)) {
                    if (token_exists(tokens, token_count, buf)) {
                        return 1;
                    }
                }
                len = 0;
            }
            if (*p == '\0') {
                break;
            }
        }
    }
    return 0;
}

static int execute_prompt(const char *model_name, const char *prompt, const char *context, int n_predict) {
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

    if (!has_overlap(prompt, context)) {
        printf("------------ Prompt: %s\n", prompt);
        printf("------------ Response: %s\n", refusal_text);
        return 0;
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

    const char *system_prefix = "System: Answer using only the Context. If the answer is not explicitly stated in Context, respond exactly: I don't have that information.\n\n";
    const char *context_prefix = "Context:\n";
    const char *prompt_prefix = "\n\nQuestion:\n";
    const char *answer_prefix = "\n\nAnswer:\n";
    size_t context_len = context ? strlen(context) : 0;
    size_t prompt_len = strlen(prompt);
    size_t full_len = strlen(system_prefix) + strlen(context_prefix) + context_len + strlen(prompt_prefix) + prompt_len + strlen(answer_prefix) + 1;
    char *full_prompt = (char *)malloc(full_len);
    if (full_prompt == NULL) {
        fprintf(stderr, "Error: failed to allocate prompt buffer\n");
        llama_model_free(model);
        return 1;
    }
    snprintf(full_prompt, full_len, "%s%s%s%s%s", system_prefix, context_prefix, context ? context : "", prompt_prefix, prompt);
    strncat(full_prompt, answer_prefix, full_len - strlen(full_prompt) - 1);

    int n_prompt = -llama_tokenize(vocab, full_prompt, strlen(full_prompt), NULL, 0, true, true);
    llama_token *prompt_tokens = (llama_token *)malloc(n_prompt * sizeof(llama_token));
    if (llama_tokenize(vocab, full_prompt, strlen(full_prompt), prompt_tokens, n_prompt, true, true) < 0) {
        fprintf(stderr, "Error: failed to tokenize the prompt\n");
        free(full_prompt);
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
        free(full_prompt);
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
            llama_sampler_free(smpl);
            free(full_prompt);
            free(prompt_tokens);
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

    printf("------------ Prompt: %s\n", prompt);
    printf("------------ Response: ");
    fflush(stdout);

    int n_pos = 0;
    llama_token new_token_id;
    size_t out_cap = 256;
    size_t out_len = 0;
    char *out = (char *)malloc(out_cap);
    if (out == NULL) {
        fprintf(stderr, "Error: failed to allocate output buffer\n");
        free(full_prompt);
        free(prompt_tokens);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    out[0] = '\0';

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
        int stop_at = n;
        for (int i = 0; i < n; i++) {
            if (buf[i] == '\n') {
                stop_at = i;
                break;
            }
        }
        if (out_len + (size_t)stop_at + 1 > out_cap) {
            while (out_len + (size_t)stop_at + 1 > out_cap) {
                out_cap *= 2;
            }
            char *next = (char *)realloc(out, out_cap);
            if (next == NULL) {
                fprintf(stderr, "Error: failed to grow output buffer\n");
                break;
            }
            out = next;
        }
        memcpy(out + out_len, buf, (size_t)stop_at);
        out_len += (size_t)stop_at;
        out[out_len] = '\0';

        if (stop_at != n) {
            break;
        }

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    if (!has_overlap(out, context)) {
        strcpy(out, refusal_text);
        out_len = strlen(out);
    }

    printf("%s\n", out);

    free(full_prompt);
    free(prompt_tokens);
    free(out);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

static char *generate_context(const char *model_name, const char *context_file, const char *prompt) {
    FILE *context_fp = fopen(context_file, "r");
    if (context_fp == NULL) {
        fprintf(stderr, "Error: unable to open context file %s\n", context_file);
        return NULL;
    }

    llama_backend_init();

    const model_config *cfg = NULL;
    if (model_name != NULL) {
        cfg = get_model_by_name(model_name);
        if (cfg == NULL) {
            fprintf(stderr, "Error: unknown model '%s'\n", model_name);
            fclose(context_fp);
            llama_backend_free();
            return NULL;
        }
    } else {
        cfg = &models[0];
    }

    /* struct llama_model *model = llama_load_model_from_file(cfg->filepath, llama_model_default_params()); */
    struct llama_model *model = llama_model_load_from_file(cfg->filepath, llama_model_default_params());
    if (model == NULL) {
        fprintf(stderr, "Error: unable to load embedding model\n");
        fclose(context_fp);
        llama_backend_free();
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;

    /* struct llama_context *embed_ctx = llama_new_context_with_model(model, cparams); */
    struct llama_context *embed_ctx = llama_init_from_model(model, cparams);
    if (embed_ctx == NULL) {
        fprintf(stderr, "Error: failed to create embedding context\n");
        llama_model_free(model);
        fclose(context_fp);
        llama_backend_free();
        return NULL;
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

    float query[VDB_EMBED_SIZE];
    int results[3];

    vdb_embed_query(&db, prompt, query);
    vdb_search(&db, query, 3, results);

    size_t context_cap = 1024;
    size_t context_len = 0;
    char *context = (char *)malloc(context_cap);
    if (context == NULL) {
        fprintf(stderr, "Error: failed to allocate context buffer\n");
        fclose(context_fp);
        llama_free(embed_ctx);
        llama_model_free(model);
        llama_backend_free();
        return NULL;
    }
    context[0] = '\0';

    for (int i = 0; i < 3; i++) {
        if (results[i] < 0) {
            continue;
        }
        const char *text = db.docs[results[i]].text;
        size_t text_len = strlen(text);
        size_t need = context_len + text_len + 2;
        if (need > context_cap) {
            while (need > context_cap) {
                context_cap *= 2;
            }
            char *next = (char *)realloc(context, context_cap);
            if (next == NULL) {
                fprintf(stderr, "Error: failed to grow context buffer\n");
                free(context);
                fclose(context_fp);
                llama_free(embed_ctx);
                llama_model_free(model);
                llama_backend_free();
                return NULL;
            }
            context = next;
        }
        memcpy(context + context_len, text, text_len);
        context_len += text_len;
        context[context_len++] = '\n';
        context[context_len] = '\0';
    }

    fclose(context_fp);
    llama_free(embed_ctx);
    llama_model_free(model);
    llama_backend_free();

    return context;
}

static void show_help(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:\n");
    printf("  -m, --model <name>    Specify model to use (default: first model)\n");
    printf("  -p, --prompt <text>   Specify prompt text (default: \"What is 2+2?\")\n");
    printf("  -c, --context <text>  Specify context file\n");
    printf("  -v, --verbose         Enable verbose logging\n");
    printf("  -h, --help            Show this help message\n");
}

int main(int argc, char **argv) {
    const char *model_name = NULL;
    const char *prompt = NULL;
    const char *context_file = NULL;
	int verbose = 0;
    
    int n_predict = 64;

    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"prompt", required_argument, 0, 'p'},
        {"context", required_argument, 0, 'c'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:p:c:vh", long_options, &option_index)) != -1) {
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
            case 'v':
                verbose = 1;
                break;
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
		printf("Prompt must be provided. Exiting...");
		return 1;
    }

    if (context_file == NULL) {
		printf("Context file must be provided. Exiting...");
		return 1;
    }

    char *context = generate_context(model_name, context_file, prompt);
    if (context == NULL) {
        return 1;
    }

    int rc = execute_prompt(model_name, prompt, context, n_predict);
    free(context);
    return rc;
}
