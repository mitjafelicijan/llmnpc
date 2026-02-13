#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "llama.h"
#include "vectordb.h"
#include "nonstd.h"

#define VDB_MAGIC 0x31424456u /* "VDB1" */
#define VDB_VERSION 1u

typedef struct {
	uint32_t magic;
	uint32_t version;
	uint32_t embed_size;
	uint32_t max_text;
	uint32_t count;
} VdbFileHeader;

static float cosine_similarity(float *a, float *b, int n) {
	float dot = 0, norm_a = 0, norm_b = 0;
	for (int i = 0; i < n; i++) {
		dot += a[i] * b[i];
		norm_a += a[i] * a[i];
		norm_b += b[i] * b[i];
	}
	return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

static void embed_text(struct llama_context *ctx, const char *text, float *out) {
	llama_token tokens[512];
	const struct llama_model *model = llama_get_model(ctx);
	const struct llama_vocab *vocab = llama_model_get_vocab(model);
	int n_tokens = llama_tokenize(vocab, text, strlen(text), tokens, 512, true, true);
	if (n_tokens < 0) {
		return;
	}

	struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
	llama_decode(ctx, batch);

	const float *emb = llama_get_embeddings(ctx);
	memcpy(out, emb, sizeof(float) * VDB_EMBED_SIZE);

}

void vdb_init(VectorDB *db, struct llama_context *embed_ctx) {
	memset(db, 0, sizeof(VectorDB));
	db->embed_ctx = embed_ctx;
}

void vdb_free(VectorDB *db) {
	(void)db;
}

void vdb_add_document(VectorDB *db, const char *text) {
	if (db->count >= VDB_MAX_DOCS) {
		log_message(stdout, LOG_INFO, "Vector database full");
		return;
	}

	VectorDoc *doc = &db->docs[db->count++];
	strncpy(doc->text, text, VDB_MAX_TEXT - 1);
	doc->text[VDB_MAX_TEXT - 1] = 0;

	log_message(stdout, LOG_INFO, "Embedding doc %d...", db->count);
	embed_text(db->embed_ctx, text, doc->embedding);
}

void vdb_embed_query(VectorDB *db, const char *text, float *out_embedding) {
	embed_text(db->embed_ctx, text, out_embedding);
}

void vdb_search(VectorDB *db, float *query, int top_k, int *results) {
	float best_scores[top_k];
	for (int i = 0; i < top_k; i++) {
		best_scores[i] = -1.0f;
		results[i] = -1;
	}

	for (int i = 0; i < db->count; i++) {
		float score = cosine_similarity(query, db->docs[i].embedding, VDB_EMBED_SIZE);

		for (int j = 0; j < top_k; j++) {
			if (score > best_scores[j]) {
				for (int k = top_k - 1; k > j; k--) {
					best_scores[k] = best_scores[k - 1];
					results[k] = results[k - 1];
				}
				best_scores[j] = score;
				results[j] = i;
				break;
			}
		}
	}
}

int vdb_save(const VectorDB *db, const char *path) {
	FILE *fp = fopen(path, "wb");
	if (!fp) {
		return 1;
	}

	VdbFileHeader header = {
		.magic = VDB_MAGIC,
		.version = VDB_VERSION,
		.embed_size = VDB_EMBED_SIZE,
		.max_text = VDB_MAX_TEXT,
		.count = (uint32_t)db->count,
	};

	if (fwrite(&header, sizeof(header), 1, fp) != 1) {
		fclose(fp);
		return 2;
	}

	if (db->count > 0) {
		size_t wrote = fwrite(db->docs, sizeof(VectorDoc), (size_t)db->count, fp);
		if (wrote != (size_t)db->count) {
			fclose(fp);
			return 3;
		}
	}

	if (fclose(fp) != 0) {
		return 4;
	}

	return 0;
}

int vdb_load(VectorDB *db, const char *path) {
	struct llama_context *ctx = db->embed_ctx;
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		return -1;
	}

	VdbFileHeader header = {0};
	if (fread(&header, sizeof(header), 1, fp) != 1) {
		fclose(fp);
		return -2;
	}

	if (header.magic != VDB_MAGIC || header.version != VDB_VERSION) {
		fclose(fp);
		return -3;
	}

	if (header.embed_size != VDB_EMBED_SIZE || header.max_text != VDB_MAX_TEXT) {
		fclose(fp);
		return -4;
	}

	if (header.count > VDB_MAX_DOCS) {
		fclose(fp);
		return -5;
	}

	memset(db, 0, sizeof(VectorDB));
	db->embed_ctx = ctx;
	db->count = (int)header.count;

	if (db->count > 0) {
		size_t read = fread(db->docs, sizeof(VectorDoc), (size_t)db->count, fp);
		if (read != (size_t)db->count) {
			fclose(fp);
			return -6;
		}
	}

	if (fclose(fp) != 0) {
		return -7;
	}

	return 0;
}
