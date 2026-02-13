#include <stdio.h>
#include <string.h>
#include <math.h>

#include "llama.h"
#include "vectordb.h"

static float cosine_similarity(float *a, float *b, int n) {
	float dot = 0, normA = 0, normB = 0;
	for (int i = 0; i < n; i++) {
		dot += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	return dot / (sqrtf(normA) * sqrtf(normB) + 1e-8f);
}

static void embed_text(struct llama_context *ctx, const char *text, float *out) {
	llama_token tokens[512];
	const struct llama_model *model = llama_get_model(ctx);
	const struct llama_vocab *vocab = llama_model_get_vocab(model);
	int n_tokens = llama_tokenize(
			vocab,
			text,
			strlen(text),
			tokens,
			512,
			true,
			true
			);
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
	(void)db; // nothing yet (future persistence etc.)
}

void vdb_add_document(VectorDB *db, const char *text) {
	if (db->count >= VDB_MAX_DOCS) {
		printf("VectorDB full!\n");
		return;
	}

	VectorDoc *doc = &db->docs[db->count++];
	strncpy(doc->text, text, VDB_MAX_TEXT - 1);
	doc->text[VDB_MAX_TEXT - 1] = 0;

	printf("Embedding doc %d...\n", db->count);
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
