#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "llama.h"
#include "vectordb.h"

// Returns cosine similarity in range [-1, 1] (approx).
// https://en.wikipedia.org/wiki/Cosine_similarity
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
	llama_token tokens[VDB_TOKENS];
	const struct llama_model *model = llama_get_model(ctx);
	const struct llama_vocab *vocab = llama_model_get_vocab(model);
	int n_tokens = llama_tokenize(vocab, text, strlen(text), tokens, VDB_TOKENS, true, true);
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
		printf("Vector database full\n");
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

VectorDBErrorCode vdb_save(const VectorDB *db, const char *path) {
	FILE *fp = fopen(path, "wb");
	if (!fp) {
		return VDB_OPEN_ERR;
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
		return VDB_HEADER_WRITE_ERR;
	}

	if (db->count > 0) {
		size_t wrote = fwrite(db->docs, sizeof(VectorDoc), (size_t)db->count, fp);
		if (wrote != (size_t)db->count) {
			fclose(fp);
			return VDB_DOC_WRITE_ERR;
		}
	}

	if (fclose(fp) != 0) {
		return VDB_CLOSE_ERR;
	}

	return VDB_SUCCESS;
}

VectorDBErrorCode vdb_load(VectorDB *db, const char *path) {
	struct llama_context *ctx = db->embed_ctx;
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		int open_err = errno;
		fprintf(stderr, "vdb_load: open failed: %s\n", strerror(open_err));
		return VDB_OPEN_ERR;
	}

	VdbFileHeader header = {0};
	if (fread(&header, sizeof(header), 1, fp) != 1) {
		int read_err = errno;
		fprintf(stderr, "vdb_load: header read failed: %s\n", strerror(read_err));
		fclose(fp);
		return VDB_HEADER_READ_ERR;
	}

	if (header.magic != VDB_MAGIC || header.version != VDB_VERSION) {
		fclose(fp);
		return VDB_MAGIC_MISMATCH_ERR;
	}

	if (header.embed_size != VDB_EMBED_SIZE || header.max_text != VDB_MAX_TEXT) {
		fclose(fp);
		return VDB_EMBED_MISMATCH_ERR;
	}

	if (header.count > VDB_MAX_DOCS) {
		fclose(fp);
		return VDB_COUNT_TOO_LARGE_ERR;
	}

	memset(db, 0, sizeof(VectorDB));
	db->embed_ctx = ctx;
	db->count = (int)header.count;

	if (db->count > 0) {
		size_t read = fread(db->docs, sizeof(VectorDoc), (size_t)db->count, fp);
		if (read != (size_t)db->count) {
			int read_err = errno;
			fprintf(stderr, "vdb_load: doc read failed: %s\n", strerror(read_err));
			fclose(fp);
			return VDB_DOC_READ_ERR;
		}
	}

	if (fclose(fp) != 0) {
		int close_err = errno;
		fprintf(stderr, "vdb_load: close failed: %s\n", strerror(close_err));
		return VDB_CLOSE_ERR;
	}

	return VDB_SUCCESS;
}

const char* vdb_error(VectorDBErrorCode err) {
	switch (err) {
		case VDB_SUCCESS:
			return "Success.";
		case VDB_OPEN_ERR:
			return "Failed to open file.";
		case VDB_CLOSE_ERR:
			return "Failed to close file.";
		case VDB_HEADER_WRITE_ERR:
			return "Failed to write header.";
		case VDB_HEADER_READ_ERR:
			return "Failed to read header.";
		case VDB_MAGIC_MISMATCH_ERR:
			return "Header magic/version mismatch.";
		case VDB_EMBED_MISMATCH_ERR:
			return "Header embed/max_text mismatch.";
		case VDB_COUNT_TOO_LARGE_ERR:
			return "Header count too large.";
		case VDB_DOC_WRITE_ERR:
			return "Failed to write documents.";
		case VDB_DOC_READ_ERR:
			return "Failed to read documents.";
		default:
			return "Unknown error.";
	}
}
