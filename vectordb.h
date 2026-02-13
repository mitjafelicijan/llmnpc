#ifndef VECTORDB_H
#define VECTORDB_H

#include "llama.h"

#define VDB_MAX_DOCS 1000
#define VDB_EMBED_SIZE 768
#define VDB_MAX_TEXT 1024

typedef struct {
	float embedding[VDB_EMBED_SIZE];
	char text[VDB_MAX_TEXT];
} VectorDoc;

typedef struct {
	VectorDoc docs[VDB_MAX_DOCS];
	int count;
	struct llama_context *embed_ctx;
} VectorDB;

void vdb_init(VectorDB *db, struct llama_context *embed_ctx);
void vdb_free(VectorDB *db);

void vdb_add_document(VectorDB *db, const char *text);

void vdb_embed_query(VectorDB *db, const char *text, float *out_embedding);
void vdb_search(VectorDB *db, float *query_embedding, int top_k, int *results);

int vdb_save(const VectorDB *db, const char *path);
int vdb_load(VectorDB *db, const char *path);

#endif
