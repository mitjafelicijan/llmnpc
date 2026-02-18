#ifndef VECTORDB_H
#define VECTORDB_H

#include "llama.h"

#define VDB_MAX_DOCS    1000
#define VDB_EMBED_SIZE  768
#define VDB_MAX_TEXT    1024

#define VDB_MAGIC       0x31424456u /* "VDB1" */
#define VDB_VERSION     1u
#define VDB_TOKENS      512

typedef struct {
	float embedding[VDB_EMBED_SIZE];
	char text[VDB_MAX_TEXT];
} VectorDoc;

typedef struct {
	VectorDoc docs[VDB_MAX_DOCS];
	int count;
	struct llama_context *embed_ctx;
} VectorDB;

typedef struct {
	uint32_t magic;
	uint32_t version;
	uint32_t embed_size;
	uint32_t max_text;
	uint32_t count;
} VdbFileHeader;

typedef enum {
	VDB_SUCCESS                = 0,
	VDB_OPEN_ERR               = 9001,
	VDB_CLOSE_ERR              = 9002,
	VDB_HEADER_WRITE_ERR       = 9003,
	VDB_HEADER_READ_ERR        = 9004,
	VDB_MAGIC_MISMATCH_ERR     = 9005,
	VDB_EMBED_MISMATCH_ERR     = 9006,
	VDB_COUNT_TOO_LARGE_ERR    = 9007,
	VDB_DOC_WRITE_ERR          = 9008,
	VDB_DOC_READ_ERR           = 9009,
} VectorDBErrorCode;

void vdb_init(VectorDB *db, struct llama_context *embed_ctx);
void vdb_free(VectorDB *db);

void vdb_add_document(VectorDB *db, const char *text);

void vdb_embed_query(VectorDB *db, const char *text, float *out_embedding);
void vdb_search(VectorDB *db, float *query_embedding, int top_k, int *results);

VectorDBErrorCode vdb_save(const VectorDB *db, const char *path);
VectorDBErrorCode vdb_load(VectorDB *db, const char *path);

const char* vdb_error(VectorDBErrorCode err);

#endif
