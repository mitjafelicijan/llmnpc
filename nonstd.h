// nonstd.h
// A collection of useful functions and macros.
// This library is licensed under the BSD 2-Clause License.
// 
// This file provides both the interface and the implementation.
// To instantiate the implementation,
//     #define NONSTD_IMPLEMENTATION
// before including this file.

#ifdef NONSTD_IMPLEMENTATION
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#ifndef NONSTD_H
#define NONSTD_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifndef NONSTD_DEF
#ifdef NONSTD_STATIC
#define NONSTD_DEF static
#else
#define NONSTD_DEF extern
#endif
#endif

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef intptr_t isize;
typedef uintptr_t usize;
typedef char c8;

#define countof(a) (sizeof(a) / sizeof((a)[0]))

#define static_foreach(type, var, array)                 \
	for (size_t _i_##var = 0, _n_##var = countof(array); \
		 _i_##var < _n_##var && ((var) = (array)[_i_##var], 1); ++_i_##var)

#define ALLOC(type, n) ((type *)safe_malloc(sizeof(type), (n)))
#define REALLOC(ptr, type, n) ((type *)safe_realloc((ptr), sizeof(type), (n)))
#define FREE(ptr)   \
	do {            \
		free(ptr);  \
		ptr = NULL; \
	} while (0)

NONSTD_DEF void *safe_malloc(size_t item_size, size_t count);
NONSTD_DEF void *safe_realloc(void *ptr, size_t item_size, size_t count);

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN((hi), MAX((lo), (x))))

// From https://github.com/tsoding/nob.h/blob/e2c9a46f01d052ab740140e74453665dc3334832/nob.h#L205-L206.
#define UNUSED(value) (void)(value)
#define TODO(message)                                                      \
	do {                                                                   \
		fprintf(stderr, "%s:%d: TODO: %s\n", __FILE__, __LINE__, message); \
		abort();                                                           \
	} while (0)
#define UNREACHABLE(message)                                                      \
	do {                                                                          \
		fprintf(stderr, "%s:%d: UNREACHABLE: %s\n", __FILE__, __LINE__, message); \
		abort();                                                                  \
	} while (0)

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define STATIC_ASSERT(expr, msg) _Static_assert((expr), msg)
#else
#define STATIC_ASSERT(expr, msg) \
	typedef char static_assertion_##msg[(expr) ? 1 : -1]
#endif

// String view - read-only, non-owning reference to a string
typedef struct {
	const char *data;
	size_t length;
} stringv;

NONSTD_DEF stringv sv_from_cstr(const char *s);
NONSTD_DEF stringv sv_from_parts(const char *data, size_t length);
NONSTD_DEF stringv sv_slice(stringv sv, size_t start, size_t end);
NONSTD_DEF int sv_equals(stringv a, stringv b);
NONSTD_DEF int sv_starts_with(stringv sv, stringv prefix);
NONSTD_DEF int sv_ends_with(stringv sv, stringv suffix);

// String builder - owning, mutable, dynamically growing string buffer
typedef struct {
	char *data;
	size_t length;
	size_t capacity;
} stringb;

NONSTD_DEF void sb_init(stringb *sb, size_t initial_cap);
NONSTD_DEF void sb_free(stringb *sb);
NONSTD_DEF void sb_ensure(stringb *sb, size_t additional);
NONSTD_DEF void sb_append_cstr(stringb *sb, const char *s);
NONSTD_DEF void sb_append_sv(stringb *sb, stringv sv);
NONSTD_DEF void sb_append_char(stringb *sb, char c);
NONSTD_DEF stringv sb_as_sv(const stringb *sb);

// Slice - generic non-owning view into an array
// Usage: SLICE_DEF(int); slice(int) view = ...;
#define SLICE_DEF(T)   \
	typedef struct {   \
		T *data;       \
		size_t length; \
	} slice_##T

#define slice(T) slice_##T

#define make_slice(T, ptr, len) ((slice(T)){.data = (ptr), .length = (len)})

#define array_as_slice(T, arr) \
	((slice(T)){.data = (arr).data, .length = (arr).length})

// Dynamic array - generic type-safe growable array using macros
// Usage: array(int) numbers; array_init(numbers);
#define array(T)         \
	struct {             \
		T *data;         \
		size_t length;   \
		size_t capacity; \
	}

#define array_init(arr)     \
	do {                    \
		(arr).capacity = 0; \
		(arr).data = NULL;  \
		(arr).length = 0;   \
	} while (0)

#define array_init_cap(arr, initial_cap)                             \
	do {                                                             \
		(arr).capacity = (initial_cap) ? (initial_cap) : 16;         \
		(arr).data = ALLOC(__typeof__(*(arr).data), (arr).capacity); \
		(arr).length = 0;                                            \
	} while (0)

#define array_free(arr)     \
	do {                    \
		FREE((arr).data);   \
		(arr).length = 0;   \
		(arr).capacity = 0; \
	} while (0)

#define array_ensure(arr, additional)                                    \
	do {                                                                 \
		size_t _needed = (arr).length + (additional);                    \
		if (_needed > (arr).capacity) {                                  \
			size_t _new_cap = (arr).capacity ? (arr).capacity : 16;      \
			while (_new_cap < _needed) {                                 \
				if (_new_cap > SIZE_MAX / 2) {                           \
					_new_cap = SIZE_MAX;                                 \
					break;                                               \
				}                                                        \
				_new_cap *= 2;                                           \
			}                                                            \
			if (_new_cap < _needed) { /* Overflow or OOM */              \
				break;                                                   \
			}                                                            \
			void *_new_data =                                            \
				safe_realloc((arr).data, sizeof(*(arr).data), _new_cap); \
			if (_new_data) {                                             \
				(arr).data = _new_data;                                  \
				(arr).capacity = _new_cap;                               \
			}                                                            \
		}                                                                \
	} while (0)

#define array_push(arr, value)                    \
	do {                                          \
		array_ensure((arr), 1);                   \
		if ((arr).length < (arr).capacity) {      \
			(arr).data[(arr).length++] = (value); \
		}                                         \
	} while (0)

#define array_pop(arr) ((arr).length > 0 ? (arr).data[--(arr).length] : 0)

#define array_get(arr, index) ((arr).data[index])

#define array_set(arr, index, value)     \
	do {                                 \
		if ((index) < (arr).length) {    \
			(arr).data[index] = (value); \
		}                                \
	} while (0)

#define array_insert(arr, index, value)                              \
	do {                                                             \
		if ((index) <= (arr).length) {                               \
			array_ensure((arr), 1);                                  \
			if ((arr).length < (arr).capacity) {                     \
				for (size_t _i = (arr).length; _i > (index); --_i) { \
					(arr).data[_i] = (arr).data[_i - 1];             \
				}                                                    \
				(arr).data[index] = (value);                         \
				(arr).length++;                                      \
			}                                                        \
		}                                                            \
	} while (0)

#define array_remove(arr, index)                                     \
	do {                                                             \
		if ((index) < (arr).length) {                                \
			for (size_t _i = (index); _i < (arr).length - 1; ++_i) { \
				(arr).data[_i] = (arr).data[_i + 1];                 \
			}                                                        \
			(arr).length--;                                          \
		}                                                            \
	} while (0)

#define array_clear(arr)  \
	do {                  \
		(arr).length = 0; \
	} while (0)

#define array_reserve(arr, new_capacity)                                       \
	do {                                                                       \
		if ((new_capacity) > (arr).capacity) {                                 \
			void *_new_data =                                                  \
				safe_realloc((arr).data, sizeof(*(arr).data), (new_capacity)); \
			if (_new_data) {                                                   \
				(arr).data = _new_data;                                        \
				(arr).capacity = (new_capacity);                               \
			}                                                                  \
		}                                                                      \
	} while (0)

#define array_foreach(arr, var)                                        \
	for (size_t _i_##var = 0;                                          \
		 _i_##var < (arr).length && ((var) = (arr).data[_i_##var], 1); \
		 ++_i_##var)

#define array_foreach_idx(arr, var, index) \
	for (size_t index = 0;                 \
		 index < (arr).length && ((var) = (arr).data[index], 1); ++index)

// Arena - block-based memory allocator
typedef struct {
	char *ptr;
	char *end;
	array(char *) blocks;
} Arena;

#define ARENA_DEFAULT_BLOCK_SIZE (4096)

NONSTD_DEF Arena arena_make(void);
NONSTD_DEF void arena_grow(Arena *a, size_t min_size);
NONSTD_DEF void *arena_alloc(Arena *a, size_t size);
NONSTD_DEF void arena_free(Arena *a);

// Image - simple RGB image structure
typedef struct {
	u8 r, g, b;
} Color;

typedef struct {
	u32 width;
	u32 height;
	Color *pixels;
} Canvas;

#define COLOR_RGB(r, g, b) ((Color){(u8)(r), (u8)(g), (u8)(b)})
#define COLOR_HEX(hex) ((Color){(u8)(((hex) >> 16) & 0xFF), (u8)(((hex) >> 8) & 0xFF), (u8)((hex) & 0xFF)})

#define COLOR_BLACK COLOR_RGB(0, 0, 0)
#define COLOR_WHITE COLOR_RGB(255, 255, 255)
#define COLOR_RED COLOR_RGB(255, 0, 0)
#define COLOR_GREEN COLOR_RGB(0, 255, 0)
#define COLOR_BLUE COLOR_RGB(0, 0, 255)
#define COLOR_YELLOW COLOR_RGB(255, 255, 0)
#define COLOR_MAGENTA COLOR_RGB(255, 0, 255)
#define COLOR_CYAN COLOR_RGB(0, 255, 255)

NONSTD_DEF Canvas ppm_init(u32 width, u32 height);
NONSTD_DEF void ppm_free(Canvas *img);
NONSTD_DEF void ppm_set_pixel(Canvas *img, u32 x, u32 y, Color color);
NONSTD_DEF Color ppm_get_pixel(const Canvas *img, u32 x, u32 y);
NONSTD_DEF int ppm_save(const Canvas *img, const char *filename);
NONSTD_DEF Canvas ppm_read(const char *filename);
NONSTD_DEF void ppm_fill(Canvas *canvas, Color color);
NONSTD_DEF void ppm_draw_rect(Canvas *canvas, u32 x, u32 y, u32 w, u32 h, Color color);
NONSTD_DEF void ppm_draw_line(Canvas *canvas, i32 x0, i32 y0, i32 x1, i32 y1, Color color);
NONSTD_DEF void ppm_draw_circle(Canvas *canvas, i32 x, i32 y, i32 r, Color color);
NONSTD_DEF void ppm_draw_triangle(Canvas *canvas, i32 x0, i32 y0, i32 x1, i32 y1, i32 x2, i32 y2, Color color);

// File I/O helpers
NONSTD_DEF char *read_entire_file(const char *filepath, size_t *out_size);
NONSTD_DEF int write_entire_file(const char *filepath, const void *data, size_t size);
NONSTD_DEF stringb read_entire_file_sb(const char *filepath);
NONSTD_DEF int write_file_sv(const char *filepath, stringv sv);
NONSTD_DEF int write_file_sb(const char *filepath, const stringb *sb);

// Logging
typedef enum {
	LOG_ERROR,
	LOG_WARN,
	LOG_INFO,
	LOG_DEBUG,
} LogLevel;

NONSTD_DEF void set_log_level(LogLevel level);
NONSTD_DEF LogLevel get_log_level_from_env(void);
NONSTD_DEF void log_message(FILE *stream, LogLevel level, const char *format, ...);

#define LOG_INFO_MSG(...) log_message(stdout, LOG_INFO, __VA_ARGS__)
#define LOG_DEBUG_MSG(...) log_message(stdout, LOG_DEBUG, __VA_ARGS__)
#define LOG_WARN_MSG(...) log_message(stderr, LOG_WARN, __VA_ARGS__)
#define LOG_ERROR_MSG(...) log_message(stderr, LOG_ERROR, __VA_ARGS__)

#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[32m"
#define COLOR_DEBUG "\033[36m"
#define COLOR_WARNING "\033[33m"
#define COLOR_ERROR "\033[31m"

#endif // NONSTD_H

#ifdef NONSTD_IMPLEMENTATION

NONSTD_DEF void *safe_malloc(size_t item_size, size_t count) {
	if (count != 0 && item_size > SIZE_MAX / count) {
		return NULL;
	}
	return malloc(item_size * count);
}

NONSTD_DEF void *safe_realloc(void *ptr, size_t item_size, size_t count) {
	if (count != 0 && item_size > SIZE_MAX / count) {
		return NULL;
	}
	return realloc(ptr, item_size * count);
}

// String View Implementation

NONSTD_DEF stringv sv_from_cstr(const char *s) {
	return (stringv){.data = s, .length = s ? strlen(s) : 0};
}

NONSTD_DEF stringv sv_from_parts(const char *data, size_t length) {
	return (stringv){.data = data, .length = length};
}

NONSTD_DEF stringv sv_slice(stringv sv, size_t start, size_t end) {
	if (start > sv.length) {
		start = sv.length;
	}
	if (end > sv.length) {
		end = sv.length;
	}
	if (start > end) {
		start = end;
	}
	return (stringv){.data = sv.data + start, .length = end - start};
}

NONSTD_DEF int sv_equals(stringv a, stringv b) {
	return a.length == b.length && (a.length == 0 || memcmp(a.data, b.data, a.length) == 0);
}

NONSTD_DEF int sv_starts_with(stringv sv, stringv prefix) {
	return sv.length >= prefix.length && memcmp(sv.data, prefix.data, prefix.length) == 0;
}

NONSTD_DEF int sv_ends_with(stringv sv, stringv suffix) {
	return sv.length >= suffix.length && memcmp(sv.data + sv.length - suffix.length, suffix.data, suffix.length) == 0;
}

// String Builder Implementation

NONSTD_DEF void sb_init(stringb *sb, size_t initial_cap) {
	sb->capacity = initial_cap ? initial_cap : 16;
	sb->data = ALLOC(char, sb->capacity);
	sb->length = 0;
	if (sb->data) {
		sb->data[0] = '\0';
	}
}

NONSTD_DEF void sb_free(stringb *sb) {
	FREE(sb->data);
	sb->length = 0;
	sb->capacity = 0;
}

NONSTD_DEF void sb_ensure(stringb *sb, size_t additional) {
	size_t needed = sb->length + additional + 1;
	size_t new_cap = sb->capacity;

	if (needed > new_cap) {
		while (new_cap < needed) {
			if (new_cap > SIZE_MAX / 2) {
				new_cap = SIZE_MAX;
				break;
			}
			new_cap *= 2;
		}
		if (new_cap < needed)
			return; // Overflow

		char *new_data = safe_realloc(sb->data, sizeof(char), new_cap);
		if (new_data) {
			sb->data = new_data;
			sb->capacity = new_cap;
		}
	}
}

NONSTD_DEF void sb_append_cstr(stringb *sb, const char *s) {
	if (!s) {
		return;
	}
	size_t slength = strlen(s);
	sb_ensure(sb, slength);
	if (sb->length + slength + 1 <= sb->capacity) {
		memcpy(sb->data + sb->length, s, slength);
		sb->length += slength;
		sb->data[sb->length] = '\0';
	}
}

NONSTD_DEF void sb_append_sv(stringb *sb, stringv sv) {
	if (!sv.data || sv.length == 0) {
		return;
	}
	sb_ensure(sb, sv.length);
	if (sb->length + sv.length + 1 <= sb->capacity) {
		memcpy(sb->data + sb->length, sv.data, sv.length);
		sb->length += sv.length;
		sb->data[sb->length] = '\0';
	}
}

NONSTD_DEF void sb_append_char(stringb *sb, char c) {
	sb_ensure(sb, 1);
	if (sb->length + 2 <= sb->capacity) {
		sb->data[sb->length++] = c;
		sb->data[sb->length] = '\0';
	}
}

NONSTD_DEF stringv sb_as_sv(const stringb *sb) {
	return (stringv){.data = sb->data, .length = sb->length};
}

NONSTD_DEF Arena arena_make(void) {
	Arena a = {0};
	array_init(a.blocks);
	return a;
}

// Arena Implementation

NONSTD_DEF void arena_grow(Arena *a, size_t min_size) {
	size_t size = MAX(ARENA_DEFAULT_BLOCK_SIZE, min_size);
	char *block = ALLOC(char, size);
	a->ptr = block;
	a->end = block + size;
	array_push(a->blocks, block);
}

NONSTD_DEF void *arena_alloc(Arena *a, size_t size) {
	// Align to 8 bytes basically
	size_t align = sizeof(void *);
	uintptr_t current = (uintptr_t)a->ptr;
	uintptr_t aligned = (current + align - 1) & ~(align - 1);
	uintptr_t end = (uintptr_t)a->end;

	// Check for overflow (aligned wrapped around) or out of bounds (aligned >= end)
	// or not enough space ((end - aligned) < size)
	if (aligned < current || aligned >= end || (end - aligned) < size) {
		arena_grow(a, size);
		current = (uintptr_t)a->ptr;
		aligned = (current + align - 1) & ~(align - 1);
		end = (uintptr_t)a->end;
	}

	// Double check after grow (in case grow failed or size is just too huge)
	if (aligned < current || aligned >= end || (end - aligned) < size) {
		return NULL;
	}

	a->ptr = (char *)(aligned + size);
	return (void *)aligned;
}

NONSTD_DEF void arena_free(Arena *a) {
	char *block;
	array_foreach(a->blocks, block) { FREE(block); }
	array_free(a->blocks);
	a->ptr = NULL;
	a->end = NULL;
}

// File I/O Implementation

NONSTD_DEF char *read_entire_file(const char *filepath, size_t *out_size) {
	FILE *f = fopen(filepath, "rb");
	if (!f) {
		return NULL;
	}

	fseek(f, 0, SEEK_END);
	long length = ftell(f);
	fseek(f, 0, SEEK_SET);

	if (length < 0) {
		fclose(f);
		return NULL;
	}

	size_t size = (size_t)length;
	char *buffer = ALLOC(char, size + 1);
	if (!buffer) {
		fclose(f);
		return NULL;
	}

	size_t read = fread(buffer, 1, size, f);
	fclose(f);

	if (read != size) {
		FREE(buffer);
		return NULL;
	}

	buffer[size] = '\0';
	if (out_size) {
		*out_size = size;
	}

	return buffer;
}

NONSTD_DEF int write_entire_file(const char *filepath, const void *data, size_t size) {
	FILE *f = fopen(filepath, "wb");
	if (!f) {
		return 0;
	}

	size_t written = fwrite(data, 1, size, f);
	fclose(f);

	return written == size;
}

NONSTD_DEF stringb read_entire_file_sb(const char *filepath) {
	size_t size = 0;
	char *data = read_entire_file(filepath, &size);
	stringb sb = {0};
	if (data) {
		sb.data = data;
		sb.length = size;
		sb.capacity = size + 1;
	}
	return sb;
}

NONSTD_DEF int write_file_sv(const char *filepath, stringv sv) {
	return write_entire_file(filepath, sv.data, sv.length);
}

NONSTD_DEF int write_file_sb(const char *filepath, const stringb *sb) {
	return write_entire_file(filepath, sb->data, sb->length);
}

// Logging Implementation

static LogLevel max_level = LOG_INFO;

static const char *level_strings[] = {
	"ERROR",
	"WARN",
	"INFO",
	"DEBUG",
};

static const char *level_colors[] = {
	COLOR_ERROR,
	COLOR_WARNING,
	COLOR_INFO,
	COLOR_DEBUG,
};

NONSTD_DEF void set_log_level(LogLevel level) {
	max_level = level;
}

NONSTD_DEF LogLevel get_log_level_from_env(void) {
	const char *env = getenv("LOG_LEVEL");
	if (env) {
		int level = atoi(env);
		if (level >= 0 && level <= 3) {
			return (LogLevel)level;
		}
	}

	return max_level;
}

NONSTD_DEF void log_message(FILE *stream, LogLevel level, const char *format, ...) {
	if (max_level < level)
		return;

	struct timeval tv;
	gettimeofday(&tv, NULL);
	struct tm *tm_info = localtime(&tv.tv_sec);

	char time_str[24];
	strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);

	const char *color = isatty(fileno(stream)) ? level_colors[level] : "";
	const char *reset = isatty(fileno(stream)) ? COLOR_RESET : "";

	const char *log_format = "%s[%s.%03d] [%-5s] ";
	fprintf(stream, log_format, color, time_str, (int)(tv.tv_usec / 1000), level_strings[level]);

	va_list args;
	va_start(args, format);
	vfprintf(stream, format, args);
	va_end(args);

	fprintf(stream, "%s\n", reset);
	fflush(stream);
}

// PPM Image Implementation

NONSTD_DEF Canvas ppm_init(u32 width, u32 height) {
	Canvas img = {0};
	img.width = width;
	img.height = height;
	img.pixels = ALLOC(Color, width * height);
	if (img.pixels) {
		memset(img.pixels, 0, sizeof(Color) * width * height);
	}
	return img;
}

NONSTD_DEF void ppm_free(Canvas *img) {
	if (img->pixels) {
		FREE(img->pixels);
	}
	img->width = 0;
	img->height = 0;
}

NONSTD_DEF void ppm_set_pixel(Canvas *img, u32 x, u32 y, Color color) {
	if (x < img->width && y < img->height) {
		img->pixels[y * img->width + x] = color;
	}
}

NONSTD_DEF Color ppm_get_pixel(const Canvas *img, u32 x, u32 y) {
	if (x < img->width && y < img->height) {
		return img->pixels[y * img->width + x];
	}
	return (Color){0, 0, 0};
}

NONSTD_DEF int ppm_save(const Canvas *img, const char *filename) {
	FILE *f = fopen(filename, "w");
	if (!f) {
		return 0;
	}

	fprintf(f, "P3\n%u %u\n255\n", img->width, img->height);
	for (u32 y = 0; y < img->height; ++y) {
		for (u32 x = 0; x < img->width; ++x) {
			Color c = ppm_get_pixel(img, x, y);
			fprintf(f, "%d %d %d ", c.r, c.g, c.b);
		}
		fprintf(f, "\n");
	}

	fclose(f);
	return 1;
}

NONSTD_DEF Canvas ppm_read(const char *filename) {
	Canvas img = {0};
	FILE *f = fopen(filename, "r");
	if (!f) {
		return img;
	}

	char magic[3];
	if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P3") != 0) {
		fclose(f);
		return img;
	}

	u32 w, h, max_val;
	if (fscanf(f, "%u %u %u", &w, &h, &max_val) != 3) {
		fclose(f);
		return img;
	}

	img = ppm_init(w, h);
	if (!img.pixels) {
		fclose(f);
		return img;
	}

	for (u32 i = 0; i < w * h; ++i) {
		int r, g, b;
		if (fscanf(f, "%d %d %d", &r, &g, &b) != 3) {
			ppm_free(&img);
			fclose(f);
			return (Canvas){0};
		}
		img.pixels[i] = (Color){(u8)r, (u8)g, (u8)b};
	}

	fclose(f);
	return img;
}

NONSTD_DEF void ppm_fill(Canvas *canvas, Color color) {
	for (u32 i = 0; i < canvas->width * canvas->height; ++i) {
		canvas->pixels[i] = color;
	}
}

NONSTD_DEF void ppm_draw_rect(Canvas *canvas, u32 x, u32 y, u32 w, u32 h, Color color) {
	if (w == 0 || h == 0) {
		return;
	}
	for (u32 i = x; i < x + w; ++i) {
		ppm_set_pixel(canvas, i, y, color);
		ppm_set_pixel(canvas, i, y + h - 1, color);
	}
	for (u32 j = y; j < y + h; ++j) {
		ppm_set_pixel(canvas, x, j, color);
		ppm_set_pixel(canvas, x + w - 1, j, color);
	}
}

NONSTD_DEF void ppm_draw_line(Canvas *canvas, i32 x0, i32 y0, i32 x1, i32 y1, Color color) {
	i32 dx = abs(x1 - x0);
	i32 dy = -abs(y1 - y0);
	i32 sx = x0 < x1 ? 1 : -1;
	i32 sy = y0 < y1 ? 1 : -1;
	i32 err = dx + dy;

	while (1) {
		ppm_set_pixel(canvas, (u32)x0, (u32)y0, color);
		if (x0 == x1 && y0 == y1) {
			break;
		}

		i32 e2 = 2 * err;
		if (e2 >= dy) {
			err += dy;
			x0 += sx;
		}
		if (e2 <= dx) {
			err += dx;
			y0 += sy;
		}
	}
}

NONSTD_DEF void ppm_draw_circle(Canvas *canvas, i32 xm, i32 ym, i32 r, Color color) {
	i32 x = -r, y = 0, err = 2 - 2 * r;
	do {
		ppm_set_pixel(canvas, (u32)(xm - x), (u32)(ym + y), color);
		ppm_set_pixel(canvas, (u32)(xm - y), (u32)(ym - x), color);
		ppm_set_pixel(canvas, (u32)(xm + x), (u32)(ym - y), color);
		ppm_set_pixel(canvas, (u32)(xm + y), (u32)(ym + x), color);
		r = err;
		if (r <= y) {
			err += ++y * 2 + 1;
		}
		if (r > x || err > y) {
			err += ++x * 2 + 1;
		}
	} while (x < 0);
}

NONSTD_DEF void ppm_draw_triangle(Canvas *canvas, i32 x0, i32 y0, i32 x1, i32 y1, i32 x2, i32 y2, Color color) {
	ppm_draw_line(canvas, x0, y0, x1, y1, color);
	ppm_draw_line(canvas, x1, y1, x2, y2, color);
	ppm_draw_line(canvas, x2, y2, x0, y0, color);
}

#endif // NONSTD_IMPLEMENTATION

/*
BSD 2-Clause License

Copyright (c) 2026, Mitja Felicijan <mitja.felicijan@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/