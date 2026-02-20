#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define TB_IMPL
#include "termbox2.h"

#define NONSTD_IMPLEMENTATION
#include "nonstd.h"

#include "llama.h"
#include "models.h"
#include "vectordb.h"
#include "maps.h"

#define MIN_W 40
#define MIN_H 12
#define SIDEBAR_W 40
#define CP_H 0x2500
#define CP_V 0x2502
#define CP_TL 0x250c
#define CP_TR 0x2510
#define CP_BL 0x2514
#define CP_BR 0x2518
#define MAP_FLOOR_CH '.'
#define MAP_BORDER_MIN 0x2500
#define MAP_BORDER_MAX 0x257f
#define MAP_FLOOR_FG 234
#define COLOR_WHITE_256 0x0f
#define COLOR_RED_256 161
#define COLOR_GREEN_256 0x2e
#define COLOR_BORDER_256 101
#define COLOR_CYAN_256 0x33
#define COLOR_ORANGE_256 0xd0
#define COLOR_BLUE_256 0x1b

typedef struct {
	char key;
	const char *name;
} InventoryItem;

typedef struct {
	array(InventoryItem) items;
} Inventory;

typedef struct {
	int x;
	int y;
	int hp;
	int hp_max;
	int ac;
	int str;
	int gold;
	Inventory inventory;
} Player;

#define DIALOG_HISTORY_MAX 16

typedef struct {
	char prompt[128];
	char response[256];
} DialogEntry;

typedef struct {
	int open;
	char input[128];
	int input_len;
	int npc_index;
	const char *npc_name;
	DialogEntry entries[DIALOG_HISTORY_MAX];
	int entry_count;
} Dialog;

typedef struct {
	const ModelConfig *model_cfg;
	struct llama_model *model;
	struct llama_model *embed_model;
	struct llama_context *embed_ctx;
	VectorDB *npc_dbs;
	int *npc_db_loaded;
	int verbose;
} GameRuntime;

static void llama_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
	(void)level;
	(void)user_data;
	(void)text;
}

static int clamp(int value, int min, int max);

static void show_help(const char *prog) {
	printf("Usage: %s [OPTIONS]\n", prog);
	printf("Options:\n");
	printf("  -m, --model <name>    Specify model to use (default: first model)\n");
	printf("  -e, --embed-model <name> Specify model to use for embeddings\n");
	printf("  -v, --verbose         Enable verbose logging\n");
	printf("  -h, --help            Show this help message\n");
}

static void draw_border(int x, int y, int w, int h, uintattr_t fg) {
	int ix;
	int iy;

	for (ix = 0; ix < w; ix++) {
		tb_set_cell(x + ix, y, CP_H, fg, TB_DEFAULT);
		tb_set_cell(x + ix, y + h - 1, CP_H, fg, TB_DEFAULT);
	}
	for (iy = 0; iy < h; iy++) {
		tb_set_cell(x, y + iy, CP_V, fg, TB_DEFAULT);
		tb_set_cell(x + w - 1, y + iy, CP_V, fg, TB_DEFAULT);
	}

	tb_set_cell(x, y, CP_TL, fg, TB_DEFAULT);
	tb_set_cell(x + w - 1, y, CP_TR, fg, TB_DEFAULT);
	tb_set_cell(x, y + h - 1, CP_BL, fg, TB_DEFAULT);
	tb_set_cell(x + w - 1, y + h - 1, CP_BR, fg, TB_DEFAULT);
}

static void draw_border_bg(int x, int y, int w, int h, uintattr_t fg,
		uintattr_t bg) {
	int ix;
	int iy;

	for (ix = 0; ix < w; ix++) {
		tb_set_cell(x + ix, y, CP_H, fg, bg);
		tb_set_cell(x + ix, y + h - 1, CP_H, fg, bg);
	}
	for (iy = 0; iy < h; iy++) {
		tb_set_cell(x, y + iy, CP_V, fg, bg);
		tb_set_cell(x + w - 1, y + iy, CP_V, fg, bg);
	}

	tb_set_cell(x, y, CP_TL, fg, bg);
	tb_set_cell(x + w - 1, y, CP_TR, fg, bg);
	tb_set_cell(x, y + h - 1, CP_BL, fg, bg);
	tb_set_cell(x + w - 1, y + h - 1, CP_BR, fg, bg);
}

static void get_layout(int w, int h, int *map_x, int *map_y, int *map_w,
		int *map_h, int *side_x, int *side_y, int *side_w, int *side_h,
		int *msg1_y, int *msg2_y) {
	*map_x = 0;
	*map_y = 0;
	*map_w = w - SIDEBAR_W;
	*map_h = h - 2;
	*side_x = w - SIDEBAR_W;
	*side_y = 0;
	*side_w = SIDEBAR_W;
	*side_h = h - 2;
	*msg1_y = h - 2;
	*msg2_y = h - 1;
}

static void map_init(Map *map, const unsigned char *data, int len) {
	array(int) line_lengths;
	int width = 0;
	int height = 0;
	int line_len = 0;
	int i = 0;

	map->data = data;
	map->len = len;
	map->cells = NULL;
	array_init(line_lengths);
	while (i < len) {
		uint32_t ch = 0;
		int consumed = tb_utf8_char_to_unicode(&ch, (const char *)&data[i]);
		if (consumed <= 0) {
			i++;
			continue;
		}
		i += consumed;
		if (ch == '\n') {
			array_push(line_lengths, line_len);
			if (line_len > width) {
				width = line_len;
			}
			height++;
			line_len = 0;
		} else {
			line_len++;
		}
	}
	if (line_len > 0 || (len > 0 && data[len - 1] != '\n')) {
		array_push(line_lengths, line_len);
		if (line_len > width) {
			width = line_len;
		}
		height++;
	}

	map->width = width;
	map->height = height;
	if (width > 0 && height > 0) {
		map->cells = ALLOC(u32, (usize)width * (usize)height);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				map->cells[(y * width) + x] = ' ';
			}
		}
	}

	i = 0;
	int x = 0;
	int y = 0;
	while (i < len && y < height) {
		uint32_t ch = 0;
		int consumed = tb_utf8_char_to_unicode(&ch, (const char *)&data[i]);
		if (consumed <= 0) {
			i++;
			continue;
		}
		i += consumed;
		if (ch == '\n') {
			y++;
			x = 0;
			continue;
		}
		if (map->cells && x < width) {
			map->cells[(y * width) + x] = ch;
		}
		x++;
	}
	array_free(line_lengths);
}

static u32 map_get(const Map *map, int x, int y) {
	if (!map->cells || x < 0 || y < 0 || x >= map->width || y >= map->height) {
		return ' ';
	}
	return map->cells[(y * map->width) + x];
}

static void map_set(Map *map, int x, int y, u32 ch) {
	if (!map->cells || x < 0 || y < 0 || x >= map->width || y >= map->height) {
		return;
	}
	map->cells[(y * map->width) + x] = ch;
}

static int map_is_walkable(const Map *map, int x, int y) {
	u32 ch = map_get(map, x, y);
	return ch == MAP_FLOOR_CH || ch == '$' || ch == 'N'
		|| (ch >= '0' && ch <= '9');
}

static int npc_index_from_tile(u32 ch) {
	if (ch >= '0' && ch <= '9') {
		return (int)(ch - '0');
	}
	return -1;
}

static void map_free(Map *map) {
	FREE(map->cells);
}

static void update_camera(const Map *map, int view_w, int view_h,
		const Player *player, int *cam_x, int *cam_y) {
	int max_cam_x;
	int max_cam_y;
	int margin_x;
	int margin_y;
	int next_x = *cam_x;
	int next_y = *cam_y;

	if (view_w <= 0 || view_h <= 0 || map->width <= 0 || map->height <= 0) {
		*cam_x = 0;
		*cam_y = 0;
		return;
	}

	margin_x = view_w > 8 ? 3 : view_w / 3;
	margin_y = view_h > 8 ? 3 : view_h / 3;
	max_cam_x = map->width - view_w;
	max_cam_y = map->height - view_h;
	if (max_cam_x < 0) {
		max_cam_x = 0;
	}
	if (max_cam_y < 0) {
		max_cam_y = 0;
	}

	if (player->x < next_x + margin_x) {
		next_x = player->x - margin_x;
	} else if (player->x > next_x + view_w - 1 - margin_x) {
		next_x = player->x - (view_w - 1 - margin_x);
	}
	if (player->y < next_y + margin_y) {
		next_y = player->y - margin_y;
	} else if (player->y > next_y + view_h - 1 - margin_y) {
		next_y = player->y - (view_h - 1 - margin_y);
	}

	*cam_x = clamp(next_x, 0, max_cam_x);
	*cam_y = clamp(next_y, 0, max_cam_y);
}

static void draw_map(const Map *map, int map_x, int map_y, int view_w,
		int view_h, const Player *player, int cam_x, int cam_y) {
	int ix;
	int iy;

	for (iy = 0; iy < view_h; iy++) {
		for (ix = 0; ix < view_w; ix++) {
			int mx = cam_x + ix;
			int my = cam_y + iy;
			u32 ch = map_get(map, mx, my);
			u32 draw_ch = (ch >= '0' && ch <= '9') ? 'N' : ch;
			uintattr_t fg = COLOR_WHITE_256;
			if (ch == MAP_FLOOR_CH) {
				fg = MAP_FLOOR_FG;
			} else if (ch == '~') {
				fg = COLOR_BLUE_256;
			} else if (ch == '$') {
				fg = COLOR_ORANGE_256;
			} else if (ch == 'B' || ch == 'S' || ch == 'G') {
				fg = COLOR_RED_256;
			} else if (ch == 'N' || (ch >= '0' && ch <= '9')) {
				fg = COLOR_CYAN_256;
			} else if (ch >= MAP_BORDER_MIN && ch <= MAP_BORDER_MAX) {
				fg = COLOR_BORDER_256;
			}
			tb_set_cell(map_x + ix, map_y + iy, draw_ch, fg, TB_DEFAULT);
		}
	}

	if (player->x >= cam_x && player->x < cam_x + view_w && player->y >= cam_y
			&& player->y < cam_y + view_h) {
		int sx = map_x + (player->x - cam_x);
		int sy = map_y + (player->y - cam_y);
		tb_set_cell(sx, sy, '@', COLOR_GREEN_256 | TB_BOLD, TB_DEFAULT);
	}
}

static void draw_progress_bar(int x, int y, int w, int value, int max) {
	int filled;
	int ix;
	int inner_w = w - 2;

	if (w < 4) {
		return;
	}
	if (max <= 0) {
		max = 1;
	}
	if (value < 0) {
		value = 0;
	}
	if (value > max) {
		value = max;
	}

	filled = (inner_w * value) / max;
	tb_set_cell(x, y, '[', COLOR_WHITE_256, TB_DEFAULT);
	for (ix = 0; ix < inner_w; ix++) {
		uintattr_t fg = ix < filled ? COLOR_GREEN_256 : COLOR_WHITE_256;
		uint32_t ch = ix < filled ? '=' : ' ';
		tb_set_cell(x + 1 + ix, y, ch, fg, TB_DEFAULT);
	}
	tb_set_cell(x + w - 1, y, ']', COLOR_WHITE_256, TB_DEFAULT);
}

static void inventory_init(Inventory *inv) {
	array_init(inv->items);
}

static void inventory_add(Inventory *inv, char key, const char *name) {
	InventoryItem item = {.key = key, .name = name};
	array_push(inv->items, item);
}

static void inventory_free(Inventory *inv) {
	array_free(inv->items);
}

static void player_init(Player *player) {
	player->x = 6;
	player->y = 4;
	player->hp = 12;
	player->hp_max = 12;
	player->ac = 7;
	player->str = 16;
	player->gold = 42;
	inventory_init(&player->inventory);
	inventory_add(&player->inventory, 'a', "dagger");
	inventory_add(&player->inventory, 'b', "ration");
	inventory_add(&player->inventory, 'c', "potion");
	inventory_add(&player->inventory, 'd', "scroll");
}

static void player_free(Player *player) {
	inventory_free(&player->inventory);
}

static void draw_stats(int x, int y, const Player *player) {
	tb_print(x, y, COLOR_WHITE_256 | TB_BOLD, TB_DEFAULT, "Stats");
	tb_printf(x, y + 2, COLOR_WHITE_256, TB_DEFAULT, "HP %d/%d", player->hp, player->hp_max);
	draw_progress_bar(x, y + 3, 18, player->hp, player->hp_max);
	tb_printf(x, y + 4, COLOR_WHITE_256, TB_DEFAULT, "AC: %d", player->ac);
	tb_printf(x, y + 5, COLOR_WHITE_256, TB_DEFAULT, "Str: %d", player->str);
	tb_printf(x, y + 6, COLOR_WHITE_256, TB_DEFAULT, "Gold: %d", player->gold);
}

static void draw_inventory(int x, int y, const Inventory *inv) {
	InventoryItem item;
	usize idx = 0;

	tb_print(x, y, COLOR_WHITE_256 | TB_BOLD, TB_DEFAULT, "Inventory");
	array_foreach(inv->items, item) {
		tb_printf(x, y + 2 + (int)idx, COLOR_WHITE_256, TB_DEFAULT, "%c) %s", item.key, item.name);
		idx++;
	}
}

static const char *status_msg = "";

static void update_status(const char *message) {
	status_msg = message ? message : "";
}

static int draw_wrapped(int x, int y, int max_lines, int box_w, uintattr_t fg,
		uintattr_t bg, const char *prefix, const char *text) {
	if (max_lines <= 0 || box_w <= 0 || text == NULL) {
		return 0;
	}
	int lines = 0;
	int prefix_len = prefix ? (int)strlen(prefix) : 0;
	if (prefix_len < 0) {
		prefix_len = 0;
	}
	int avail = box_w - 4 - prefix_len;
	if (avail < 1) {
		return 0;
	}
	char pad[64];
	int pad_len = prefix_len < (int)sizeof(pad) - 1 ? prefix_len : (int)sizeof(pad) - 1;
	for (int i = 0; i < pad_len; i++) {
		pad[i] = ' ';
	}
	pad[pad_len] = '\0';
	const char *p = text;
	while (*p != '\0' && lines < max_lines) {
		while (*p == ' ') {
			p++;
		}
		int line_len = 0;
		int last_space = -1;
		for (int i = 0; i < avail && p[i] != '\0'; i++) {
			if (p[i] == '\n') {
				line_len = i;
				break;
			}
			if (p[i] == ' ') {
				last_space = i;
			}
			line_len = i + 1;
		}
		if (line_len == 0) {
			break;
		}
		int cut = line_len;
		if (cut == avail && p[cut] != '\0' && last_space > 0) {
			cut = last_space;
		}
		char buf[512];
		int copy_len = cut < (int)sizeof(buf) - 1 ? cut : (int)sizeof(buf) - 1;
		memcpy(buf, p, (size_t)copy_len);
		buf[copy_len] = '\0';
		while (copy_len > 0 && buf[copy_len - 1] == ' ') {
			buf[copy_len - 1] = '\0';
			copy_len--;
		}
		const char *line_prefix = (lines == 0) ? (prefix ? prefix : "") : pad;
		tb_printf(x, y + lines, fg, bg, "%s%s", line_prefix, buf);
		lines++;
		p += cut;
		if (*p == '\n') {
			p++;
		}
	}
	return lines;
}

static int count_wrapped_lines(int box_w, const char *prefix, const char *text) {
	if (box_w <= 0 || text == NULL) {
		return 0;
	}
	int prefix_len = prefix ? (int)strlen(prefix) : 0;
	if (prefix_len < 0) {
		prefix_len = 0;
	}
	int avail = box_w - 4 - prefix_len;
	if (avail < 1) {
		return 0;
	}
	int lines = 0;
	const char *p = text;
	while (*p != '\0') {
		while (*p == ' ') {
			p++;
		}
		int line_len = 0;
		int last_space = -1;
		for (int i = 0; i < avail && p[i] != '\0'; i++) {
			if (p[i] == '\n') {
				line_len = i;
				break;
			}
			if (p[i] == ' ') {
				last_space = i;
			}
			line_len = i + 1;
		}
		if (line_len == 0) {
			break;
		}
		int cut = line_len;
		if (cut == avail && p[cut] != '\0' && last_space > 0) {
			cut = last_space;
		}
		lines++;
		p += cut;
		if (*p == '\n') {
			p++;
		}
	}
	return lines;
}

static void dialog_open(Dialog *dialog, int npc_index, const char *npc_name) {
	dialog->open = 1;
	dialog->input_len = 0;
	dialog->input[0] = '\0';
	dialog->npc_index = npc_index;
	dialog->npc_name = npc_name;
}

static void dialog_close(Dialog *dialog) {
	dialog->open = 0;
	dialog->npc_index = -1;
	dialog->npc_name = NULL;
}

static void dialog_append(Dialog *dialog, uint32_t ch) {
	if (ch < 32 || ch > 126) {
		return;
	}
	if (dialog->input_len >= (int)(sizeof(dialog->input) - 1)) {
		return;
	}
	dialog->input[dialog->input_len++] = (char)ch;
	dialog->input[dialog->input_len] = '\0';
}

static void dialog_backspace(Dialog *dialog) {
	if (dialog->input_len <= 0) {
		return;
	}
	dialog->input_len--;
	dialog->input[dialog->input_len] = '\0';
}

static void trim_leading(char **text) {
	while (**text == ' ' || **text == '\t' || **text == '\n' || **text == '\r') {
		(*text)++;
	}
}

static void trim_leading_punct(char **text) {
	while (**text == '"' || **text == '\'' || **text == '`') {
		(*text)++;
		trim_leading(text);
	}
}

static void trim_trailing(char *text) {
	size_t len = strlen(text);
	while (len > 0) {
		char ch = text[len - 1];
		if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
			break;
		}
		text[len - 1] = '\0';
		len--;
	}
}

static void strip_any_prefix(char **text, const char *prefix) {
	if (strncasecmp(*text, prefix, strlen(prefix)) == 0) {
		*text += strlen(prefix);
		trim_leading(text);
	}
}


static char *sanitize_reply(char *reply, const char *name) {
	if (reply == NULL) {
		return NULL;
	}
	char *start = reply;
	trim_leading(&start);
	trim_leading_punct(&start);
	strip_any_prefix(&start, "Answer:");
	strip_any_prefix(&start, "NPC:");
	strip_any_prefix(&start, "Context:");
	strip_any_prefix(&start, "System:");
	if (strncmp(start, "<context>", 9) == 0) {
		start += 9;
		trim_leading(&start);
	}
	char *reminder = strstr(start, "<system-reminder>");
	if (reminder) {
		*reminder = '\0';
	}
	char *system_tag = strstr(start, "<system");
	if (system_tag) {
		*system_tag = '\0';
	}
	char *tag = strstr(start, "<|");
	if (tag) {
		*tag = '\0';
	}
	char *eos = strstr(start, "</s>");
	if (eos) {
		*eos = '\0';
	}
	char *hash = strstr(start, "###");
	if (hash) {
		*hash = '\0';
	}
	if (name && name[0] != '\0') {
		size_t name_len = strlen(name);
		for (;;) {
			if (strncasecmp(start, name, name_len) != 0) {
				break;
			}
			start += name_len;
			while (*start == ':' || *start == '-' || *start == ',') {
				start++;
			}
			trim_leading(&start);
			trim_leading_punct(&start);
		}
	}
	if (start != reply) {
		memmove(reply, start, strlen(start) + 1);
	}
	trim_trailing(reply);
	return reply;
}

static int find_substr_offset(const char *buf, int n, const char *needle) {
	int needle_len = (int)strlen(needle);
	if (needle_len <= 0 || n <= 0 || needle_len > n) {
		return -1;
	}
	for (int i = 0; i + needle_len <= n; i++) {
		int match = 1;
		for (int j = 0; j < needle_len; j++) {
			if (buf[i + j] != needle[j]) {
				match = 0;
				break;
			}
		}
		if (match) {
			return i;
		}
	}
	return -1;
}

static int find_stop_offset(const char *buf, int n) {
	int stop_at = n;
	for (int i = 0; i < n; i++) {
		if (buf[i] == '\n') {
			stop_at = i;
			break;
		}
	}
	int off = find_substr_offset(buf, n, "</s>");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "<system-reminder>");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "<system");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "<|");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "###");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "System:");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "User:");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	off = find_substr_offset(buf, n, "Assistant:");
	if (off >= 0 && off < stop_at) {
		stop_at = off;
	}
	return stop_at;
}

static void append_prompt_context(stringb *sb, const char *npc_name, const char *context,
		const char *question) {
	sb_append_cstr(sb, "Context:\n");
	if (npc_name && npc_name[0] != '\0') {
		sb_append_cstr(sb, "NPC Name: ");
		sb_append_cstr(sb, npc_name);
		sb_append_cstr(sb, "\n");
	}
	if (context && context[0] != '\0') {
		sb_append_cstr(sb, context);
	}
	sb_append_cstr(sb, "\nQuestion:\n");
	sb_append_cstr(sb, question ? question : "");
}

static char *build_prompt(const ModelConfig *cfg, const char *system, const char *npc_name,
		const char *context, const char *question) {
	stringb full = {0};
	sb_init(&full, 0);

	switch (cfg->prompt_style) {
		case PROMPT_STYLE_T5:
			sb_append_cstr(&full, "instruction: ");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\nquestion: ");
			sb_append_cstr(&full, question ? question : "");
			sb_append_cstr(&full, "\ncontext:\n");
			if (npc_name && npc_name[0] != '\0') {
				sb_append_cstr(&full, "NPC Name: ");
				sb_append_cstr(&full, npc_name);
				sb_append_cstr(&full, "\n");
			}
			if (context && context[0] != '\0') {
				sb_append_cstr(&full, context);
			}
			sb_append_cstr(&full, "\nanswer:");
			break;
		case PROMPT_STYLE_CHAT:
			sb_append_cstr(&full, "System:\n");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\nUser:\n");
			append_prompt_context(&full, npc_name, context, question);
			sb_append_cstr(&full, "\nAssistant:");
			break;
		case PROMPT_STYLE_PLAIN:
		default:
			sb_append_cstr(&full, "System:\n");
			sb_append_cstr(&full, system ? system : "");
			sb_append_cstr(&full, "\n");
			append_prompt_context(&full, npc_name, context, question);
			sb_append_cstr(&full, "\nAnswer:");
			break;
	}

	return full.data;
}

static char *generate_npc_reply(const GameRuntime *runtime, const GameMap *game_map,
		int npc_index, const char *prompt) {
	if (runtime == NULL || prompt == NULL) {
		return NULL;
	}
	const char *fallback = "Demo reply: The old ruins are north of here.";
	const char *npc_name = NULL;
	if (game_map && npc_index >= 0 && npc_index < 10) {
		const char *npc_reply = game_map->npcs[npc_index].reply;
		npc_name = game_map->npcs[npc_index].name;
		if (npc_reply && npc_reply[0] != '\0') {
			fallback = npc_reply;
		}
	}

	if (runtime->model == NULL || runtime->model_cfg == NULL || runtime->embed_ctx == NULL
			|| runtime->npc_dbs == NULL || runtime->npc_db_loaded == NULL) {
		return strdup(fallback);
	}
	if (npc_index < 0 || npc_index >= 10 || runtime->npc_db_loaded[npc_index] == 0) {
		return strdup(fallback);
	}

	VectorDB *db = &runtime->npc_dbs[npc_index];
	float query[VDB_EMBED_SIZE];
	int results[5];
	for (int i = 0; i < 5; i++) {
		results[i] = -1;
	}
	vdb_embed_query(db, prompt, query);
	vdb_search(db, query, 5, results);

	size_t context_cap = 1024;
	size_t context_len = 0;
	char *context = (char *)malloc(context_cap);
	if (context == NULL) {
		return strdup(fallback);
	}
	context[0] = '\0';
	if (runtime->verbose) {
		fprintf(stderr, "[npc] question: %s\n", prompt);
	}
	for (int i = 0; i < 5; i++) {
		if (results[i] < 0) {
			continue;
		}
		const char *text = db->docs[results[i]].text;
		if (runtime->verbose) {
			fprintf(stderr, "[npc] context[%d]: %s\n", i, text);
		}
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
				free(context);
				return strdup(fallback);
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

	const char *system_prompt = "You are a helpful NPC. Speak in first person. "
		"Use only the provided context. If the context does not contain the answer, say \"I don't know.\" "
		"If asked your name, answer with the NPC Name from the context. "
		"Do not mention context, system messages, or prompts. Reply with one short sentence.";

	char *full_prompt = build_prompt(runtime->model_cfg, system_prompt, npc_name, context, prompt);
	if (full_prompt == NULL) {
		free(context);
		return strdup(fallback);
	}
	free(context);

	if (runtime->verbose) {
		printf(">> %s\n", full_prompt);
	}

	const struct llama_vocab *vocab = llama_model_get_vocab(runtime->model);
	int n_prompt = -llama_tokenize(vocab, full_prompt, strlen(full_prompt), NULL, 0, true, true);
	llama_token *prompt_tokens = (llama_token *)malloc((size_t)n_prompt * sizeof(llama_token));
	if (prompt_tokens == NULL) {
		free(full_prompt);
		return strdup(fallback);
	}
	if (llama_tokenize(vocab, full_prompt, strlen(full_prompt), prompt_tokens, n_prompt, true, true) < 0) {
		free(full_prompt);
		free(prompt_tokens);
		return strdup(fallback);
	}

	struct llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = runtime->model_cfg->n_ctx;
	ctx_params.n_batch = runtime->model_cfg->n_batch;
	ctx_params.embeddings = false;

	struct llama_context *ctx = llama_init_from_model(runtime->model, ctx_params);
	if (ctx == NULL) {
		free(full_prompt);
		free(prompt_tokens);
		return strdup(fallback);
	}

	struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
	struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
	if (runtime->model_cfg->top_k > 0) {
		llama_sampler_chain_add(smpl, llama_sampler_init_top_k(runtime->model_cfg->top_k));
	}
	if (runtime->model_cfg->top_p > 0.0f && runtime->model_cfg->top_p < 1.0f) {
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(runtime->model_cfg->top_p, 1));
	}
	if (runtime->model_cfg->min_p > 0.0f) {
		llama_sampler_chain_add(smpl, llama_sampler_init_min_p(runtime->model_cfg->min_p, 1));
	}
	llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
				runtime->model_cfg->repeat_last_n,
				runtime->model_cfg->repeat_penalty,
				runtime->model_cfg->freq_penalty,
				runtime->model_cfg->presence_penalty));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(runtime->model_cfg->temperature));
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(runtime->model_cfg->seed));

	struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);

	if (llama_model_has_encoder(runtime->model)) {
		if (llama_encode(ctx, batch)) {
			llama_sampler_free(smpl);
			free(full_prompt);
			free(prompt_tokens);
			llama_free(ctx);
			return strdup(fallback);
		}
		llama_token decoder_start = llama_model_decoder_start_token(runtime->model);
		if (decoder_start == LLAMA_TOKEN_NULL) {
			decoder_start = llama_vocab_bos(vocab);
		}
		batch = llama_batch_get_one(&decoder_start, 1);
	}

	int n_pos = 0;
	llama_token new_token_id;
	size_t out_cap = 256;
	size_t out_len = 0;
	char *out = (char *)malloc(out_cap);
	if (out == NULL) {
		llama_sampler_free(smpl);
		free(full_prompt);
		free(prompt_tokens);
		llama_free(ctx);
		return strdup(fallback);
	}
	out[0] = '\0';
	int n_predict = runtime->model_cfg->n_predict > 0 ? runtime->model_cfg->n_predict : 64;
	if (n_predict > 64) {
		n_predict = 64;
	}
	while (n_pos + batch.n_tokens < n_prompt + n_predict) {
		if (llama_decode(ctx, batch)) {
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
			break;
		}
		int stop_at = find_stop_offset(buf, n);
		if (out_len == 0 && stop_at == 0 && n > 0 && buf[0] == '\n') {
			batch = llama_batch_get_one(&new_token_id, 1);
			continue;
		}
		if (out_len + (size_t)stop_at + 1 > out_cap) {
			while (out_len + (size_t)stop_at + 1 > out_cap) {
				out_cap *= 2;
			}
			char *next = (char *)realloc(out, out_cap);
			if (next == NULL) {
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

	llama_sampler_free(smpl);
	free(full_prompt);
	free(prompt_tokens);
	llama_free(ctx);

	if (out_len == 0) {
		free(out);
		return strdup(fallback);
	}
	return out;
}

static void dialog_submit(Dialog *dialog, const GameMap *game_map, const GameRuntime *runtime) {
	if (dialog->input_len == 0) {
		return;
	}
	{
		const char *npc_name = NULL;
		char *reply = generate_npc_reply(runtime, game_map, dialog->npc_index, dialog->input);
		const char *fallback = "";
		if (game_map && dialog->npc_index >= 0 && dialog->npc_index < 10) {
			npc_name = game_map->npcs[dialog->npc_index].name;
			fallback = game_map->npcs[dialog->npc_index].reply;
			if (fallback == NULL) {
				fallback = "";
			}
		}
		reply = sanitize_reply(reply, npc_name);
		if (reply == NULL || reply[0] == '\0') {
			free(reply);
			reply = NULL;
		}
		const char *reply_text = reply != NULL ? reply : fallback;
		if (dialog->entry_count >= DIALOG_HISTORY_MAX) {
			for (int i = 1; i < DIALOG_HISTORY_MAX; i++) {
				dialog->entries[i - 1] = dialog->entries[i];
			}
			dialog->entry_count = DIALOG_HISTORY_MAX - 1;
		}
		snprintf(dialog->entries[dialog->entry_count].prompt,
			sizeof(dialog->entries[dialog->entry_count].prompt), "%s", dialog->input);
		snprintf(dialog->entries[dialog->entry_count].response,
			sizeof(dialog->entries[dialog->entry_count].response), "%s", reply_text);
		dialog->entry_count++;
		free(reply);
	}
	dialog->input_len = 0;
	dialog->input[0] = '\0';
}

static void update_npc_status(const GameMap *game_map, int npc_index) {
	static char status_buf[128];
	const char *name = NULL;
	if (game_map && npc_index >= 0 && npc_index < 10) {
		name = game_map->npcs[npc_index].name;
	}
	if (name && name[0] != '\0') {
		snprintf(status_buf, sizeof(status_buf), "You approach %s.", name);
	} else {
		snprintf(status_buf, sizeof(status_buf), "You approach the NPC.");
	}
	update_status(status_buf);
}

static void render(const Map *map, const Player *player, int *cam_x,
		int *cam_y, int *out_view_w, int *out_view_h, const Dialog *dialog) {
	int w;
	int h;
	int map_x;
	int map_y;
	int map_w;
	int map_h;
	int side_x;
	int side_y;
	int side_w;
	int side_h;
	int stats_x;
	int stats_y;
	int stats_w;
	int stats_h;
	int inv_x;
	int inv_y;
	int inv_w;
	int inv_h;
	int msg1_y;
	int msg2_y;
	int view_w;
	int view_h;
	int draw_w;
	int draw_h;
	int pad_x;
	int pad_y;

	w = tb_width();
	h = tb_height();
	get_layout(w, h, &map_x, &map_y, &map_w, &map_h, &side_x, &side_y, &side_w, &side_h, &msg1_y, &msg2_y);

	tb_clear();
	if (w < MIN_W || h < MIN_H || map_w < 8 || map_h < 3) {
		tb_print(1, 1, COLOR_RED_256 | TB_BOLD, TB_DEFAULT, "Window too small. Resize to at least 40x12.");
		tb_present();
		*out_view_w = map_w;
		*out_view_h = map_h;
		return;
	}

	view_w = map_w - 2;
	view_h = map_h - 2;
	draw_w = view_w;
	draw_h = view_h;
	if (map->width < draw_w) {
		draw_w = map->width;
	}
	if (map->height < draw_h) {
		draw_h = map->height;
	}
	pad_x = view_w > draw_w ? (view_w - draw_w) / 2 : 0;
	pad_y = view_h > draw_h ? (view_h - draw_h) / 2 : 0;

	draw_border(map_x, map_y, map_w, map_h, COLOR_WHITE_256);
	update_camera(map, view_w, view_h, player, cam_x, cam_y);
	draw_map(map, map_x + 1 + pad_x, map_y + 1 + pad_y, draw_w, draw_h, player, *cam_x, *cam_y);

	stats_x = side_x;
	stats_y = side_y;
	stats_w = side_w;
	stats_h = 11;
	inv_x = side_x;
	inv_y = side_y + stats_h;
	inv_w = side_w;
	inv_h = side_h - stats_h;
	if (stats_w >= 12 && stats_h >= 9) {
		draw_border(stats_x, stats_y, stats_w, stats_h, COLOR_WHITE_256);
		draw_stats(stats_x + 2, stats_y + 1, player);
	}
	if (inv_w >= 12 && inv_h >= 7) {
		draw_border(inv_x, inv_y, inv_w, inv_h, COLOR_WHITE_256);
		draw_inventory(inv_x + 2, inv_y + 1, &player->inventory);
	}

	tb_print(2, msg1_y, COLOR_GREEN_256, TB_DEFAULT, status_msg);
	tb_print(2, msg2_y, COLOR_WHITE_256, TB_DEFAULT, "Move: arrows  Quit: q/ESC");

	if (dialog->open) {
		int box_w = map_w - 4;
		int box_h = 12;
		int box_x = map_x + 2;
		int box_y = map_y + map_h - box_h - 1;
		if (box_w > w - 2) {
			box_w = w - 2;
			box_x = 1;
		}
		if (box_h > h - 2) {
			box_h = h - 2;
			box_y = 1;
		}
		if (box_w < 20) {
			box_w = 20;
			box_x = map_x + 1;
		}
		if (box_y < map_y + 1) {
			box_y = map_y + 1;
		}
		for (int iy = 0; iy < box_h; iy++) {
			for (int ix = 0; ix < box_w; ix++) {
				tb_set_cell(box_x + ix, box_y + iy, ' ', COLOR_WHITE_256, 19);
			}
		}
		draw_border_bg(box_x, box_y, box_w, box_h, COLOR_WHITE_256, 19);
		{
			int input_y = box_y + box_h - 3;
			int footer_y = box_y + box_h - 2;
			int log_y = box_y + 1;
			int max_lines = input_y - log_y;
			int max_text = box_w - 4 - 5;
			int line = 0;

			if (max_text < 0) {
				max_text = 0;
			}
			int start = dialog->entry_count;
			if (start < 0) {
				start = 0;
			}
			int used_lines = 0;
			for (int i = dialog->entry_count - 1; i >= 0; i--) {
				const char *prompt_text = dialog->entries[i].prompt;
				const char *response_text = dialog->entries[i].response;
				const char *name = dialog->npc_name && dialog->npc_name[0] != '\0' ? dialog->npc_name : "NPC";
				char prefix_you[16];
				char prefix_npc[64];
				snprintf(prefix_you, sizeof(prefix_you), "You: ");
				snprintf(prefix_npc, sizeof(prefix_npc), "%s: ", name);
				int need = count_wrapped_lines(box_w, prefix_you, prompt_text)
					+ count_wrapped_lines(box_w, prefix_npc, response_text);
				if (used_lines + need > max_lines && used_lines > 0) {
					break;
				}
				used_lines += need;
				start = i;
				if (used_lines >= max_lines) {
					break;
				}
			}
			for (int i = start; i < dialog->entry_count && line + 1 <= max_lines; i++) {
				const char *prompt_text = dialog->entries[i].prompt;
				const char *response_text = dialog->entries[i].response;
				const char *name = dialog->npc_name && dialog->npc_name[0] != '\0' ? dialog->npc_name : "NPC";
				char prefix_you[16];
				char prefix_npc[64];
				snprintf(prefix_you, sizeof(prefix_you), "You: ");
				snprintf(prefix_npc, sizeof(prefix_npc), "%s: ", name);
				int used = draw_wrapped(box_x + 2, log_y + line, max_lines - line, box_w,
						COLOR_WHITE_256, 19, prefix_you, prompt_text);
				line += used;
				if (line >= max_lines) {
					break;
				}
				used = draw_wrapped(box_x + 2, log_y + line, max_lines - line, box_w,
						COLOR_GREEN_256, 19, prefix_npc, response_text);
				line += used;
				if (line >= max_lines) {
					break;
				}
			}

			tb_printf(box_x + 2, input_y, COLOR_WHITE_256, 19, "Say: %s", dialog->input);
			{
				int cursor_x = box_x + 2 + 5 + dialog->input_len;
				int cursor_y = input_y;
				if (cursor_x < box_x + box_w - 1) {
					tb_set_cell(cursor_x, cursor_y, '_', COLOR_WHITE_256 | TB_BOLD, 19);
				}
			}
			tb_print(box_x + 2, footer_y, COLOR_WHITE_256, 19, "Enter: send  ESC: close");
		}
	}

	tb_present();
	*out_view_w = view_w;
	*out_view_h = view_h;
}

static int clamp(int value, int min, int max) {
	if (value < min) {
		return min;
	}
	if (value > max) {
		return max;
	}
	return value;
}

int main(int argc, char **argv) {
	const char *model_name = NULL;
	const char *embed_model_name = NULL;
	const ModelConfig *model_cfg = NULL;
	struct llama_model *embed_model = NULL;
	struct llama_model *gen_model = NULL;
	struct llama_context *embed_ctx = NULL;
	int tb_ready = 0;
	int llama_ready = 0;
	int exit_code = 0;
	int verbose = 0;

	static struct option long_options[] = {
		{"model", required_argument, 0, 'm'},
		{"embed-model", required_argument, 0, 'e'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "m:e:vh", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'm':
				model_name = optarg;
				break;
			case 'e':
				embed_model_name = optarg;
				break;
			case 'v':
				verbose = 1;
				break;
			case 'h':
				show_help(argv[0]);
				return 0;
			default:
				fprintf(stderr, "Usage: %s [-m model] [-v] [-h]\n", argv[0]);
				return 1;
		}
	}

	if (model_name != NULL) {
		model_cfg = get_model_by_name(model_name);
		if (model_cfg == NULL) {
			fprintf(stderr, "Unknown model '%s'\n", model_name);
			return 1;
		}
	} else {
		model_cfg = &models[0];
	}

	Player player = {0};
	array(GameMap) maps;
	GameMap map1 = {0};
	GameMap *current_map = NULL;
	VectorDB *npc_dbs = NULL;
	int *npc_db_loaded = NULL;
	int running = 1;
	int view_w = 0;
	int view_h = 0;
	int cam_x = 0;
	int cam_y = 0;
	Dialog dialog = {0};
	GameRuntime runtime = {0};

	player_init(&player);
	array_init(maps);
	map1 = make_map1();
	array_push(maps, map1);
	current_map = &maps.data[0];
	map_init(&current_map->map, current_map->data, current_map->len);

	if (verbose == 0) {
		llama_log_set(llama_log_callback, NULL);
	}

	npc_dbs = (VectorDB *)calloc(10, sizeof(VectorDB));
	npc_db_loaded = (int *)calloc(10, sizeof(int));
	if (npc_dbs == NULL || npc_db_loaded == NULL) {
		fprintf(stderr, "Failed to allocate NPC vector databases\n");
		exit_code = 1;
		goto cleanup;
	}

	llama_backend_init();
	ggml_backend_load_all();
	llama_ready = 1;
	const ModelConfig *embed_cfg = NULL;
	if (embed_model_name != NULL) {
		embed_cfg = get_model_by_name(embed_model_name);
		if (embed_cfg == NULL) {
			fprintf(stderr, "Unknown embedding model '%s'\n", embed_model_name);
			exit_code = 1;
			goto cleanup;
		}
	} else if (model_cfg->embed_model_name != NULL) {
		embed_cfg = get_model_by_name(model_cfg->embed_model_name);
	}
	if (embed_cfg == NULL) {
		embed_cfg = model_cfg;
	}

	struct llama_model_params gen_params = llama_model_default_params();
	gen_params.n_gpu_layers = model_cfg->n_gpu_layers;
	gen_params.use_mmap = model_cfg->use_mmap;
	gen_model = llama_model_load_from_file(model_cfg->filepath, gen_params);
	if (gen_model == NULL) {
		fprintf(stderr, "Unable to load generation model\n");
		exit_code = 1;
		goto cleanup;
	}

	struct llama_model_params embed_params = llama_model_default_params();
	embed_params.n_gpu_layers = embed_cfg->n_gpu_layers;
	embed_params.use_mmap = embed_cfg->use_mmap;
	embed_model = llama_model_load_from_file(embed_cfg->filepath, embed_params);
	if (embed_model == NULL) {
		fprintf(stderr, "Unable to load embedding model\n");
		exit_code = 1;
		goto cleanup;
	}

	struct llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx = embed_cfg->n_ctx;
	cparams.n_batch = embed_cfg->n_batch;
	cparams.embeddings = true;
	embed_ctx = llama_init_from_model(embed_model, cparams);
	if (embed_ctx == NULL) {
		fprintf(stderr, "Failed to create embedding context\n");
		exit_code = 1;
		goto cleanup;
	}

	for (int i = 0; i < 10; i++) {
		const char *vdb_path = current_map->npcs[i].vdb_path;
		if (vdb_path == NULL || vdb_path[0] == '\0') {
			continue;
		}
		vdb_init(&npc_dbs[i], embed_ctx);
		VectorDBErrorCode vdb_rc = vdb_load(&npc_dbs[i], vdb_path);
		if (vdb_rc != VDB_SUCCESS) {
			fprintf(stderr, "Failed to load vector database %s: %s\n", vdb_path, vdb_error(vdb_rc));
			vdb_free(&npc_dbs[i]);
			continue;
		}
		npc_db_loaded[i] = 1;
	}

	runtime.model_cfg = model_cfg;
	runtime.model = gen_model;
	runtime.embed_model = embed_model;
	runtime.embed_ctx = embed_ctx;
	runtime.npc_dbs = npc_dbs;
	runtime.npc_db_loaded = npc_db_loaded;
	runtime.verbose = verbose;

	if (tb_init() != TB_OK) {
		fprintf(stderr, "Failed to init termbox.\n");
		exit_code = 1;
		goto cleanup;
	}
	tb_ready = 1;

	tb_set_input_mode(TB_INPUT_ESC);
	tb_set_output_mode(TB_OUTPUT_256);
	update_status("You feel like you have a lot of potential.");
	while (running) {
		struct tb_event ev;

		render(&current_map->map, &player, &cam_x, &cam_y, &view_w, &view_h, &dialog);
		tb_poll_event(&ev);
		if (ev.type == TB_EVENT_KEY) {
			if (dialog.open) {
				if (ev.key == TB_KEY_ESC) {
					dialog_close(&dialog);
				} else if (ev.key == TB_KEY_ENTER) {
					dialog_submit(&dialog, current_map, &runtime);
				} else if (ev.key == TB_KEY_BACKSPACE || ev.key == TB_KEY_BACKSPACE2) {
					dialog_backspace(&dialog);
				} else if (ev.ch) {
					dialog_append(&dialog, ev.ch);
				}
			} else {
				if (ev.key == TB_KEY_ESC || ev.ch == 'q') {
					running = 0;
				} else if (ev.key == TB_KEY_ARROW_UP) {
					int next_y = player.y - 1;
					u32 target = map_get(&current_map->map, player.x, next_y);
					int npc_index = npc_index_from_tile(target);
					if (target == 'N' || npc_index >= 0) {
						const char *npc_name = current_map && npc_index >= 0 && npc_index < 10
							? current_map->npcs[npc_index].name
							: NULL;
						dialog_open(&dialog, npc_index, npc_name);
						update_npc_status(current_map, npc_index);
					} else if (map_is_walkable(&current_map->map, player.x, next_y)) {
						player.y = next_y;
					}
				} else if (ev.key == TB_KEY_ARROW_DOWN) {
					int next_y = player.y + 1;
					u32 target = map_get(&current_map->map, player.x, next_y);
					int npc_index = npc_index_from_tile(target);
					if (target == 'N' || npc_index >= 0) {
						const char *npc_name = current_map && npc_index >= 0 && npc_index < 10
							? current_map->npcs[npc_index].name
							: NULL;
						dialog_open(&dialog, npc_index, npc_name);
						update_npc_status(current_map, npc_index);
					} else if (map_is_walkable(&current_map->map, player.x, next_y)) {
						player.y = next_y;
					}
				} else if (ev.key == TB_KEY_ARROW_LEFT) {
					int next_x = player.x - 1;
					u32 target = map_get(&current_map->map, next_x, player.y);
					int npc_index = npc_index_from_tile(target);
					if (target == 'N' || npc_index >= 0) {
						const char *npc_name = current_map && npc_index >= 0 && npc_index < 10
							? current_map->npcs[npc_index].name
							: NULL;
						dialog_open(&dialog, npc_index, npc_name);
						update_npc_status(current_map, npc_index);
					} else if (map_is_walkable(&current_map->map, next_x, player.y)) {
						player.x = next_x;
					}
				} else if (ev.key == TB_KEY_ARROW_RIGHT) {
					int next_x = player.x + 1;
					u32 target = map_get(&current_map->map, next_x, player.y);
					int npc_index = npc_index_from_tile(target);
					if (target == 'N' || npc_index >= 0) {
						const char *npc_name = current_map && npc_index >= 0 && npc_index < 10
							? current_map->npcs[npc_index].name
							: NULL;
						dialog_open(&dialog, npc_index, npc_name);
						update_npc_status(current_map, npc_index);
					} else if (map_is_walkable(&current_map->map, next_x, player.y)) {
						player.x = next_x;
					}
				}
				if (map_get(&current_map->map, player.x, player.y) == '$') {
					player.gold += 10;
					map_set(&current_map->map, player.x, player.y, MAP_FLOOR_CH);
					update_status("You pick up 10 gold.");
				}
			}
			player.x = clamp(player.x, 0, current_map->map.width > 1 ? current_map->map.width - 1 : 0);
			player.y = clamp(player.y, 0, current_map->map.height > 1 ? current_map->map.height - 1 : 0);
		} else if (ev.type == TB_EVENT_RESIZE) {
			player.x = clamp(player.x, 0, current_map->map.width > 1 ? current_map->map.width - 1 : 0);
			player.y = clamp(player.y, 0, current_map->map.height > 1 ? current_map->map.height - 1 : 0);
		}
	}

cleanup:
	player_free(&player);
	for (size_t i = 0; i < maps.length; i++) {
		map_free(&maps.data[i].map);
	}
	array_free(maps);
	if (tb_ready) {
		tb_shutdown();
	}
	for (int i = 0; i < 10; i++) {
		if (npc_db_loaded && npc_db_loaded[i]) {
			vdb_free(&npc_dbs[i]);
		}
	}
	free(npc_db_loaded);
	free(npc_dbs);
	if (embed_ctx != NULL) {
		llama_free(embed_ctx);
	}
	if (embed_model != NULL) {
		llama_model_free(embed_model);
	}
	if (gen_model != NULL) {
		llama_model_free(gen_model);
	}
	if (llama_ready) {
		llama_backend_free();
	}
	return exit_code;
}
