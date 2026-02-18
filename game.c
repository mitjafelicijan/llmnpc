#include <stdio.h>

#define TB_IMPL
#include "termbox2.h"

#define NONSTD_IMPLEMENTATION
#include "nonstd.h"

#include "maps/map1.h"

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

typedef struct {
	const unsigned char *data;
	int len;
	int width;
	int height;
	u32 *cells;
} Map;

static int clamp(int value, int min, int max);

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

static void draw_room(int x, int y, int w, int h, uintattr_t fg) {
	draw_border(x, y, w, h, fg);
	tb_set_cell(x + w / 2, y, '+', fg, TB_DEFAULT);
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
	return ch == MAP_FLOOR_CH || ch == '$';
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
			uintattr_t fg = COLOR_WHITE_256;
			if (ch == MAP_FLOOR_CH) {
				fg = MAP_FLOOR_FG;
			} else if (ch == '~') {
				fg = COLOR_BLUE_256;
			} else if (ch == '$') {
				fg = COLOR_ORANGE_256;
			} else if (ch == 'B' || ch == 'S' || ch == 'G') {
				fg = COLOR_RED_256;
			} else if (ch == 'N') {
				fg = COLOR_CYAN_256;
			} else if (ch >= MAP_BORDER_MIN && ch <= MAP_BORDER_MAX) {
				fg = COLOR_BORDER_256;
			}
			tb_set_cell(map_x + ix, map_y + iy, ch, fg, TB_DEFAULT);
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

static void render(const Map *map, const Player *player, int *cam_x,
	int *cam_y, int *out_view_w, int *out_view_h) {
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

int main(void) {
	Player player = {0};
	Map map = {0};
	int running = 1;
	int view_w = 0;
	int view_h = 0;
	int cam_x = 0;
	int cam_y = 0;

	player_init(&player);
	map_init(&map, maps_map1_txt, (int)maps_map1_txt_len);

	if (tb_init() != TB_OK) {
		fprintf(stderr, "Failed to init termbox.\n");
		return 1;
	}

	tb_set_input_mode(TB_INPUT_ESC);
	tb_set_output_mode(TB_OUTPUT_256);
	update_status("You feel like you have a lot of potential.");
	while (running) {
		struct tb_event ev;

		render(&map, &player, &cam_x, &cam_y, &view_w, &view_h);
		tb_poll_event(&ev);
		if (ev.type == TB_EVENT_KEY) {
			if (ev.key == TB_KEY_ESC || ev.ch == 'q') {
				running = 0;
			} else if (ev.key == TB_KEY_ARROW_UP) {
				int next_y = player.y - 1;
				if (map_is_walkable(&map, player.x, next_y)) {
					player.y = next_y;
				}
			} else if (ev.key == TB_KEY_ARROW_DOWN) {
				int next_y = player.y + 1;
				if (map_is_walkable(&map, player.x, next_y)) {
					player.y = next_y;
				}
			} else if (ev.key == TB_KEY_ARROW_LEFT) {
				int next_x = player.x - 1;
				if (map_is_walkable(&map, next_x, player.y)) {
					player.x = next_x;
				}
			} else if (ev.key == TB_KEY_ARROW_RIGHT) {
				int next_x = player.x + 1;
				if (map_is_walkable(&map, next_x, player.y)) {
					player.x = next_x;
				}
			}
			if (map_get(&map, player.x, player.y) == '$') {
				player.gold += 10;
				map_set(&map, player.x, player.y, MAP_FLOOR_CH);
				update_status("You pick up 10 gold.");
			}
			player.x = clamp(player.x, 0, map.width > 1 ? map.width - 1 : 0);
			player.y = clamp(player.y, 0, map.height > 1 ? map.height - 1 : 0);
		} else if (ev.type == TB_EVENT_RESIZE) {
			player.x = clamp(player.x, 0, map.width > 1 ? map.width - 1 : 0);
			player.y = clamp(player.y, 0, map.height > 1 ? map.height - 1 : 0);
		}
	}

	player_free(&player);
	map_free(&map);
	tb_shutdown();
	return 0;
}
