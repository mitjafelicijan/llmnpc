#include <stdio.h>

#define TB_IMPL
#include "termbox2.h"

#define MIN_W 40
#define MIN_H 12
#define SIDEBAR_W 40
#define CP_H 0x2500
#define CP_V 0x2502
#define CP_TL 0x250c
#define CP_TR 0x2510
#define CP_BL 0x2514
#define CP_BR 0x2518

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

static void draw_map(int map_x, int map_y, int map_w, int map_h, int px,
	int py) {
	if (px >= 0 && py >= 0) {
		tb_set_cell(map_x + px, map_y + py, '@', TB_WHITE | TB_BOLD, TB_DEFAULT);
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
	tb_set_cell(x, y, '[', TB_WHITE, TB_DEFAULT);
	for (ix = 0; ix < inner_w; ix++) {
		uintattr_t fg = ix < filled ? TB_GREEN : TB_WHITE;
		uint32_t ch = ix < filled ? '=' : ' ';
		tb_set_cell(x + 1 + ix, y, ch, fg, TB_DEFAULT);
	}
	tb_set_cell(x + w - 1, y, ']', TB_WHITE, TB_DEFAULT);
}

static void draw_stats(int x, int y) {
	tb_print(x, y, TB_WHITE | TB_BOLD, TB_DEFAULT, "Stats");
	tb_print(x, y + 2, TB_WHITE, TB_DEFAULT, "HP 12/12");
	draw_progress_bar(x, y + 3, 18, 12, 12);
	tb_print(x, y + 4, TB_WHITE, TB_DEFAULT, "AC: 7");
	tb_print(x, y + 5, TB_WHITE, TB_DEFAULT, "Str: 16");
	tb_print(x, y + 6, TB_WHITE, TB_DEFAULT, "Gold: 42");
}

static void draw_inventory(int x, int y) {
	tb_print(x, y, TB_WHITE | TB_BOLD, TB_DEFAULT, "Inventory");
	tb_print(x, y + 2, TB_WHITE, TB_DEFAULT, "a) dagger");
	tb_print(x, y + 3, TB_WHITE, TB_DEFAULT, "b) ration");
	tb_print(x, y + 4, TB_WHITE, TB_DEFAULT, "c) potion");
	tb_print(x, y + 5, TB_WHITE, TB_DEFAULT, "d) scroll");
}

static const char *status_msg = "";

static void update_status(const char *message) {
	status_msg = message ? message : "";
}

static void render(int px, int py, int *out_map_w, int *out_map_h) {
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

	w = tb_width();
	h = tb_height();
	get_layout(w, h, &map_x, &map_y, &map_w, &map_h, &side_x, &side_y, &side_w, &side_h, &msg1_y, &msg2_y);

	tb_clear();
	if (w < MIN_W || h < MIN_H || map_w < 8 || map_h < 3) {
		tb_print(1, 1, TB_RED | TB_BOLD, TB_DEFAULT, "Window too small. Resize to at least 40x12.");
		tb_present();
		*out_map_w = map_w;
		*out_map_h = map_h;
		return;
	}

	draw_border(map_x, map_y, map_w, map_h, TB_WHITE);
	draw_map(map_x + 1, map_y + 1, map_w - 2, map_h - 2, px, py);

	stats_x = side_x;
	stats_y = side_y;
	stats_w = side_w;
	stats_h = 11;
	inv_x = side_x;
	inv_y = side_y + stats_h;
	inv_w = side_w;
	inv_h = side_h - stats_h;
	if (stats_w >= 12 && stats_h >= 9) {
		draw_border(stats_x, stats_y, stats_w, stats_h, TB_WHITE);
		draw_stats(stats_x + 2, stats_y + 1);
	}
	if (inv_w >= 12 && inv_h >= 7) {
		draw_border(inv_x, inv_y, inv_w, inv_h, TB_WHITE);
		draw_inventory(inv_x + 2, inv_y + 1);
	}

	tb_print(2, msg1_y, TB_GREEN, TB_DEFAULT, status_msg);
	tb_print(2, msg2_y, TB_WHITE, TB_DEFAULT, "Move: arrows  Quit: q/ESC");

	tb_present();
	*out_map_w = map_w - 2;
	*out_map_h = map_h - 2;
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
	int px = 6;
	int py = 4;
	int running = 1;
	int map_w = 0;
	int map_h = 0;

	if (tb_init() != TB_OK) {
		fprintf(stderr, "Failed to init termbox.\n");
		return 1;
	}

	tb_set_input_mode(TB_INPUT_ESC);
	update_status("You feel like you have a lot of potential.");
	while (running) {
		struct tb_event ev;

		render(px, py, &map_w, &map_h);
		tb_poll_event(&ev);
		if (ev.type == TB_EVENT_KEY) {
			if (ev.key == TB_KEY_ESC || ev.ch == 'q') {
				running = 0;
			} else if (ev.key == TB_KEY_ARROW_UP) {
				py -= 1;
			} else if (ev.key == TB_KEY_ARROW_DOWN) {
				py += 1;
			} else if (ev.key == TB_KEY_ARROW_LEFT) {
				px -= 1;
			} else if (ev.key == TB_KEY_ARROW_RIGHT) {
				px += 1;
			}
			px = clamp(px, 0, map_w > 1 ? map_w - 1 : 0);
			py = clamp(py, 0, map_h > 1 ? map_h - 1 : 0);
		} else if (ev.type == TB_EVENT_RESIZE) {
			px = clamp(px, 0, map_w > 1 ? map_w - 1 : 0);
			py = clamp(py, 0, map_h > 1 ? map_h - 1 : 0);
		}
	}

	tb_shutdown();
	return 0;
}
