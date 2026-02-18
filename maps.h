#ifndef MAPS_H
#define MAPS_H

#include "nonstd.h"

#include "maps/map1.h"

typedef struct {
	const unsigned char *data;
	int len;
	int width;
	int height;
	u32 *cells;
} Map;

typedef struct {
	const char *name;
	const char *reply;
} NpcSettings;

typedef struct {
	const unsigned char *data;
	int len;
	Map map;
	NpcSettings npcs[10];
} GameMap;

static inline GameMap make_map1(void) {
	GameMap map = {0};
	map.data = maps_map1_txt;
	map.len = (int)maps_map1_txt_len;
	map.npcs[0] = (NpcSettings){.name = "Bromm", .reply = "Bromm: The old ruins are north of here."};
	map.npcs[1] = (NpcSettings){.name = "Dagna", .reply = "Dagna: The well is safe, mostly."};
	map.npcs[2] = (NpcSettings){.name = "Keldor", .reply = "Keldor: I saw lights in the marsh last night."};
	map.npcs[3] = (NpcSettings){.name = "Thrain", .reply = "Thrain: Mind the bridge; the beams sing when they're weak."};
	map.npcs[4] = (NpcSettings){.name = "Skara", .reply = "Skara: If you hear bells in the fog, turn back."};
	return map;
}

#endif
