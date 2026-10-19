#ifndef FATE_GAME_H
#define FATE_GAME_H

/*
 * Prototypes for Banjo-Kazooie functions and data the mod uses that the
 * decomp doesn't declare in a shared header. Every name here exists in
 * BanjoRecompSyms (bk.us.rev0), and each signature is copied from its
 * definition in the pinned decomp commit.
 */

#include <ultra64.h>
#include "functions.h"
#include "variables.h"
#include "core1/ml.h"
#include "actor.h"
#include "prop.h"

/* code_7060.c / playerutils.c: player queries */
f32 player_getYaw(void);
enum bswatergroup_e player_getWaterState(void);
enum bsgroup_e player_movementGroup(void);
bool player_isDead(void);
bool player_is_present(void);
bool func_8028F1E0(void);          /* current state has an interrupt handler */
bool func_8028F098(void);          /* player isn't in an interrupt/locked group */
bool func_8028F25C(void);
s32 baiFrame_getState(void);
s32 func_80294554(void);           /* feet are below a water surface */
bool floor_isCurrentFloorunk59(void);
bool player_isStable(void);

/* level / map / game mode */
enum level_e level_get(void);
enum map_e map_get(void);
s32 exit_get(void);
s32 getGameMode(void);
bool func_802E4A08(void);          /* demo / replay / bonus game modes */
void func_80335110(s32 world_updates_enabled);
s32 func_8033511C(void);

/* save state */
bool fileProgressFlag_get(enum file_progress_e index);
s32 honeycombscore_get_total(void);
s32 jiggyscore_total(void);
s32 jiggyscore_leveltotal(s32 lvl);
bool gameFile_isNotEmpty(s32 gamenum);
s32 gameSelect_getGameNumber(void);

/* items (code_BEF20.c) */
extern s32 D_80385F30[0x2C];       /* item counters */
extern s32 D_80385FE4;             /* "player died" latch */
s32 item_getCount(enum item_e item);
s32 item_adjustByDiffWithHud(enum item_e item, s32 diff);
s32 item_empty(enum item_e item);
void item_inc(enum item_e item);
void item_set(s32 item, s32 val);
void func_802FACA4(enum item_e item_id);
s32 fxairscore_time_to_count(s32 time);
void func_80346DB4(s32 note_count);

/* dialog */
int gcdialog_hasCurrentTextId(void);

/* controller (pfsmanager.c): per-button held-frame counters */
void func_8024E6E0(s32 controller_index, s32 dst[4]);      /* d-pad up, down, left, right */
s32 controller_copySideButtons(s32 controller_index, s32 dst[3]); /* Z, L, R */

/* enemy hit table (code_B62B0.c) */
typedef struct {
    u16 markerId;
    u16 collisionUnion;
    u16 collisionTypes[11];
} FateHitTableRow;
extern FateHitTableRow D_80370AC0[0xBB];
extern s16 D_80383F10[0x2A9];      /* marker id -> hit table row, -1 if none */
s32 func_8033D594(struct5Cs *arg0); /* damage to player */
s32 func_8033D5B4(struct5Cs *arg0); /* drop bundle on death */
s32 func_8033D584(struct5Cs *arg0); /* honeycomb knocked out on hit */
s32 func_8033D5A4(struct5Cs *arg0); /* hits to trigger */

/* enemy damage application (code_9E370.c) */
extern ActorArray *suBaddieActorArray;
extern s32 D_8036E564;
Actor *__bundle_spawnFromFirstActor(enum bundle_e bundle_id, Actor *actor);
void func_8032B3A0(Actor *this, ActorMarker *arg1);
typedef bool (*FateNodeFilter)(NodeProp *, s32);
void func_8032EE0C(FateNodeFilter arg0, s32 arg1);
bool func_8032B38C(NodeProp *node, s32 arg1);
void func_8032EE20(void);
void __spawnQueue_add_5(void (*arg0)(void), s32 arg1, s32 arg2, s32 arg3, s32 arg4, s32 arg5);
Actor *spawnQueue_bundleWithYaw_f32(enum bundle_e bundle_id, s32 x, s32 y, s32 z, s32 yaw);
s32 func_803048E0(s32 arg0[3], s32 arg1[3], s32 arg2, s32 arg3, s32 arg4);
f32 func_8033229C(ActorMarker *marker);
void sfx_playFadeShorthand(u32 arg0, f32 position[3], u32 fadeDistance);

/* misc */
f32 gu_sqrtf(f32 value);
f32 func_80309B24(f32 arg0[3]);   /* floor height below a point */

#endif
