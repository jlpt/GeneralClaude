/*
 * chmyenemy.c - example custom enemy for the Banjo-Kazooie decomp
 *
 * A stationary enemy that:
 *   - idles and turns to face Banjo when he is nearby,
 *   - taunts (plays an attack/alert animation + sound) when he gets close,
 *   - says a line of dialog the first time Banjo walks up to it,
 *   - hurts Banjo on touch and dies to his attacks (both via collision_table),
 *   - puffs out steam and despawns when killed.
 *
 * It borrows the Grublin's model, animations and sounds so it works without
 * new assets. Swap the ASSET_* / SFX_* ids for your own.
 *
 * To use it (see chapter 4 of the guide):
 *   1. Copy to e.g. src/MM/ch/myenemy.inc.c and add
 *          #include "ch/myenemy.inc.c"
 *      at the bottom of src/MM/mmspawnqueue.c (or list it in the splat yaml).
 *   2. Add ACTOR_3CC_MY_ENEMY / MARKER_29F_MY_ENEMY to include/enums.h
 *      (the #defines below are only a fallback).
 *   3. Register it in MM_func_803888B0() (src/MM/mmspawnqueue.c):
 *          spawnableActorList_add(&chMyEnemyInfo, actor_new,
 *              ACTOR_FLAG_UNKNOWN_25 | ACTOR_FLAG_UNKNOWN_24 | ACTOR_FLAG_UNKNOWN_8);
 *   4. Add a row to collision_table in src/core2/code_B62B0.c and bump
 *      COLLISION_TABLE_SIZE:
 *          {MARKER_29F_MY_ENEMY, 0x0001, {0x0908, 0x0988, 0x8100, 0x0988, 0x0A04,
 *              0x0A04, 0x0A84, 0x0988, 0x0A04, 0x0000, 0x1040}},
 *   5. Place it (setup file node with actor id 0x3CC, or spawn it from code).
 *
 * This file compiles with the decomp's IDO flags against commit e8d31fa5.
 */
#include <ultra64.h>
#include "functions.h"
#include "variables.h"

/* Defined in core2 but not declared in a header */
extern bool subaddie_playerIsWithinCylinder(Actor *this, s32 radius, s32 d_y); /* src/core2/code_9E370.c */
extern enum bsgroup_e player_movementGroup(void);                              /* src/core2/code_7060.c */

#ifndef ACTOR_3CC_MY_ENEMY
#define ACTOR_3CC_MY_ENEMY  0x3CC
#endif
#ifndef MARKER_29F_MY_ENEMY
#define MARKER_29F_MY_ENEMY 0x29F
#endif

/* Tuning */
#define MYENEMY_NOTICE_RADIUS   1500   /* start turning toward Banjo */
#define MYENEMY_TAUNT_RADIUS     500   /* taunt when this close */
#define MYENEMY_TALK_RADIUS      350   /* first-meeting dialog */
#define MYENEMY_TURN_SPEED         6   /* passed to subaddie_turnToYaw */
#define MYENEMY_TAUNT_COOLDOWN   3.0f  /* seconds between taunts */

void chMyEnemy_update(Actor *this);

/* The actor state is also the index into this table (see subaddie_set_state). */
enum chmyenemy_state_e {
    MYENEMY_STATE_0_NONE,
    MYENEMY_STATE_1_IDLE,
    MYENEMY_STATE_2_TAUNT,
    MYENEMY_STATE_3_DIE
};

ActorAnimationInfo chMyEnemyAnimations[] = {
    {0, 0.0f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK, 2.5f},   /* MYENEMY_STATE_1_IDLE  */
    {ASSET_64_ANIM_GRUBLIN_ALERT,     0.9f},   /* MYENEMY_STATE_2_TAUNT */
    {ASSET_94_ANIM_GRUBLIN_DIE,       1.0f},   /* MYENEMY_STATE_3_DIE   */
};

ActorInfo chMyEnemyInfo = {
    MARKER_29F_MY_ENEMY, ACTOR_3CC_MY_ENEMY, ASSET_3C5_MODEL_GRUBLIN,
    MYENEMY_STATE_1_IDLE, chMyEnemyAnimations,
    chMyEnemy_update, actor_update_func_80326224, actor_draw,
    2500,      /* only update while Banjo is within 2500 units */
    0,         /* default draw distance */
    1.0f,      /* shadow size */
    0
};

/* Per-instance data lives in Actor::local (about 0x70 bytes available). */
typedef struct {
    f32 taunt_cooldown;
} ActorLocal_MyEnemy;

/* Shown once per overlay load. Use a file progress flag to remember it in the save. */
static bool sChMyEnemyHasTalked;

/* .code */
static void __chMyEnemy_ow(ActorMarker *marker, ActorMarker *other) {
    /* Hit, but the collision_table row said it isn't dead yet (multi-hit enemies) */
    Actor *this = marker_getActor(marker);

    sfx_playFadeShorthandDefault(SFX_1F_HITTING_AN_ENEMY_3, 1.0f, 28000, this->position, 1250, 2500);
}

static void __chMyEnemy_die(ActorMarker *marker, ActorMarker *other) {
    Actor *this = marker_getActor(marker);

    subaddie_set_state(this, MYENEMY_STATE_3_DIE);
    actor_playAnimationOnce(this);
    actor_collisionOff(this);
    this->marker->propPtr->unk8_3 = FALSE;
    sfx_playFadeShorthandDefault(SFX_C2_GRUBLIN_EGH, 1.0f, 20000, this->position, 1250, 2500);

    /* Spawns are deferred through the spawn queue while inside callbacks (see chBeehive_die) */
    __spawnQueue_add_4((GenFunction_4)spawnQueue_actor_f32, ACTOR_4C_STEAM,
        reinterpret_cast(s32, this->position[0]),
        reinterpret_cast(s32, this->position[1]),
        reinterpret_cast(s32, this->position[2]));

    /* Reward example: jiggy_spawn(JIGGY_xx_..., this->position); */
}

static void __chMyEnemy_faceBanjo(Actor *this) {
    this->yaw_ideal = (f32) subaddie_getYawToPlayer(this);
    subaddie_turnToYaw(this, MYENEMY_TURN_SPEED);
}

static void __chMyEnemy_tryFirstMeetDialog(Actor *this) {
    if (sChMyEnemyHasTalked) {
        return;
    }
    if (!subaddie_playerIsWithinCylinder(this, MYENEMY_TALK_RADIUS, 300)) {
        return;
    }
    /* Only talk to Banjo when he's standing or walking normally */
    if (player_getTransformation() != TRANSFORM_1_BANJO) {
        return;
    }
    if (player_movementGroup() != BSGROUP_0_NONE) {
        return;
    }
    /* Replace with your own dialog asset (see chapter 8) */
    if (gcdialog_showDialog(ASSET_D96_DIALOG_BEEHIVE_MEET, 0, NULL, NULL, NULL, NULL)) {
        sChMyEnemyHasTalked = TRUE;
    }
}

void chMyEnemy_update(Actor *this) {
    ActorLocal_MyEnemy *local = (ActorLocal_MyEnemy *) &this->local;

    /* Per-load setup. volatile_initialized is cleared when the map's actor
     * savestate is restored, so register callbacks here, not in "initialized". */
    if (!this->volatile_initialized) {
        this->volatile_initialized = TRUE;
        marker_setCollisionScripts(this->marker, __chMyEnemy_ow, NULL, __chMyEnemy_die);
        this->marker->propPtr->unk8_3 = TRUE;   /* vanilla actors set this on init */
    }

    /* Once-per-spawn setup */
    if (!this->initialized) {
        this->initialized = TRUE;
        local->taunt_cooldown = 0.0f;
        subaddie_set_state(this, MYENEMY_STATE_1_IDLE);
        actor_loopAnimation(this);
    }

    if (local->taunt_cooldown > 0.0f) {
        local->taunt_cooldown -= time_getDelta();
    }

    switch (this->state) {
        case MYENEMY_STATE_1_IDLE:
            if (subaddie_playerIsWithinCylinder(this, MYENEMY_NOTICE_RADIUS, 600)) {
                __chMyEnemy_faceBanjo(this);
            }
            __chMyEnemy_tryFirstMeetDialog(this);
            if (local->taunt_cooldown <= 0.0f
                && subaddie_playerIsWithinCylinder(this, MYENEMY_TAUNT_RADIUS, 300)
            ) {
                subaddie_set_state(this, MYENEMY_STATE_2_TAUNT);
                actor_playAnimationOnce(this);
                sfx_playFadeShorthandDefault(SFX_29_GRUBLIN_NYAHAHA, 1.0f, 22000, this->position, 1250, 2500);
                local->taunt_cooldown = MYENEMY_TAUNT_COOLDOWN;
            }
            break;

        case MYENEMY_STATE_2_TAUNT:
            __chMyEnemy_faceBanjo(this);
            if (anctrl_isStopped(this->anctrl)) {
                subaddie_set_state(this, MYENEMY_STATE_1_IDLE);
                actor_loopAnimation(this);
            }
            break;

        case MYENEMY_STATE_3_DIE:
            if (anctrl_isStopped(this->anctrl)) {
                marker_despawn(this->marker);
            }
            break;
    }
}
