# 4. Custom Enemies and Actors

[← Assets](03-assets.md) · [Index](README.md) · [Next: Levels and maps →](05-levels-and-maps.md)

This chapter walks through everything needed to put a new enemy (or any
interactive object) into the game: the data structures, the IDs to reserve,
the tables to register in, how it fights the player, and how to place it in a
level.

**Checklist** (details below):

1. Pick a free **actor ID** (`0x3CC`–`0x3D4`) and **marker ID** (`0x29F`–`0x2A8`) and
   add them to `include/enums.h`.
2. Write the actor: an `ActorAnimationInfo[]`, an `ActorInfo`, and an update
   function.
3. **Register** it with `spawnableActorList_add()` in the right spawn list.
4. Add a row to the enemy **hit table** `collision_table` (`src/core2/code_B62B0.c`).
5. **Place** it: in a setup file, or spawn it from code.
6. Build with `ANTI_TAMPER=0 ANTI_PIRACY=0` and test.

## 4.1 Anatomy of a vanilla enemy: the Grublin

`src/MM/ch/grublin.c` is the smallest real enemy in the game. Annotated:

```c
/* Animation table. The INDEX IN THIS ARRAY IS THE ACTOR STATE.
 * subaddie_set_state(actor, 3) switches to state 3 and plays entry [3]. */
ActorAnimationInfo chGrublinAnimations[11] = {
    {0, 0.0f},                                  // state 0: unused / no anim
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,  4.0f},   // state 1 (asset id, duration in seconds)
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,  0.7f},   // state 2
    {ASSET_63_ANIM_GRUBLIN_CHASE,      0.7f},   // state 3
    {ASSET_64_ANIM_GRUBLIN_ALERT,      0.9f},   // state 4
    {ASSET_94_ANIM_GRUBLIN_DIE,        1.0f},   // state 5 = CHHUMANOIDBADDIE_STATE_5_DIE
    ...
};

ActorInfo chgrublinInfo = {
    MARKER_5_GRUBLIN, ACTOR_6_GRUBLIN, ASSET_3C5_MODEL_GRUBLIN, // marker, actor, model
    1, chGrublinAnimations,                                     // start state, anim table
    chGrublin_update, actor_update_func_80326224, actor_draw,   // update, update2, draw
    2500, 0, 1.0f, 0                                            // update dist, draw dist, shadow, cheat flag
};

void chGrublin_die(ActorMarker *this, s32 arg1){        // called when the hit table says "die"
    Actor *actorPtr = marker_getActor(this);
    subaddie_set_state_with_direction(actorPtr, CHHUMANOIDBADDIE_STATE_5_DIE, 0.0f, 1);
    actor_playAnimationOnce(actorPtr);
    sfx_playFadeShorthandDefault(SFX_C2_GRUBLIN_EGH, 1.0f, 32000, actorPtr->position, 1250, 2500);
    actor_collisionOff(actorPtr);
}

void chGrublin_initialize(Actor *this){                  // configure the shared "humanoid baddie" AI
    Humanoid_Baddies_Actor *local = (Humanoid_Baddies_Actor *)&this->local;
    local->unk0 = 1.5f;  local->unk4 = 2.3f;               // wander timing range
    local->unk8 = 5;     local->unk9 = 7;
    local->yaw = 2;      local->unkA = 9;  local->unkB = 9; // turn/chase tuning
    local->foundPlayerSfx = SFX_29_GRUBLIN_NYAHAHA;
    local->foundPlayerSampleRate = 25000;
    local->unkC_28 = TRUE;
    local->hitFunction = NULL;
    local->dieFunction = chGrublin_die;
    local->foundPlayerVolume = 1.0f;
}

void chGrublin_update(Actor *this) {
    if (!this->volatile_initialized){        // first frame (and after savestate restore)
        chGrublin_initialize(this);
    }
    humanoidBaddie_update(this);              // shared walk/chase/attack/die AI (src/core2/code_53C10.c)
    ...                                       // extra per-state sound effects
}
```

It's registered in `src/MM/mmspawnqueue.c`:

```c
spawnableActorList_add(&chgrublinInfo, actor_new,
    ACTOR_FLAG_UNKNOWN_25 | ACTOR_FLAG_UNKNOWN_8 | ACTOR_FLAG_UNKNOWN_5 | ACTOR_FLAG_UNKNOWN_0);
```

Its hit rules are in `src/core2/code_B62B0.c`:

```c
{MARKER_5_GRUBLIN, 0x0001, {0x0904, 0x0984, 0x8100, 0x0984, 0x0A04, 0x0A04, 0x0A84, 0x0984, 0x0A04, 0x0000, 0x1020}},
```

and it's placed in Mumbo's Mountain's setup file (asset `0x71E`) as actor
nodes with actor ID `6`.

## 4.2 `ActorInfo` field by field

From `include/prop.h`:

```c
typedef struct actor_info_s{
    s16     markerId;        // enum marker_e: runtime type (collision, hit table)
    s16     actorId;         // enum actor_e: what setup files / spawn calls use
    s16     modelId;         // enum asset_e model, or 0 for no model (invisible controllers)
    s16     startAnimation;  // initial actor STATE (index into animations[])
    ActorAnimationInfo* animations; // NULL if no animations
    void    (* update_func)(Actor *);   // your logic, every frame
    void    (* update2_func)(Actor *);  // usually actor_update_func_80326224 (shared housekeeping) or NULL
    Actor*  (* draw_func)(ActorMarker *, Gfx **, Mtx **, Vtx **); // usually actor_draw
    u16     unk18;           // UPDATE distance: actor only updates while the player is within this (0 = always)
    u16     draw_distance;   // draw-distance multiplier: 0x400 = 1.0x, 0 = default
    f32     shadow_scale;    // blob shadow size; 0.0f = no shadow
    u16     unk20;           // a volatile flag; if set while a cheat is active, the actor despawns (usually 0)
} ActorInfo;
```

Notes:

- The update distance (`unk18`) is checked in `src/core2/code_9E370.c`.
  Enemies use 2000–3000. Controllers that must always run use 0.
- `draw_distance` is used in `src/core2/code_A5BC0.c`
  (`percentage = draw_dist / 0x400`).
- `actor_draw` handles models with skeletal animation automatically.

## 4.3 The `Actor` you get each frame

Commonly used fields (`include/prop.h`):

| Field | Use |
|---|---|
| `position[3]` (`position_x/y/z`) | World position (floats) |
| `yaw`, `yaw_ideal`, `pitch`, `roll` | Rotation in degrees. `subaddie_turnToYaw(this, step)` turns `yaw` toward `yaw_ideal`. |
| `velocity[3]` | Free to use for your own physics |
| `state` (6 bits, 0–63) | Your state machine. Set it with `subaddie_set_state()` to also switch animations. |
| `anctrl` | Animation controller (`anctrl_isStopped`, `anctrl_getAnimTimer`, `anctrl_setDuration`, …) |
| `scale` | Model scale (setup nodes can set it) |
| `spawn_position[3]` | Where it spawned (handy for "return home") |
| `volatile_initialized`, `initialized` | First-frame init flags. `volatile_initialized` is also cleared when the map's actor savestate is restored, so put per-load setup (collision callbacks, sound sources) behind it. |
| `secondaryId`, `actorTypeSpecificField` | Per-placement parameters copied from the setup node (`unk10_31`, `selector_or_radius`). Use them to make one actor type behave differently per placement. |
| `local` (~0x70 bytes) | Your private struct: `MyLocal *local = (MyLocal *)&this->local;` |
| `marker` | The `ActorMarker`: collision callbacks, `collidable`, `id` |
| `lifetime_value`, `actor_specific_1_f`, `unk1C[3]` | Misc. floats many actors repurpose |

## 4.4 States and animations

`subaddie_set_state(this, N)` (`src/core2/code_9E370.c`) sets `this->state = N`
and, if `animations[N].index != 0`, loads that animation with that duration and
starts it. The state number and the animation table index are the same thing.
Plan your enum so they line up:

```c
enum chmyenemy_state_e {
    MYENEMY_STATE_0_NONE,
    MYENEMY_STATE_1_IDLE,
    MYENEMY_STATE_2_CHASE,
    MYENEMY_STATE_3_ATTACK,
    MYENEMY_STATE_4_DIE,
};
ActorAnimationInfo chMyEnemyAnimations[] = {
    {0, 0.0f},
    {ASSET_xxx_ANIM_IDLE,   1.2f},   // state 1
    {ASSET_xxx_ANIM_WALK,   0.8f},   // state 2
    {ASSET_xxx_ANIM_ATTACK, 0.6f},   // state 3
    {ASSET_xxx_ANIM_DIE,    1.0f},   // state 4
};
```

Helpers:

| Function | Does |
|---|---|
| `subaddie_set_state(a, s)` | Set state + start its animation |
| `subaddie_set_state_forward(a, s)` | Same, explicitly forward from 0 |
| `subaddie_set_state_with_direction(a, s, start, dir)` | Start at a position (0–1) and direction |
| `actor_playAnimationOnce(a)` / `actor_loopAnimation(a)` | Playback mode |
| `actor_animationIsAt(a, 0.5f)` | True on the frame the animation crosses that point. Use it for footsteps, attack frames and sounds. |
| `anctrl_isStopped(a->anctrl)` | One-shot animation finished |

> Reusing vanilla animations only works with a model that has a **compatible
> skeleton**. A custom model needs its own animations, or a vanilla skeleton.

## 4.5 Choosing IDs

From scanning every `ActorInfo` in the decomp:

| ID kind | Highest vanilla | Safe range for mods | Hard limits |
|---|---|---|---|
| Actor (`actor_e`) | `0x3CA` (enum lists `0x3CB`) | **`0x3CC`–`0x3D4`** (9 slots) | The model cache in `src/core2/code_A5BC0.c` is indexed by actor ID and has `AssetCacheSize` (`0x3D5`) entries. `u16` in setup nodes, `s16` in `ActorInfo` |
| Marker (`marker_e`) | `0x29E` | **`0x29F`–`0x2A8`** (10 slots) | `ActorMarker.id` is 10 bits (`≤ 0x3FF`). The hit-table index array `D_80383F10` in `src/core2/code_B62B0.c` has `0x2A9` entries. |

An actor ID of `0x3D5` or higher spawns fine but crashes the first time the
actor is drawn: `marker_loadModelBin()` reads and writes past the end of
`modelCache`. To use more actor IDs, raise `AssetCacheSize` in
`src/core2/code_A5BC0.c`. A recomp mod can't change it, so recomp mods are
limited to these nine.

To use more than 10 new marker IDs, enlarge `D_80383F10[0x2A9]` and the `0x2A9`
loop bound in `func_8033D2F4()` in the same file. Without that, a marker ID ≥
`0x2A9` makes `func_8033D410()` read out of bounds on every player contact.

You can also **share** a marker ID between several of your actors if they
should take hits the same way. Only the hit table and the `ba/marker.c`
special cases care about it.

Add the names:

```c
// include/enums.h, end of enum actor_e
    ACTOR_3CB_UNKNOWN,
    ACTOR_3CC_MY_ENEMY       // new
};
// end of enum marker_e
    MARKER_29E_BIGBUTT_KNOCKED_DOWN,
    MARKER_29F_MY_ENEMY      // new
};
```

## 4.6 Writing the actor

### Option A: reskin the "humanoid baddie" AI (fastest)

Grublins, Gruntlings, Mum-Mums, the RBB/CCW Grublin variants and others all
run `humanoidBaddie_update()` (`src/core2/code_53C10.c`). It handles wandering,
noticing the player, chasing, attacking, terrain following, enemy boundaries
and dying. You give it a model, animations, sounds and tuning numbers:

```c
#include <ultra64.h>
#include "functions.h"
#include "variables.h"
#include "actor.h"

extern void humanoidBaddie_update(Actor *this);

void chMyGrunt_update(Actor *this);

/* Must have 11 entries laid out like the Grublin's (the AI uses states 1-10) */
ActorAnimationInfo chMyGruntAnimations[11] = {
    {0, 0.0f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,       4.0f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,       0.7f},
    {ASSET_63_ANIM_GRUBLIN_CHASE,           0.5f},   // faster chase than vanilla
    {ASSET_64_ANIM_GRUBLIN_ALERT,           0.9f},
    {ASSET_94_ANIM_GRUBLIN_DIE,             1.0f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,       1.5f},
    {ASSET_63_ANIM_GRUBLIN_CHASE,           0.35f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK,       1.5f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK, 1000000.0f},
    {ASSET_62_ANIM_GRUBLIN_IDLE_WALK, 1000000.0f}
};

ActorInfo chMyGruntInfo = {
    MARKER_29F_MY_ENEMY, ACTOR_3CC_MY_ENEMY, ASSET_3C5_MODEL_GRUBLIN, // swap the model for a recolor/new one
    1, chMyGruntAnimations,
    chMyGrunt_update, actor_update_func_80326224, actor_draw,
    2500, 0, 1.2f, 0
};

static void __chMyGrunt_die(ActorMarker *marker, ActorMarker *other){
    Actor *this = marker_getActor(marker);
    subaddie_set_state_with_direction(this, CHHUMANOIDBADDIE_STATE_5_DIE, 0.0f, 1);
    actor_playAnimationOnce(this);
    sfx_playFadeShorthandDefault(SFX_C2_GRUBLIN_EGH, 0.8f, 20000, this->position, 1250, 2500);
    actor_collisionOff(this);
}

void chMyGrunt_update(Actor *this){
    Humanoid_Baddies_Actor *local = (Humanoid_Baddies_Actor *)&this->local;

    if(!this->volatile_initialized){
        local->unk0 = 1.5f;  local->unk4 = 2.3f;
        local->unk8 = 5;     local->unk9 = 7;
        local->yaw  = 2;     local->unkA = 12;  local->unkB = 12;   // a bit more aggressive
        local->foundPlayerSfx        = SFX_29_GRUBLIN_NYAHAHA;
        local->foundPlayerSampleRate = 18000;                      // deeper voice
        local->foundPlayerVolume     = 1.0f;
        local->unkC_28      = TRUE;
        local->hitFunction  = NULL;
        local->dieFunction  = __chMyGrunt_die;
    }
    humanoidBaddie_update(this);
}
```

Register it with the **same flags as the Grublin** (see 4.7) and copy the
Grublin's hit-table row (see 4.8) with tweaks.

### Option B: write your own logic from scratch

For anything that isn't a walker (turrets, flyers, bosses, switches, NPCs),
write your own state machine. A complete, commented example is in
[`examples/chmyenemy.c`](examples/chmyenemy.c): a stationary enemy that turns
to face Banjo, attacks when close, dies to attacks, says a line of dialog the
first time you meet it, and rewards a note when killed.

The core pattern used by every vanilla actor:

```c
void chMyThing_update(Actor *this){
    MyLocal *local = (MyLocal *)&this->local;

    if(!this->volatile_initialized){
        this->volatile_initialized = TRUE;
        marker_setCollisionScripts(this->marker, __ow, __hit, __die);   // see 4.8
        // allocate sound sources, read this->secondaryId, etc.
    }

    if(!this->initialized){            // once per spawn (survives savestates)
        this->initialized = TRUE;
        // pick initial state, maybe based on a saved flag
    }

    switch(this->state){
        case MY_STATE_1_IDLE:  ...; break;
        case MY_STATE_2_CHASE: ...; break;
        case MY_STATE_4_DIE:
            if(anctrl_isStopped(this->anctrl))
                marker_despawn(this->marker);
            break;
    }
}
```

### Useful helpers for actor logic

| Helper | Where | Does |
|---|---|---|
| `player_getPosition(f32 dst[3])` | functions.h | Player position |
| `ml_vec3f_distance(a, b)` | core1/ml.h | Distance |
| `subaddie_getYawToPlayer(this)` | functions.h | Yaw (degrees) from actor to player |
| `subaddie_turnToYaw(this, step)` | functions.h | Rotate `yaw` toward `yaw_ideal` |
| `subaddie_playerIsWithinCylinder(this, radius, height)` | `code_9E370.c` (declare `extern`) | Proximity test |
| `subaddie_playerIsWithinSphereAndActive(this, dist)` | `code_9E370.c` (declare `extern`) | Sphere test that also requires the player to be "active" (`func_8028F098()`, `VOLATILE_FLAG_BF` clear) |
| `ml_vec3f_yaw_rotate_copy(dst, src, yaw)` | core1/ml.h | Rotate a vector. Forward motion = rotate `{dist,0,0}` by `yaw - 90` (as `func_802DAF2C` does). |
| `player_getTransformation()` | functions.h | `TRANSFORM_1_BANJO`, `_2_TERMITE`, `_3_PUMPKIN`, `_4_WALRUS`, `_5_CROC`, `_6_BEE` |
| `func_8030E6D4(SFX_xx)` | functions.h | Play a 2D sound effect |
| `sfx_playFadeShorthandDefault(sfx, vol, rate, pos, near, far)` | functions.h (macro) | Positional sound effect |
| `gcdialog_showDialog(text_id, 0, NULL, NULL, NULL, NULL)` | functions.h | Show dialog. Returns TRUE if shown. |
| `jiggy_spawn(JIGGY_xx, pos)` | functions.h | Spawn a jiggy |
| `actor_spawnWithYaw_f32(ACTOR_xx, pos, yaw)` | functions.h | Spawn another actor (must be in the spawnable list) |
| `__spawnQueue_add_4((GenFunction_4)spawnQueue_actor_f32, ACTOR_xx, reinterpret_cast(s32, x), …)` | | Deferred spawn (safe inside update/collision callbacks; see `chBeehive_die`) |
| `timedFunc_set_0/1/2/3(seconds, func, args…)` | | Call something later |
| `fileProgressFlag_get/set`, `levelSpecificFlags_get/set`, `mapSpecificFlags_get/set` | | Persistence ([2.7](02-architecture.md#27-the-four-flag-systems)) |
| `marker_despawn(marker)` | functions.h | Remove the actor |
| `actor_collisionOff/On(this)` | functions.h | Toggle collision |

When a function you need isn't in a header, find its definition in `src/`
and add an `extern` prototype at the top of your file. Vanilla files do this
all the time.

## 4.7 Registering the actor (spawnable actor lists)

Setup files and `actor_spawnWithYaw_*()` find actors through a per-map
**spawnable actor list**. It's rebuilt on every map load by `spawnQueue_reset()`
in `src/core2/spawnqueue.c`:

1. A long list of **global** actors (`spawnableActorList_add(&chJiggy, …)`,
   collectibles, Mumbo, Bottles, common enemies like Bigbutt).
2. Some enemies are added with `spawnableActorList_addIfMapVisited(&info, actor_new, flags, MAP_xx)`.
   They're always added outside the lair. **Inside the lair** they're only
   added if you've visited that map (so Grunty's lair only contains enemies
   from worlds you've entered).
3. Then a `switch(loaded_asm_file)` calls the **level overlay's** list:
   `MM_func_803888B0()` (`src/MM/mmspawnqueue.c`), `code26D0_resetSpawnableActorsForTTC()`,
   `bgs_updateSpawnableActors()`, `lair_func_8038A0C4()`, and so on.

Where to register:

| Your actor's code lives in… | Register in… |
|---|---|
| core2 (usable everywhere) | The global list in `spawnQueue_reset()` |
| A level overlay (e.g. `src/MM/ch/…`) | That overlay's `*spawnqueue.c` function |

```c
// src/MM/mmspawnqueue.c
extern ActorInfo chMyGruntInfo;
...
spawnableActorList_add(&chMyGruntInfo, actor_new,
    ACTOR_FLAG_UNKNOWN_25 | ACTOR_FLAG_UNKNOWN_8 | ACTOR_FLAG_UNKNOWN_5 | ACTOR_FLAG_UNKNOWN_0);
```

> Don't register an overlay actor in core2's global list. Its `ActorInfo` lives
> in overlay memory and is garbage when another overlay is loaded.

### Spawn flags

The third argument goes to `actor_new()` (`src/core2/code_9E370.c`), which sets
up optional features per flag. Most are still named `ACTOR_FLAG_UNKNOWN_n`.
**Copy the flags of the vanilla actor most like yours.** Observed effects:

| Flag | Effect in `actor_new()` | Typical users |
|---|---|---|
| `UNKNOWN_0` | Finds the **enemy boundary** region (setup category 7 node) around the spawn point, keeping walkers inside their area | Grublin, Bigbutt, most walkers |
| `UNKNOWN_11` | Allocates a skeletal animation instance (`skeletalAnim_new()`) | Some animated props |
| `UNKNOWN_24` / `UNKNOWN_25` | Spawn dust/impact effects when hit (non-lethal / lethal) | Most enemies |
| `UNKNOWN_8`, `UNKNOWN_5`, others | Various marker/actor bits, not fully understood | Copy from a similar actor |

## 4.8 Fighting: the hit table and collision callbacks

There are **two layers** that decide what happens when Banjo touches your actor.

### Layer 1: `collision_table` (`src/core2/code_B62B0.c`)

If your marker ID has a row here, the engine handles the fight for you: health,
hits to kill, damage to the player, knockback or bounce, honeycomb drops. Each
row has one 16-bit `CollisionParams` per attack type:

```c
typedef struct{
    u16 markerId;
    u16 collisionUnion;        // OR'ed into every column at init (often sets the drop count)
    u16 collisionTypes[11];    // columns, in this order:
    //  [0] claw swipe   [1] roll        [2] jump on top / fast fall  [3] rat-a-tat rap
    //  [4] beak barge   [5] beak buster [6] beak bomb   [7] egg      [8] wonderwing
    //  [9] croc bite    [10] other contact (just touching it)
} CollisionTable;
```

Each `CollisionParams` (`include/structs.h`) packs, from the high bits down:

```
 15..12  playerInteraction  1–5 = knock the player back (strength), 7–11 = bounce the player
 11..10  nextState          0 = nothing, 1 = "alternate" (collision2Func), 2 = die (dieFunc)
  9..7   unkBit7            on hit: spawn bundle BUNDLE_18 + n, a honeycomb knocked out at
                            distance variant n (halved for multi-hit enemies; ice cube variant
                            for actors spawned with ACTOR_FLAG_UNKNOWN_21)
  6..5   damageToPlayer     HP lost (3 = instant death)
  4..2   hitsToTrigger      each hit removes 100/n % of the enemy's health
  1..0   dropBundleNum      on death: spawn bundle BUNDLE_15 + n (1, 2 or 3 honeycombs)
```

Bundles are defined in `gBundle_array` in `src/core2/code_41FB0.c`. Applying
damage, health and drops is done in `func_8032B5C0()` (`src/core2/code_9E370.c`)
for the enemy and in `baMarker_8028D7B8()` (`src/core2/ba/marker.c`) for the
player.

The Grublin's row decoded:

| Column | Value | Meaning |
|---|---|---|
| union | `0x0001` | All columns get `dropBundleNum = 1` |
| claw | `0x0904` | die, 1 hit |
| roll | `0x0984` | die, 1 hit |
| jump on top | `0x8100` | bounce the player (8), no damage either way |
| rat-a-tat | `0x0984` | die, 1 hit |
| barge / buster | `0x0A04` | die, 1 hit |
| beak bomb | `0x0A84` | die, 1 hit |
| egg | `0x0984` | die, 1 hit |
| wonderwing | `0x0A04` | die, 1 hit |
| croc bite | `0x0000` | no interaction |
| touch | `0x1020` | knock the player back (1), 1 damage |

Bigbutt's claw entry is `0x0914`: `hitsToTrigger = 5`, so it takes five
swipes.

To add your enemy, **append a row and bump the size**:

```c
#define COLLISION_TABLE_SIZE        188   /* was 187 */
...
    {MARKER_29F_MY_ENEMY, 0x0001, {0x0908, 0x0988, 0x8100, 0x0988, 0x0A04, 0x0A04, 0x0A84, 0x0988, 0x0A04, 0x0000, 0x1040}},
    // 2 hits to kill with claw/roll/rap/eggs, 2 damage on touch
```

A Python one-liner to build values:

```python
def cp(interact=0, state=0, fx=0, dmg=0, hits=0, drop=0):
    return f"0x{(interact<<12)|(state<<10)|(fx<<7)|(dmg<<5)|(hits<<2)|drop:04X}"
cp(state=2, fx=2, hits=2)     # '0x0908'  die after 2 hits
cp(interact=1, dmg=2)          # '0x1040'  knockback + 2 damage
```

When the table decides the enemy got hit, it calls your marker callbacks with
the `nextState`:

```c
marker_setCollisionScripts(this->marker,
    __ow,     // MARKER_COLLISION_FUNC_0: hit but not killed / touched
    __alt,    // MARKER_COLLISION_FUNC_1: "alternate state" (e.g. flipped over Snippet)
    __die);   // MARKER_COLLISION_FUNC_2_DIE: health reached 0
```

`humanoidBaddie_update()` sets these itself from `local->hitFunction` and
`local->dieFunction`.

### Layer 2: special cases in `src/core2/ba/marker.c`

`__baMarker_resolveCollision()` runs for every actor the player touches. It
first calls `func_8033D410()`. **If your marker has a hit-table row, the table
handles the collision and returns early.** Otherwise it falls into a big
`switch(marker->id)` with hand-written behavior for collectibles, Mumbo
tokens, wading boots, Zubbas and more. Markers in neither place get no
automatic interaction, and both sides' `collisionFunc` is called with type 0.

So for:

- **an enemy**, add a hit-table row (layer 1);
- **a pickup or switch**, either handle it in your own `collisionFunc` (see
  `src/RBB/ch/anchorswitch.c`, which uses `marker_setCollisionScripts(m, NULL, press, NULL)`),
  or add a case to the `ba/marker.c` switch.

### Making your actor solid

Collision with the *world* for the player (standing on it, bumping into it)
comes from the **model's collision data** (see [chapter 6](06-models-textures-animation.md)).
Many vanilla objects get this for free from their model. Whether actor models
are tested against the player also depends on marker setup (`marker->unk3E_0`,
flags set by `actor_new`). Again, copy a similar vanilla object.

## 4.9 Placing the actor in a level

### Method 1: spawn from code (no setup editing)

The level overlay `init` callback runs right after the spawn list is rebuilt
(`gsworld_set()` calls `spawnQueue_reset()` then
`leveloverlay_initCallback_NotFP()`). Several are empty, so they make a good
hook. For Mumbo's Mountain, in `src/core2/code_9B990.c`:

```c
static void leveloverlay_MM_init(void) {
    f32 pos[3];
    if (gsworld_getMap() == MAP_2_MM_MUMBOS_MOUNTAIN) {
        pos[0] = 0.0f; pos[1] = 0.0f; pos[2] = 0.0f;   // pick coords with a debug position display
        actor_spawnWithYaw_f32(ACTOR_3CC_MY_ENEMY, pos, 90);
    }
}
```

These run on **every** load of a map belonging to that level, so check
`gsworld_getMap()`. Spawned actors aren't remembered across reloads unless you
gate them with a flag.

> In Freezeezy Peak the init callback runs via
> `leveloverlay_initCallback_OnlyFP()` *after* the player spawns.

### Method 2: put it in the setup file (how vanilla does it)

Actor placements are `NodeProp`s with **category 6**, inside the cube that
contains their position, in the map's setup asset (`map_id + 0x71C`). The
format is in [chapter 5](05-levels-and-maps.md#53-the-setup-file-format).

The quickest trick: **retarget existing placements.** List the actor nodes in
a setup and change, say, three Grublins in Mumbo's Mountain into your enemy:

```sh
python3 banjo-kazooie-modding/examples/bk_setup_tool.py dump assets/lvl_setup/071E.lvl_setup.bin
python3 banjo-kazooie-modding/examples/bk_setup_tool.py replace-actor \
        assets/lvl_setup/071E.lvl_setup.bin 0x6 0x3CC --limit 3
```

Adding brand-new placements means inserting nodes into a cube's list. Either
use the tool's writer as a starting point or a level editor such as Banjo's
Backpack (see [chapter 11](11-debugging-limits-references.md)).

Per-placement parameters you can set on a node:

| Node field | Arrives in the actor as |
|---|---|
| position (s16 x/y/z) | `position` |
| `yaw` (9 bits, degrees) | `yaw` |
| `scale` (23 bits, percent; 0 = 100%) | model scale |
| `unk10_31` (12 bits) | `secondaryId` |
| `selector_or_radius` (9 bits) | `actorTypeSpecificField` |

## 4.10 Rewards, persistence and "kill all" puzzles

- **Drops**: honeycombs come from `dropBundleNum` in the hit table. For other
  drops, spawn things in your die callback with the spawn queue (see
  `chBeehive_die` in `src/core2/ch/beehive.c`, which spawns steam and wood
  explosion actors).
- **Jiggy reward**: `jiggy_spawn(JIGGY_xx, position)`. Pick a jiggy ID from the
  level (`enum jiggy_e`). To *replace* a vanilla jiggy's source, remove or
  repurpose its original spawn so the jiggy can't be collected twice.
- **Stay dead**: use a file progress flag (saved) or level-specific flag
  (until you leave the level). On init, `if (flag) marker_despawn(this->marker);`.
- **Kill-N puzzles**: keep a counter in a level-specific flag group, or a
  static in your overlay (reset in the overlay `init`), and spawn the reward
  at N.

## 4.11 Debugging your enemy

| Symptom | Likely cause |
|---|---|
| Never appears | Not registered for this overlay; wrong actor ID in the node; update distance too small; spawned in an unloaded cube |
| Appears as a Bigbutt | Anti-tamper still on (see [1.4](01-setup-and-building.md#14-building-a-mod-turn-off-anti-tamper)) |
| Can't be hurt / player passes through | No hit-table row, or `marker->collidable` is off |
| Crash on contact | Marker ID ≥ `0x2A9` (see 4.5); NULL animation table with `subaddie_set_state` |
| Crash the first time it's drawn (in `marker_loadModelBin`) | Actor ID ≥ `0x3D5`, past the end of the model cache (see 4.5) |
| Crash when changing levels | Overlay `ActorInfo` registered in core2's global list (see 4.7) |
| T-pose / garbled animation | Animation's skeleton doesn't match the model |
| Works in emulator, crashes on console | Usually a heap overflow or an uninitialized `local` field. Test on ares. |
