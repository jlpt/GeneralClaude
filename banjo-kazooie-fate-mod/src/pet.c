#include "fate.h"
#include "recompconfig.h"

/*
 * Fate's pet, played by a Jinjo (or a Bigbutt calf). It is a real actor that
 * the mod registers in every map's spawn list and spawns next to Banjo. It
 * follows him, fights monsters, carries loot, can be sent to Grove to sell
 * that loot, and changes form when fed fish.
 */

#define PET_MARKER_ID  0x2A6        /* free marker id below the hit-table limit */
#define PET_ACTOR_BASE 0x3CC        /* free actor ids, one per form. Must stay below 0x3D5: the
                                     * engine's modelCache is indexed by actor id and has 0x3D5 slots. */

enum pet_anim_state {
    PET_ANIM_NONE,
    PET_ANIM_IDLE,
    PET_ANIM_MOVE,
    PET_ANIM_ATTACK,
    PET_ANIM_HURT,
};

enum pet_mode {
    PET_MODE_FOLLOW,
    PET_MODE_ATTACK,
    PET_MODE_FLEE,
    PET_MODE_AWAY,
};

static void fate_pet_update(Actor *this);

static ActorAnimationInfo sJinjoAnims[] = {
    { 0, 0.0f },
    { ASSET_2D_ANIM_JINJO_IDLE, 1.5f },
    { ASSET_31_ANIM_JINJO_JUMP, 0.5f },
    { ASSET_2F_ANIM_JINJO_HELP, 0.5f },
    { ASSET_131_ANIM_JINJO_FLY_END, 0.8f },
};

static ActorAnimationInfo sBigbuttAnims[] = {
    { 0, 0.0f },
    { ASSET_33_ANIM_BIGBUTT_EAT, 2.0f },
    { ASSET_10_ANIM_BIGBUTT_RUN, 0.5f },
    { ASSET_32_ANIM_BIGBUTT_ATTACK, 0.6f },
    { ASSET_253_ANIM_BIGBUTT_OW, 0.8f },
};

static ActorInfo sPetInfos[PET_FORM_COUNT] = {
    { PET_MARKER_ID, PET_ACTOR_BASE + 0, ASSET_3BB_MODEL_JINJO_YELLOW, PET_ANIM_IDLE, sJinjoAnims,   fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.5f, 0 },
    { PET_MARKER_ID, PET_ACTOR_BASE + 1, ASSET_3BC_MODEL_JINJO_ORANGE, PET_ANIM_IDLE, sJinjoAnims,   fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.5f, 0 },
    { PET_MARKER_ID, PET_ACTOR_BASE + 2, ASSET_3C0_MODEL_JINJO_BLUE,   PET_ANIM_IDLE, sJinjoAnims,   fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.5f, 0 },
    { PET_MARKER_ID, PET_ACTOR_BASE + 3, ASSET_3C2_MODEL_JINJO_GREEN,  PET_ANIM_IDLE, sJinjoAnims,   fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.5f, 0 },
    { PET_MARKER_ID, PET_ACTOR_BASE + 4, ASSET_3C1_MODEL_JINJO_PINK,   PET_ANIM_IDLE, sJinjoAnims,   fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.5f, 0 },
    { PET_MARKER_ID, PET_ACTOR_BASE + 5, ASSET_353_MODEL_BIGBUTT,      PET_ANIM_IDLE, sBigbuttAnims, fate_pet_update, actor_update_func_80326224, actor_draw, 0, 0, 0.8f, 0 },
};

static struct {
    ActorMarker *marker;
    u8 mode;
    u8 spawnedForm;
    u8 needSpawn;
    u8 animState;
    ActorMarker *target;
    f32 attackCooldown;
    f32 awayTimer;
    f32 bob;
    f32 formTick;
    f32 regenAccum;
} sPet;

f32 gPetFrenzyTimer;

/* ------------------------------------------------------------------ */
/* Stats                                                               */
/* ------------------------------------------------------------------ */

static s32 pet_equip_effect(s32 effect) {
    s32 total = 0;
    s32 i;
    for (i = 0; i < PET_EQUIP_COUNT; i++) {
        total += fate_item_effect_total(&gFate.petEquip[i], effect);
    }
    return total;
}

static const FatePetFormDef *pet_form(void) {
    u8 f = gFate.petForm < PET_FORM_COUNT ? gFate.petForm : PET_FORM_YELLOW_JINJO;
    return &gPetForms[f];
}

s32 fate_pet_max_hp(void) {
    s32 hp = 30 + 10 * gFate.petLevel + pet_equip_effect(FX_HP) + pet_equip_effect(FX_VIT) * 5;
    s32 pct = 100 + pet_form()->hpPct + gDerived.petHpPct + pet_equip_effect(FX_PET_HP) + gDerived.skill[SKILL_CHARM_MAGIC] * 8;
    return hp * pct / 100;
}

s32 fate_pet_damage(void) {
    s32 dmg = 4 + 2 * gFate.petLevel + pet_equip_effect(FX_DMG_FLAT) + pet_equip_effect(FX_STR) / 2;
    s32 pct = 100 + pet_form()->dmgPct + gDerived.petDmgPct + pet_equip_effect(FX_PET_DMG) + pet_equip_effect(FX_DMG_PCT)
              + gDerived.skill[SKILL_CHARM_MAGIC] * 8;
    return dmg * pct / 100;
}

static f32 pet_attack_interval(void) {
    f32 t = 1.4f * 100.0f / (f32)(100 + pet_form()->speedPct);
    if (gPetFrenzyTimer > 0.0f) {
        t /= 3.0f;
    }
    return t;
}

const char *fate_pet_name(void) {
    return pet_form()->name;
}

void fate_pet_gain_xp(s32 amount) {
    char buf[64];
    if (gFate.petSpecies == PET_SPECIES_NONE || amount <= 0) {
        return;
    }
    gFate.petXp += amount;
    while (gFate.petLevel < 60 && gFate.petXp >= (u32)(80 * gFate.petLevel * gFate.petLevel)) {
        gFate.petLevel++;
        gFate.petHp = (u16)fate_pet_max_hp();
        buf[0] = '\0';
        fate_strcat(buf, "Your pet grew to level ");
        fate_strcat_int(buf, gFate.petLevel);
        fate_strcat(buf, "!");
        fate_message(buf, MSG_COLOR_PET);
    }
}

void fate_pet_heal_full(void) {
    gFate.petHp = (u16)fate_pet_max_hp();
    if (sPet.mode == PET_MODE_FLEE) {
        sPet.mode = PET_MODE_FOLLOW;
    }
}

bool fate_pet_present(void) {
    return sPet.marker != NULL && sPet.mode != PET_MODE_AWAY;
}

f32 fate_pet_town_seconds(void) {
    return sPet.mode == PET_MODE_AWAY ? sPet.awayTimer : 0.0f;
}

const char *fate_pet_status(void) {
    if (gFate.petSpecies == PET_SPECIES_NONE) {
        return "No pet";
    }
    switch (sPet.mode) {
        case PET_MODE_ATTACK: return "Fighting";
        case PET_MODE_FLEE:   return "Wounded, fleeing";
        case PET_MODE_AWAY:   return "Selling loot in Grove";
        default:              return sPet.marker ? "Following" : "Catching up";
    }
}

/* ------------------------------------------------------------------ */
/* Actor management                                                    */
/* ------------------------------------------------------------------ */

void fate_pet_register_actors(void) {
    s32 i;
    for (i = 0; i < PET_FORM_COUNT; i++) {
        spawnableActorList_add(&sPetInfos[i], actor_new, ACTOR_FLAG_UNKNOWN_6 | ACTOR_FLAG_UNKNOWN_8);
    }
}

void fate_pet_on_map_load(void) {
    sPet.marker = NULL;
    sPet.target = NULL;
    sPet.needSpawn = TRUE;
    if (sPet.mode != PET_MODE_AWAY) {
        sPet.mode = PET_MODE_FOLLOW;
    }
}

static ActorMarker *find_existing_pet(void) {
    Actor *begin;
    Actor *a;
    ActorMarker *found = NULL;

    if (suBaddieActorArray == NULL) {
        return NULL;
    }
    begin = suBaddieActorArray->data;
    for (a = begin; a - begin < suBaddieActorArray->cnt; a++) {
        if (!a->despawn_flag && a->marker != NULL && a->marker->id == PET_MARKER_ID) {
            if (found == NULL) {
                found = a->marker;
            } else {
                marker_despawn(a->marker); /* duplicate restored from a map savestate */
            }
        }
    }
    return found;
}

static void spawn_pet(void) {
    f32 pos[3];
    s32 ipos[3];
    Actor *a;
    u8 form = gFate.petForm < PET_FORM_COUNT ? gFate.petForm : PET_FORM_YELLOW_JINJO;

    player_getPosition(pos);
    ipos[0] = (s32)(pos[0] - 80.0f);
    ipos[1] = (s32)(pos[1] + 40.0f);
    ipos[2] = (s32)(pos[2] - 80.0f);
    /* actor_spawnWithYaw_f32 has no return statement in vanilla, so call the s32 spawner it wraps. */
    FATE_TRACE("spawning pet form %d at %d %d %d\n", form, ipos[0], ipos[1], ipos[2]);
    a = __actor_spawnWithYaw_s32(PET_ACTOR_BASE + form, ipos, (s32)player_getYaw());
    FATE_TRACE("pet spawned: actor %p\n", a);
    if (a != NULL) {
        sPet.marker = a->marker;
        sPet.spawnedForm = form;
    }
}

static void despawn_pet(void) {
    if (sPet.marker != NULL) {
        marker_despawn(sPet.marker);
        sPet.marker = NULL;
    }
}

static void respawn_with_form(void) {
    despawn_pet();
    sPet.needSpawn = TRUE;
}

/* ------------------------------------------------------------------ */
/* Behaviour                                                           */
/* ------------------------------------------------------------------ */

static void set_anim(Actor *this, s32 state) {
    if (sPet.animState != state) {
        sPet.animState = (u8)state;
        subaddie_set_state(this, state);
        actor_loopAnimation(this);
    }
}

static void move_towards(Actor *this, f32 target[3], f32 speed, f32 dt) {
    f32 dx = target[0] - this->position[0];
    f32 dy = target[1] - this->position[1];
    f32 dz = target[2] - this->position[2];
    f32 dist = gu_sqrtf(dx * dx + dz * dz);
    f32 step = speed * dt;

    if (dist > 1.0f) {
        if (step > dist) {
            step = dist;
        }
        this->position[0] += dx / dist * step;
        this->position[2] += dz / dist * step;
        this->yaw_ideal = func_80257204(this->position[0], this->position[2], target[0], target[2]);
    }
    this->position[1] += dy * fate_clampf(dt * 6.0f, 0.0f, 1.0f);
    /* turn smoothly towards the ideal yaw */
    {
        f32 diff = this->yaw_ideal - this->yaw;
        while (diff > 180.0f) diff -= 360.0f;
        while (diff < -180.0f) diff += 360.0f;
        this->yaw += diff * fate_clampf(dt * 8.0f, 0.0f, 1.0f);
        if (this->yaw < 0.0f) this->yaw += 360.0f;
        if (this->yaw >= 360.0f) this->yaw -= 360.0f;
    }
}

static void pet_height(Actor *this, f32 target[3], f32 playerPos[3]) {
    if (gFate.petForm == PET_FORM_BIGBUTT) {
        f32 probe[3];
        f32 floorY;
        probe[0] = target[0];
        probe[1] = playerPos[1] + 150.0f;
        probe[2] = target[2];
        floorY = func_80309B24(probe);
        target[1] = (floorY > playerPos[1] - 300.0f && floorY < playerPos[1] + 200.0f) ? floorY : playerPos[1];
    } else {
        /* Jinjos hover at Banjo's shoulder with a gentle bob. */
        target[1] = playerPos[1] + 70.0f + ml_sin_deg(sPet.bob * 120.0f) * 12.0f;
    }
}

static void pet_attack(Actor *this, Actor *enemy) {
    s32 dmg = fate_pet_damage();
    s32 dealt = fate_damage_enemy(enemy, dmg, FALSE, FALSE, NULL);
    const FatePetFormDef *form = pet_form();

    set_anim(this, PET_ANIM_ATTACK);
    if (gFate.petForm == PET_FORM_BIGBUTT) {
        fate_play_sfx_pitch(SFX_3C_BULL_GROWN, 1.2f, 18000);
    } else {
        fate_play_sfx_pitch(SFX_27_JINJO_HI, 1.0f + fate_randf() * 0.3f, 16000);
    }
    if (dealt > 0 && form->healsOwner > 0) {
        gFateHp = fate_clampf(gFateHp + (f32)(form->healsOwner * (1 + gFate.petLevel / 5)), 0.0f, (f32)gDerived.maxHp);
        fate_health_sync_to_bk();
    }
    /* Monsters fight back. */
    if (dealt > 0 && fate_chance(4500)) {
        s32 hurt = fate_monster_damage_scale() * 6 / 10;
        if (hurt < 2) {
            hurt = 2;
        }
        gFate.petHp = (u16)fate_maxi(0, (s32)gFate.petHp - hurt);
        if (gFate.petHp == 0) {
            sPet.mode = PET_MODE_FLEE;
            fate_message("Your pet is badly hurt and runs to your side!", MSG_COLOR_BAD);
        }
    }
    sPet.attackCooldown = pet_attack_interval();
}

static void fate_pet_update(Actor *this) {
    f32 dt = time_getDelta();
    f32 playerPos[3];
    f32 target[3];
    f32 distToPlayer;
    f32 speed;

    if (!this->volatile_initialized) {
        FATE_TRACE("pet first update\n");
        this->volatile_initialized = TRUE;
        actor_collisionOff(this);
        this->scale = (gFate.petForm == PET_FORM_BIGBUTT) ? 0.45f : 0.8f;
        sPet.animState = PET_ANIM_NONE;
        set_anim(this, PET_ANIM_IDLE);
    }
    if (sPet.marker != this->marker) {
        /* Orphaned pet (left over from a savestate or a form change). */
        marker_despawn(this->marker);
        return;
    }
    if (!gFate.created || gFate.petSpecies == PET_SPECIES_NONE || sPet.mode == PET_MODE_AWAY) {
        return;
    }

    sPet.bob += dt;
    player_getPosition(playerPos);
    distToPlayer = fate_dist3d(this->position, playerPos);

    /* Too far behind (warped, fell): pop back next to Banjo. */
    if (distToPlayer > 2200.0f) {
        this->position[0] = playerPos[0] - 60.0f;
        this->position[1] = playerPos[1] + 60.0f;
        this->position[2] = playerPos[2] - 60.0f;
        return;
    }

    if (sPet.attackCooldown > 0.0f) {
        sPet.attackCooldown -= dt;
    }

    if (sPet.mode != PET_MODE_FLEE && !fate_in_town() && gFate.petHp > 0) {
        Actor *enemy = sPet.target ? marker_getActor(sPet.target) : NULL;
        if (enemy == NULL || enemy->despawn_flag || !enemy->marker->collidable
            || fate_dist3d(enemy->position, playerPos) > 1100.0f) {
            enemy = fate_find_nearest_enemy(playerPos, 900.0f);
            sPet.target = enemy ? enemy->marker : NULL;
        }
        sPet.mode = enemy ? PET_MODE_ATTACK : PET_MODE_FOLLOW;
        if (enemy) {
            f32 d = fate_dist2d(this->position, enemy->position);
            target[0] = enemy->position[0];
            target[1] = enemy->position[1] + (gFate.petForm == PET_FORM_BIGBUTT ? 0.0f : 60.0f);
            target[2] = enemy->position[2];
            if (d < 130.0f) {
                if (sPet.attackCooldown <= 0.0f) {
                    pet_attack(this, enemy);
                }
            } else {
                move_towards(this, target, 750.0f, dt);
                set_anim(this, PET_ANIM_MOVE);
            }
            return;
        }
    }

    /* Follow: trail behind Banjo's left shoulder. */
    {
        f32 yaw = player_getYaw();
        target[0] = playerPos[0] + ml_sin_deg(yaw + 210.0f) * 140.0f;
        target[2] = playerPos[2] + ml_cos_deg(yaw + 210.0f) * 140.0f;
        pet_height(this, target, playerPos);
    }
    speed = distToPlayer > 500.0f ? 1100.0f : 520.0f;
    if (fate_dist2d(this->position, target) > 40.0f) {
        move_towards(this, target, speed, dt);
        set_anim(this, sPet.mode == PET_MODE_FLEE ? PET_ANIM_HURT : PET_ANIM_MOVE);
    } else {
        this->position[1] += (target[1] - this->position[1]) * fate_clampf(dt * 4.0f, 0.0f, 1.0f);
        this->yaw_ideal = player_getYaw();
        move_towards(this, this->position, 0.0f, dt);
        set_anim(this, sPet.mode == PET_MODE_FLEE ? PET_ANIM_HURT : PET_ANIM_IDLE);
    }
}

/* ------------------------------------------------------------------ */
/* Frame upkeep, selling, feeding                                      */
/* ------------------------------------------------------------------ */

static void return_from_town(void) {
    s32 gold = 0;
    s32 sold = 0;
    s32 i;
    char msg[96];

    for (i = 0; i < PET_BAG_SIZE; i++) {
        FateItem *it = &gFate.petBag[i];
        if (fate_item_empty(it) || it->rarity == RARITY_ARTIFACT) {
            continue; /* Fate's pets won't sell quest artifacts */
        }
        gold += fate_item_sell_price(it);
        fate_item_clear(it);
        sold++;
    }
    fate_add_gold(gold);
    fate_pet_heal_full();
    sPet.mode = PET_MODE_FOLLOW;
    sPet.needSpawn = TRUE;
    msg[0] = '\0';
    fate_strcat(msg, "Your pet is back from Grove! Sold ");
    fate_strcat_int(msg, sold);
    fate_strcat(msg, " items for ");
    fate_strcat_int(msg, gold);
    fate_strcat(msg, " gold.");
    fate_message(msg, MSG_COLOR_PET);
    fate_play_sfx(SFX_17_JINJO_WHISTLE);
}

bool fate_pet_send_to_town(void) {
    if (gFate.petSpecies == PET_SPECIES_NONE || sPet.mode == PET_MODE_AWAY) {
        return FALSE;
    }
    if (fate_in_town()) {
        /* Already in Grove: sell instantly. */
        return_from_town();
        return TRUE;
    }
    sPet.mode = PET_MODE_AWAY;
    sPet.awayTimer = 30.0f + 4.0f * (f32)fate_depth();
    despawn_pet();
    fate_message("Your pet scurries off to Grove to sell your loot.", MSG_COLOR_PET);
    return TRUE;
}

bool fate_pet_feed_fish(s32 fishType, s32 size) {
    const FateFishDef *def;
    s32 idx = fishType * FISH_SIZE_COUNT + size;
    char msg[96];
    static const u16 durations[FISH_SIZE_COUNT] = { 120, 300, 600, 0 };

    if (fishType < 0 || fishType >= FISH_COUNT || size < 0 || size >= FISH_SIZE_COUNT || gFate.fish[idx] == 0) {
        return FALSE;
    }
    def = &gFishDefs[fishType];
    if (def->petForm == PET_FORM_NONE) {
        fate_message("Your pet sniffs it and refuses. It's a boot.", MSG_COLOR_DEFAULT);
        return FALSE;
    }
    gFate.fish[idx]--;
    if (def->petForm == PET_FORM_REVERT) {
        gFate.petFormPermanent = (gFate.petSpecies == PET_SPECIES_BIGBUTT) ? PET_FORM_BIGBUTT : PET_FORM_YELLOW_JINJO;
        gFate.petForm = gFate.petFormPermanent;
        gFate.petFormSeconds = 0;
        fate_message("The Dogfish returns your pet to its true form.", MSG_COLOR_PET);
    } else {
        gFate.petForm = def->petForm;
        if (size == FISH_SIZE_FLAWLESS) {
            gFate.petFormPermanent = def->petForm;
            gFate.petFormSeconds = 0;
        } else {
            gFate.petFormSeconds = durations[size];
        }
        msg[0] = '\0';
        fate_strcat(msg, "Your pet gobbles the fish and becomes a ");
        fate_strcat(msg, gPetForms[def->petForm].name);
        fate_strcat(msg, size == FISH_SIZE_FLAWLESS ? " for good!" : "!");
        fate_message(msg, MSG_COLOR_PET);
    }
    fate_play_jingle(COMUSIC_1D_MUMBO_TRANSFORMATION);
    gFate.petHp = (u16)fate_pet_max_hp();
    respawn_with_form();
    return TRUE;
}

void fate_pet_frame(f32 dt) {
    s32 maxHp;

    if (gFate.petSpecies == PET_SPECIES_NONE) {
        if (sPet.marker != NULL) {
            despawn_pet();
        }
        return;
    }
    if (gPetFrenzyTimer > 0.0f) {
        gPetFrenzyTimer -= dt;
    }

    maxHp = fate_pet_max_hp();
    if (gFate.petHp > maxHp) {
        gFate.petHp = (u16)maxHp;
    }

    if (sPet.mode == PET_MODE_AWAY) {
        sPet.awayTimer -= dt;
        if (sPet.awayTimer <= 0.0f) {
            return_from_town();
        }
        return;
    }

    /* Regenerate: slowly in the field, quickly while fleeing or in Grove. */
    sPet.regenAccum += dt * (f32)maxHp * (fate_in_town() ? 0.2f : (sPet.mode == PET_MODE_FLEE ? 0.04f : 0.01f));
    if (gPetFrenzyTimer > 0.0f) {
        sPet.regenAccum += dt * (f32)maxHp * 0.05f;
    }
    if (sPet.regenAccum >= 1.0f) {
        s32 h = (s32)sPet.regenAccum;
        sPet.regenAccum -= (f32)h;
        gFate.petHp = (u16)fate_mini(maxHp, gFate.petHp + h);
    }
    if (sPet.mode == PET_MODE_FLEE && gFate.petHp >= maxHp / 2) {
        sPet.mode = PET_MODE_FOLLOW;
        fate_message("Your pet is ready to fight again.", MSG_COLOR_PET);
    }

    /* Temporary fish forms wear off. */
    if (gFate.petFormSeconds > 0) {
        sPet.formTick += dt;
        while (sPet.formTick >= 1.0f && gFate.petFormSeconds > 0) {
            sPet.formTick -= 1.0f;
            gFate.petFormSeconds--;
        }
        if (gFate.petFormSeconds == 0) {
            gFate.petForm = gFate.petFormPermanent;
            fate_message("Your pet's transformation wears off.", MSG_COLOR_PET);
            respawn_with_form();
        }
    }

    if (sPet.needSpawn && fate_gameplay_active() && !player_isDead()) {
        ActorMarker *existing = find_existing_pet();
        sPet.needSpawn = FALSE;
        if (existing != NULL && marker_getActor(existing)->modelCacheIndex == PET_ACTOR_BASE + gFate.petForm) {
            sPet.marker = existing;
        } else {
            if (existing != NULL) {
                marker_despawn(existing);
            }
            spawn_pet();
        }
    }
}
