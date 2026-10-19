#include "fate.h"
#include "bkrecomp_api.h"
#include "recompconfig.h"

#define ENEMY_INIT_MAGIC 0xA5
#define ROSTER_NONE 0xFF
#define ROSTER_VANILLA_HP 0x04

static MarkerExtensionId sEnemyExt;
static u8 sRosterLookup[0x400]; /* marker id -> roster index + 1 */

/* Snapshot of the last monster Banjo hit, shown in the HUD target bar. */
Actor *gFateTargetActor;
f32 gFateTargetTimer;
s32 gFateLastDamage;
bool gFateLastCrit;
char gFateTargetName[64];
s32 gFateTargetHp;
s32 gFateTargetMaxHp;
u8 gFateTargetChampion;

static bool sQuestChampionSpawned;

/* The engine keeps three per-collision-type health bytes at unk164 (the third
 * one is declared as unk166), indexed by enum marker_collision_func_type_e. */
static u8 *health_bytes(Actor *actor) {
    return (u8 *)actor + __builtin_offsetof(Actor, unk164);
}

/* ------------------------------------------------------------------ */
/* Setup                                                               */
/* ------------------------------------------------------------------ */

void fate_combat_init(void) {
    s32 i;
    sEnemyExt = bkrecomp_extend_marker_all(sizeof(FateEnemy));
    bzero(sRosterLookup, sizeof(sRosterLookup));
    for (i = 0; i < gRosterCount; i++) {
        if (gRoster[i].marker < 0x400) {
            sRosterLookup[gRoster[i].marker] = (u8)(i + 1);
        }
    }
}

void fate_combat_on_map_load(void) {
    sQuestChampionSpawned = FALSE;
    gFateTargetTimer = 0.0f;
}

const FateRosterEntry *fate_roster_find(s32 markerId) {
    if (markerId < 0 || markerId >= 0x400 || sRosterLookup[markerId] == 0) {
        return NULL;
    }
    return &gRoster[sRosterLookup[markerId] - 1];
}

static s32 difficulty_pct(void) {
    switch (recomp_get_config_u32("difficulty")) {
        case 0:  return 70;   /* Casual */
        case 2:  return 140;  /* Hardcore */
        default: return 100;  /* Normal */
    }
}

static s32 hit_table_max_hits(s32 markerId) {
    s32 row;
    s32 best = 1;
    s32 i;
    if (markerId < 0 || markerId >= 0x2A9) {
        return 1;
    }
    row = D_80383F10[markerId];
    if (row < 0) {
        return 1;
    }
    for (i = 0; i < 8; i++) {
        u16 c = D_80370AC0[row].collisionTypes[i];
        if (((c >> 10) & 3) == 2) {
            s32 hits = (c >> 2) & 7;
            if (hits > best) {
                best = hits;
            }
        }
    }
    return best;
}

static void setup_enemy(Actor *actor, FateEnemy *e) {
    const FateRosterEntry *r = fate_roster_find(actor->marker->id);
    s32 depth = fate_depth();
    s32 hp;
    s32 champChance;

    bzero(e, sizeof(*e));
    e->init = ENEMY_INIT_MAGIC;
    e->rosterIdx = ROSTER_NONE;
    if (r == NULL) {
        return;
    }
    e->rosterIdx = (u8)(r - gRoster);
    e->nameSeed = (u16)fate_rand();

    /* Town foes, breakables and minigame foes keep vanilla hits-to-kill. */
    if (depth == 0 || (r->flags & (ROSTERF_TOWN | ROSTERF_BREAKABLE | ROSTER_VANILLA_HP))) {
        e->maxHp = 0;
        e->hp = 0;
        return;
    }

    hp = (10 + 6 * r->tier + 4 * (hit_table_max_hits(actor->marker->id) - 1));
    hp = hp * (100 + 45 * (depth - 1)) / 100;
    hp = hp * difficulty_pct() / 100;

    /* Champion monsters: Fate's glowing, named elites. */
    champChance = 600 + depth * 50;
    if (champChance > 1500) {
        champChance = 1500;
    }
    if (!sQuestChampionSpawned && fate_quest_champion_for(level_get(), actor->marker->id) >= 0) {
        e->questChampion = (u8)(fate_quest_champion_for(level_get(), actor->marker->id) + 1);
        e->champion = CHAMP_TOUGH;
        e->nameSeed = gFate.quests[e->questChampion - 1].seed;
        sQuestChampionSpawned = TRUE;
    } else if (recomp_get_config_u32("champions") == 0 && fate_chance(champChance)) {
        e->champion = (u8)fate_rand_range(1, CHAMP_COUNT - 1);
    }
    if (e->champion != CHAMP_NONE) {
        hp = hp * (e->champion == CHAMP_TOUGH ? 500 : 350) / 100;
        actor->scale *= 1.4f;
    }
    e->maxHp = hp < 1 ? 1 : hp;
    e->hp = e->maxHp;
}

static FateEnemy *enemy_data(ActorMarker *marker) {
    if (marker == NULL || sEnemyExt == 0) {
        return NULL;
    }
    return (FateEnemy *)bkrecomp_get_extended_marker_data(marker, sEnemyExt);
}

FateEnemy *fate_enemy_get(ActorMarker *marker) {
    FateEnemy *e = enemy_data(marker);
    Actor *actor;

    if (e == NULL) {
        return NULL;
    }
    if (e->init != ENEMY_INIT_MAGIC) {
        if (fate_roster_find(marker->id) == NULL) {
            /* Not a monster. Cheap early-out without touching the actor. */
            bzero(e, sizeof(*e));
            e->init = ENEMY_INIT_MAGIC;
            e->rosterIdx = ROSTER_NONE;
            return NULL;
        }
        actor = marker_getActor(marker);
        if (actor == NULL) {
            return NULL;
        }
        setup_enemy(actor, e);
    }
    if (e->rosterIdx == ROSTER_NONE) {
        return NULL;
    }
    return e;
}

void fate_champion_name(u16 seed, s32 markerId, char *out, s32 cap) {
    const FateRosterEntry *r = fate_roster_find(markerId);
    out[0] = '\0';
    fate_strcat(out, gChampionPrefixes[seed % gChampionPrefixCount]);
    fate_strcat(out, " ");
    fate_strcat(out, r ? r->name : "Beast");
    fate_strcat(out, " ");
    fate_strcat(out, gChampionTitles[(seed >> 4) % gChampionTitleCount]);
}

void fate_enemy_name(ActorMarker *marker, char *out, s32 cap) {
    FateEnemy *e = fate_enemy_get(marker);
    out[0] = '\0';
    if (e == NULL) {
        fate_strcpy(out, "Monster");
        return;
    }
    if (e->champion != CHAMP_NONE) {
        fate_champion_name(e->nameSeed, marker->id, out, cap);
        return;
    }
    fate_strcpy(out, gRoster[e->rosterIdx].name);
}

/* ------------------------------------------------------------------ */
/* Damage                                                              */
/* ------------------------------------------------------------------ */

enum attack_source {
    SRC_PAW,
    SRC_BEAK,
    SRC_EGG,
    SRC_ANY,
};

typedef struct {
    u8 source;
    u8 skill;
    u16 mult;
} AttackInfo;

/* Indexed by hit-table column. */
static const AttackInfo sAttackInfo[11] = {
    { SRC_PAW,  SKILL_CLAW,   100 },  /* claw swipe */
    { SRC_PAW,  SKILL_ROLL,   120 },  /* forward roll */
    { SRC_PAW,  SKILL_BUSTER,  70 },  /* jump on top / fast fall */
    { SRC_BEAK, SKILL_PECK,    60 },  /* rat-a-tat rap */
    { SRC_BEAK, SKILL_BARGE,  130 },  /* beak barge */
    { SRC_BEAK, SKILL_BUSTER, 160 },  /* beak buster */
    { SRC_BEAK, SKILL_BOMB,   250 },  /* beak bomb */
    { SRC_EGG,  SKILL_EGG,     80 },  /* eggs */
    { SRC_ANY,  SKILL_COUNT,  300 },  /* wonderwing */
    { SRC_PAW,  SKILL_CLAW,   150 },  /* croc bite */
    { SRC_ANY,  SKILL_COUNT,   50 },  /* touch */
};

static void snapshot_target(Actor *actor, FateEnemy *e, s32 dmg, bool crit) {
    gFateTargetActor = actor;
    gFateTargetTimer = 4.0f;
    gFateLastDamage = dmg;
    gFateLastCrit = crit;
    fate_enemy_name(actor->marker, gFateTargetName, sizeof(gFateTargetName));
    gFateTargetHp = e->hp;
    gFateTargetMaxHp = e->maxHp;
    gFateTargetChampion = e->champion;
    fate_hud_mark_dirty();
}

static s32 apply_crit_and_mods(s32 dmg, FateEnemy *e, bool canCrit, bool *crit) {
    *crit = FALSE;
    if (canCrit && fate_chance(gDerived.critChance)) {
        dmg = dmg * gDerived.critMult / 100;
        *crit = TRUE;
    }
    if (e->champion == CHAMP_ARMORED) {
        dmg = dmg * 60 / 100;
    }
    if (e->fearTimer > 0.0f) {
        dmg = dmg * 125 / 100;
    }
    return dmg < 1 ? 1 : dmg;
}

static s32 player_hit_damage(FateEnemy *e, s32 col, bool *crit) {
    const AttackInfo *a = &sAttackInfo[(col >= 0 && col < 11) ? col : 10];
    s32 lo, hi, base, pct, dmg;

    switch (a->source) {
        case SRC_PAW:
            lo = gDerived.pawMin; hi = gDerived.pawMax;
            base = fate_rand_range(lo, hi) + gDerived.stat[STAT_STR] / 4;
            break;
        case SRC_EGG:
            lo = gDerived.beakMin; hi = gDerived.beakMax;
            base = fate_rand_range(lo, hi) + gDerived.stat[STAT_DEX] / 4;
            break;
        case SRC_BEAK:
            lo = gDerived.beakMin; hi = gDerived.beakMax;
            base = fate_rand_range(lo, hi) + gDerived.stat[STAT_STR] / 6 + gDerived.stat[STAT_DEX] / 6;
            break;
        default:
            lo = fate_maxi(gDerived.pawMin, gDerived.beakMin);
            hi = fate_maxi(gDerived.pawMax, gDerived.beakMax);
            base = fate_rand_range(lo, hi) + gDerived.stat[STAT_STR] / 4;
            break;
    }
    base += gDerived.dmgFlat;

    pct = 100 + gDerived.dmgPct;
    if (a->skill < SKILL_COUNT) {
        pct += gDerived.skill[a->skill] * 6;
    }
    if (a->source == SRC_PAW) pct += gDerived.pawPct;
    if (a->source == SRC_BEAK) pct += gDerived.beakPct;
    if (a->source == SRC_EGG) pct += gDerived.eggPct;

    dmg = base * a->mult / 100;
    dmg = dmg * pct / 100 + gDerived.fireDmg;
    return apply_crit_and_mods(dmg, e, TRUE, crit);
}

static s32 hit_column(ActorMarker *enemy, struct5Cs *params) {
    s32 row;
    s32 col;
    if (enemy->id >= 0x2A9) {
        return 10;
    }
    row = D_80383F10[enemy->id];
    if (row < 0) {
        return 10;
    }
    col = (s32)((u16 *)params - D_80370AC0[row].collisionTypes);
    return (col >= 0 && col < 11) ? col : 10;
}

static void life_steal(s32 dmg) {
    if (gDerived.lifeSteal > 0 && dmg > 0) {
        gFateHp = fate_clampf(gFateHp + (f32)(dmg * gDerived.lifeSteal) / 100.0f, 0.0f, (f32)gDerived.maxHp);
        fate_health_sync_to_bk();
    }
}

/* Damage from the pet or a spell. Returns damage dealt; kills the monster the vanilla way. */
s32 fate_damage_enemy(Actor *actor, s32 damage, bool fromPlayer, bool canCrit, s32 *outCrit) {
    FateEnemy *e;
    bool crit;

    if (actor == NULL || actor->despawn_flag || actor->marker == NULL || !actor->marker->collidable) {
        return 0;
    }
    e = fate_enemy_get(actor->marker);
    if (e == NULL || e->maxHp <= 0 || e->killed) {
        return 0;
    }
    damage = apply_crit_and_mods(damage, e, canCrit, &crit);
    if (outCrit) {
        *outCrit = crit;
    }
    e->hp -= damage;
    if (fromPlayer) {
        life_steal(damage);
    }
    if (e->hp <= 0) {
        e->hp = 0;
        health_bytes(actor)[MARKER_COLLISION_FUNC_2_DIE] = 0;
        snapshot_target(actor, e, damage, crit);
        fate_on_enemy_killed(actor, e);
        marker_callCollisionFunc(actor->marker, baMarker_get(), MARKER_COLLISION_FUNC_2_DIE);
    } else {
        health_bytes(actor)[MARKER_COLLISION_FUNC_2_DIE] = (u8)fate_clampi(e->hp * 99 / e->maxHp, 1, 99);
        snapshot_target(actor, e, damage, crit);
    }
    return damage;
}

/*
 * @fate Replaces the engine's enemy hit handler (code_9E370.c). The vanilla
 * logic is kept intact, except that hits which would normally take a
 * fraction of a monster's "health percentage" now subtract real damage from
 * a Fate-style hit point pool, so monsters scale with depth and gear.
 */
RECOMP_PATCH void func_8032B5C0(ActorMarker *arg0, ActorMarker *arg1, struct5Cs *arg2) {
    Actor *this;
    s32 sp70;
    s32 sp6C;
    s32 sp68;
    enum marker_collision_func_type_e sp64;
    s32 var_v0;
    f32 player_yaw;
    f32 sp50[3];
    s32 sp4C;
    NodeProp *sp48;
    s32 sp3C[3];
    f32 sp38;
    u8 *health;
    FateEnemy *fe;
    s32 col;
    bool useHp;

    this = marker_getActor(arg0);
    health = health_bytes(this);
    sp70 = func_8033D5B4(arg2);
    sp6C = func_8033D584(arg2);
    sp68 = func_8033D5A4(arg2);
    sp64 = func_8033D574(arg2);
    if (((baiFrame_getState() != 3) && func_8028F1E0()) || (func_8033D594(arg2) == 0)) {
        if (sp64 == 0) {
            if ((sp68 != 0) || (arg1->id == 0)) {
                if (sp68 <= 0) {
                    sp68 = 1;
                }
            } else {
                return;
            }
        }

        fe = fate_enemy_get(arg0);
        col = hit_column(arg0, arg2);
        /* Wonderwing and self-destructing touches stay instant kills. */
        useHp = fe != NULL && fe->maxHp > 0 && !fe->killed && sp64 == MARKER_COLLISION_FUNC_2_DIE
                && sp68 != 0 && col != 8 && col != 10 && fate_gameplay_active();

        if (sp68 != 0) {
            if (useHp) {
                bool crit;
                s32 dmg = player_hit_damage(fe, col, &crit);
                fe->hp -= dmg;
                life_steal(dmg);
                if (fe->hp <= 0) {
                    fe->hp = 0;
                    health[sp64] = 0;
                } else {
                    health[sp64] = (u8)fate_clampi(fe->hp * 99 / fe->maxHp, 1, 99);
                    sp6C = 0; /* honeycombs only fly out on the killing blow */
                }
                snapshot_target(this, fe, dmg, crit);
                if (crit) {
                    fate_play_sfx(SFX_1F_HITTING_AN_ENEMY_3);
                }
            } else {
                var_v0 = MAX(0, (health[sp64] - (100 / sp68)));
                if ((health[sp64] = var_v0) && (sp68 >= 2)) {
                    sp6C /= 2;
                }
            }
        }
        if (sp6C != 0) {
            bundle_setYaw(func_80257204(arg0->propPtr->x, arg0->propPtr->z, arg1->propPtr->x, arg1->propPtr->z) + 90.0f);
            D_8036E564 = sp6C;
            if (this->unk138_25) {
                __bundle_spawnFromFirstActor(sp6C + BUNDLE_21__ICECUBE_B, this);
            } else {
                if ((this->marker->id < 0x1A1) || (this->marker->id >= 0x1A5)) {
                    __bundle_spawnFromFirstActor(sp6C + BUNDLE_18__HONEYCOMB, this);
                }
            }
        }
        if (sp68 != 0) {
            if ((sp64 == 2) && this->unk16C_1) {
                func_8032B3A0(this, arg1);
            }
            if ((sp64 == 1) && this->unk16C_2) {
                func_8032B3A0(this, arg1);
            }

            if ((health[sp64] == 0) || ((sp64 != 0) && this->unk138_9)) {
                if (health[sp64] != 0) {
                    sp64 -= 1;
                }
                if (sp64 != 2) {
                    health[sp64] = 0x63;
                }
                if ((sp64 == 2) && (sp70 != 0)) {
                    player_yaw = player_getYaw();
                    sp3C[0] = (s32) this->position[0];
                    sp3C[1] = (s32) this->position[1];
                    sp3C[2] = (s32) this->position[2];
                    if ((s32)arg0->unk44 < 0) {
                        func_8034A174(arg0->unk44, 0x20, sp50);
                    }
                    func_8032EE0C(func_8032B38C, (s32)this);
                    if (((s32)arg0->unk44 < 0) && ((sp50[0] != 0.0f) || (sp50[1] != 0.0f) || (sp50[2] != 0.0f))) {
                        __spawnQueue_add_5((void (*)(void))spawnQueue_bundleWithYaw_f32, sp70 + BUNDLE_15__JIGGY, reinterpret_cast(s32, sp50[0]), reinterpret_cast(s32, sp50[1]), reinterpret_cast(s32, sp50[2]), reinterpret_cast(s32, player_yaw));
                    } else if (this->unk16C_3 && func_803048E0(sp3C, &sp4C, (s32)&sp48, 3, (s32) (func_8033229C(arg0) * 4.0f))) {
                        sp50[0] = (f32) sp48->x;
                        sp50[1] = (f32) sp48->y;
                        sp50[2] = (f32) sp48->z;
                        __spawnQueue_add_5((void (*)(void))spawnQueue_bundleWithYaw_f32, sp70 + BUNDLE_15__JIGGY, reinterpret_cast(s32, sp50[0]), reinterpret_cast(s32, sp50[1]), reinterpret_cast(s32, sp50[2]), reinterpret_cast(s32, player_yaw));
                    } else {
                        sp38 = this->position[1] + func_8033229C(arg0);
                        __spawnQueue_add_5((void (*)(void))spawnQueue_bundleWithYaw_f32, sp70 + BUNDLE_15__JIGGY, reinterpret_cast(s32, this->position[0]), reinterpret_cast(s32, sp38), reinterpret_cast(s32, this->position[2]), reinterpret_cast(s32, player_yaw));
                    }
                    func_8032EE20();
                }
                if (sp64 == 2 && fe != NULL) {
                    fate_on_enemy_killed(this, fe);
                }
                marker_callCollisionFunc(arg0, arg1, sp64);
            }
            if ((sp64 != 0) && (sp6C != 0)) {
                sfx_playFadeShorthandDefault(SFX_1D_HITTING_AN_ENEMY_1, 1.0f, 25984, this->position, (s32)((500.0f + func_8033229C(arg0)) * 0.5), (s32)((500.0f + func_8033229C(arg0)) * 5));
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Kills and loot                                                      */
/* ------------------------------------------------------------------ */

static void loot_item(FateItem *item) {
    char name[64];
    char msg[96];

    fate_item_name(item, name, sizeof(name));
    msg[0] = '\0';
    if (fate_inventory_add(item)) {
        fate_strcat(msg, "Found: ");
    } else if (fate_pet_present() && fate_pet_bag_add(item)) {
        fate_strcat(msg, "Your pet carries: ");
    } else {
        fate_strcat(msg, "Pack full! Left behind: ");
    }
    fate_strcat(msg, name);
    fate_message(msg, fate_rarity_color(item->rarity));
    if (item->rarity >= RARITY_UNIQUE && item->rarity != RARITY_ARTIFACT) {
        fate_play_jingle(COMUSIC_3D_JIGGY_SPAWN);
    }
}

void fate_drop_loot(f32 pos[3], s32 depth, bool champion) {
    FateItem item;
    s32 itemChance = 1000 + depth * 30 + gDerived.magicFind * 10;
    s32 rolls = champion ? 2 : 1;
    s32 i;

    for (i = 0; i < rolls; i++) {
        if ((champion && i == 0) || fate_chance(itemChance)) {
            fate_item_generate(&item, depth, -1, gDerived.magicFind + (champion ? 100 : 0));
            loot_item(&item);
        }
    }
    if (fate_chance(800)) {
        s32 cons;
        if (fate_chance(7000)) {
            cons = (depth >= 5 && fate_chance(4000)) ? CONS_HP_POTION_GREATER : CONS_HP_POTION;
        } else {
            cons = (depth >= 5 && fate_chance(4000)) ? CONS_MANA_POTION_GREATER : CONS_MANA_POTION;
        }
        gFate.consumables[cons]++;
        {
            char msg[64];
            msg[0] = '\0';
            fate_strcat(msg, "Found: ");
            fate_strcat(msg, gConsumableNames[cons]);
            fate_message(msg, MSG_COLOR_DEFAULT);
        }
    }
    if (fate_chance(200)) {
        s32 cons = fate_chance(5000) ? CONS_TOWN_PORTAL : CONS_IDENTIFY;
        char msg[64];
        gFate.consumables[cons]++;
        msg[0] = '\0';
        fate_strcat(msg, "Found: ");
        fate_strcat(msg, gConsumableNames[cons]);
        fate_message(msg, MSG_COLOR_DEFAULT);
    }
    if (fate_chance(champion ? 1500 : 200)) {
        /* Better gems deeper down: chipped early, full gems from depth 5. */
        s32 gem = fate_rand_range(1, depth >= 5 ? gGemCount - 1 : 8);
        char msg[64];
        if (gem < 16) {
            gFate.gems[gem]++;
        }
        msg[0] = '\0';
        fate_strcat(msg, "Found a gem: ");
        fate_strcat(msg, gGems[gem].name);
        fate_message(msg, 0x7FFFD4FF);
    }
}

void fate_on_enemy_killed(Actor *actor, FateEnemy *enemy) {
    const FateRosterEntry *r;
    s32 depth = fate_depth();
    s32 xp, gold;
    char msg[128];
    char name[64];
    bool champ;

    if (enemy == NULL || enemy->killed || enemy->rosterIdx == ROSTER_NONE || !gFate.created) {
        return;
    }
    enemy->killed = TRUE;
    r = &gRoster[enemy->rosterIdx];
    champ = enemy->champion != CHAMP_NONE;

    if (r->flags & ROSTERF_BREAKABLE) {
        gold = fate_rand_range(1, 4) * fate_maxi(1, depth);
        fate_add_gold(gold);
        if (fate_chance(1500)) {
            gFate.consumables[CONS_HP_POTION]++;
            fate_message("Found: Health Potion", MSG_COLOR_DEFAULT);
        }
        return;
    }

    if (depth == 0 || (r->flags & ROSTERF_TOWN)) {
        xp = 3;
        gold = 1;
    } else {
        xp = (8 + 6 * r->tier) * (100 + 60 * (depth - 1)) / 100;
        gold = fate_rand_range(2, 6) * depth;
        if (champ) {
            xp *= 5;
            gold *= 5;
        }
    }
    gold = gold + gold * gDerived.goldFind / 100;
    gFate.kills++;

    fate_enemy_name(actor->marker, name, sizeof(name));
    msg[0] = '\0';
    fate_strcat(msg, name);
    fate_strcat(msg, " slain. +");
    fate_strcat_int(msg, gold);
    fate_strcat(msg, " gold");
    fate_message(msg, champ ? MSG_COLOR_GOLD : MSG_COLOR_DEFAULT);

    fate_add_gold(gold);
    fate_gain_xp(xp, champ);

    if (champ) {
        gFate.championKills++;
        fate_banner("CHAMPION SLAIN!", MSG_COLOR_GOLD);
        fate_gain_fame(8 + 3 * depth);
    }
    if (depth > 0 && !(r->flags & ROSTERF_TOWN)) {
        fate_drop_loot(actor->position, depth, champ);
    }
    fate_quest_on_kill(actor, enemy);
}

/* ------------------------------------------------------------------ */
/* Queries and per-frame upkeep                                        */
/* ------------------------------------------------------------------ */

static bool is_live_monster(Actor *a, FateEnemy **outE) {
    FateEnemy *e;
    if (a->despawn_flag || a->marker == NULL || !a->marker->collidable) {
        return FALSE;
    }
    if (fate_roster_find(a->marker->id) == NULL) {
        return FALSE;
    }
    e = fate_enemy_get(a->marker);
    if (e == NULL || e->maxHp <= 0 || e->killed) {
        return FALSE;
    }
    if (outE) {
        *outE = e;
    }
    return TRUE;
}

void fate_find_enemies_near(f32 pos[3], f32 radius, Actor **out, s32 *count, s32 max) {
    Actor *begin;
    Actor *a;
    s32 n = 0;

    *count = 0;
    if (suBaddieActorArray == NULL) {
        return;
    }
    begin = suBaddieActorArray->data;
    for (a = begin; a - begin < suBaddieActorArray->cnt && n < max; a++) {
        if (is_live_monster(a, NULL) && fate_dist3d(pos, a->position) <= radius) {
            out[n++] = a;
        }
    }
    *count = n;
}

Actor *fate_find_nearest_enemy(f32 pos[3], f32 radius) {
    Actor *begin;
    Actor *a;
    Actor *best = NULL;
    f32 bestDist = radius;

    if (suBaddieActorArray == NULL) {
        return NULL;
    }
    begin = suBaddieActorArray->data;
    for (a = begin; a - begin < suBaddieActorArray->cnt; a++) {
        f32 d;
        if (!is_live_monster(a, NULL)) {
            continue;
        }
        d = fate_dist3d(pos, a->position);
        if (d < bestDist) {
            bestDist = d;
            best = a;
        }
    }
    return best;
}

s32 fate_monster_damage_scale(void) {
    s32 depth = fate_depth();
    if (depth <= 0) {
        return 0;
    }
    return (12 + 7 * (depth - 1)) * difficulty_pct() / 100;
}

void fate_combat_frame(f32 dt) {
    Actor *begin;
    Actor *a;
    ActorMarker *fearVictims[32];
    s32 fearDamage[32];
    s32 nFear = 0;
    s32 i;

    if (gFateTargetTimer > 0.0f) {
        gFateTargetTimer -= dt;
        if (gFateTargetTimer <= 0.0f) {
            gFateTargetActor = NULL;
            fate_hud_mark_dirty();
        }
    }
    if (suBaddieActorArray == NULL) {
        return;
    }
    begin = suBaddieActorArray->data;
    for (a = begin; a - begin < suBaddieActorArray->cnt; a++) {
        FateEnemy *e;
        if (a->despawn_flag || a->marker == NULL || fate_roster_find(a->marker->id) == NULL) {
            continue;
        }
        /* Touching the data here also applies champion scaling soon after spawning. */
        e = fate_enemy_get(a->marker);
        if (e == NULL || e->maxHp <= 0 || e->killed) {
            continue;
        }
        if (e->champion == CHAMP_REGEN && e->hp < e->maxHp) {
            e->regenAccum += dt * (f32)e->maxHp * 0.03f;
            if (e->regenAccum >= 1.0f) {
                s32 heal = (s32)e->regenAccum;
                e->regenAccum -= (f32)heal;
                e->hp = fate_mini(e->maxHp, e->hp + heal);
            }
        }
        if (e->fearTimer > 0.0f && a->marker->collidable) {
            e->fearTimer -= dt;
            e->regenAccum -= dt * (f32)(10 + gDerived.stat[STAT_MAG]) * (100 + gDerived.spellPct + gDerived.skill[SKILL_CHARM_MAGIC] * 8) / 100.0f;
            if (e->regenAccum <= -1.0f && nFear < 32) {
                s32 dmg = (s32)(-e->regenAccum);
                e->regenAccum += (f32)dmg;
                fearVictims[nFear] = a->marker;
                fearDamage[nFear] = dmg;
                nFear++;
            }
        }
    }

    /* Applied after the scan: a kill can spawn actors and move the actor array. */
    for (i = 0; i < nFear; i++) {
        fate_damage_enemy(marker_getActor(fearVictims[i]), fearDamage[i], TRUE, FALSE, NULL);
    }
}
