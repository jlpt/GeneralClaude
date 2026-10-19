#include "fate.h"

/*
 * Fate's three schools of magic, learned from Cheato's spellbooks and cast
 * with D-Pad Right (L or D-Pad Left cycles the selected spell).
 */

f32 gAuraTimer;
f32 gHasteTimer;
static f32 sCooldowns[SPELL_COUNT];


bool fate_spell_known(s32 spell) {
    return spell >= 0 && spell < SPELL_COUNT && (gFate.spellsKnown & (1u << spell));
}

s32 fate_spell_mana_cost(s32 spell) {
    const FateSpellDef *d = &gSpellDefs[spell];
    s32 cost = d->manaCost + d->manaCost * gFate.level / 30;
    cost = cost * (100 - fate_mini(gDerived.skill[d->school] * 3, 45)) / 100;
    return cost < 1 ? 1 : cost;
}

f32 fate_spell_cooldown(s32 spell) {
    return sCooldowns[spell];
}

static s32 spell_power(s32 spell) {
    return 100 + gDerived.spellPct + gDerived.skill[gSpellDefs[spell].school] * 8;
}

/* A puff of magic smoke, the same effect enemies make when they vanish. */
static void magic_puff(f32 x, f32 y, f32 z) {
    __spawnQueue_add_4((GenFunction_4)spawnQueue_actor_f32, ACTOR_4C_STEAM,
                       reinterpret_cast(s32, x), reinterpret_cast(s32, y), reinterpret_cast(s32, z));
}

void fate_cycle_spell(void) {
    s32 i;
    s32 s = gFate.selectedSpell;
    char msg[48];
    for (i = 0; i < SPELL_COUNT; i++) {
        s = (s + 1) % SPELL_COUNT;
        if (fate_spell_known(s)) {
            gFate.selectedSpell = (u8)s;
            msg[0] = '\0';
            fate_strcat(msg, "Spell: ");
            fate_strcat(msg, gSpellDefs[s].name);
            fate_message(msg, MSG_COLOR_FAME);
            fate_hud_mark_dirty();
            return;
        }
    }
}

bool fate_cast_spell(s32 spell) {
    const FateSpellDef *d;
    s32 cost;
    s32 power;
    f32 pos[3];

    if (!fate_spell_known(spell)) {
        fate_message("You don't know any spells. Visit Cheato in Grove.", MSG_COLOR_BAD);
        return FALSE;
    }
    d = &gSpellDefs[spell];
    if (gDerived.stat[STAT_MAG] < d->reqMag) {
        fate_message("Your Magic is too low to cast that.", MSG_COLOR_BAD);
        return FALSE;
    }
    if (sCooldowns[spell] > 0.0f) {
        return FALSE;
    }
    cost = fate_spell_mana_cost(spell);
    if (gFateMana < (f32)cost) {
        fate_message("Not enough mana.", MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return FALSE;
    }

    power = spell_power(spell);
    player_getPosition(pos);

    switch (spell) {
        case SPELL_EGG_STORM: {
            ActorMarker *victims[24];
            Actor *found[24];
            s32 n, i;
            s32 dmg = (12 + gDerived.stat[STAT_MAG] * 4 / 5 + gFate.level * 3 / 2) * power / 100;
            fate_find_enemies_near(pos, 650.0f, found, &n, 24);
            for (i = 0; i < n; i++) {
                victims[i] = found[i]->marker;
            }
            for (i = 0; i < 8; i++) {
                f32 ang = (f32)i * 45.0f;
                magic_puff(pos[0] + ml_sin_deg(ang) * 300.0f, pos[1] + 40.0f, pos[2] + ml_cos_deg(ang) * 300.0f);
            }
            for (i = 0; i < n; i++) {
                Actor *a = marker_getActor(victims[i]);
                magic_puff(a->position[0], a->position[1] + 50.0f, a->position[2]);
                fate_damage_enemy(a, dmg, TRUE, TRUE, NULL);
            }
            fate_play_sfx(SFX_1B_EXPLOSION_1);
            break;
        }
        case SPELL_THUNDER_PECK: {
            Actor *target = fate_find_nearest_enemy(pos, 1200.0f);
            s32 dmg = (30 + gDerived.stat[STAT_MAG] * 8 / 5 + gFate.level * 3) * power / 100;
            if (target == NULL) {
                fate_message("No foe in range.", MSG_COLOR_DEFAULT);
                return FALSE;
            }
            magic_puff(target->position[0], target->position[1] + 80.0f, target->position[2]);
            fate_damage_enemy(target, dmg, TRUE, TRUE, NULL);
            fate_play_sfx(SFX_B5_THUNDERBOLT);
            break;
        }
        case SPELL_HONEY_HEAL: {
            f32 heal = (f32)((20 + gDerived.stat[STAT_MAG] * 3 / 2 + gFate.level * 2) * power / 100);
            gFateHp = fate_clampf(gFateHp + heal, 0.0f, (f32)gDerived.maxHp);
            fate_health_sync_to_bk();
            magic_puff(pos[0], pos[1] + 60.0f, pos[2]);
            fate_play_sfx(SFX_A6_MAGICAL_FINISH);
            break;
        }
        case SPELL_GOLDEN_AURA:
            gAuraTimer = 20.0f + (f32)gDerived.skill[SKILL_DEFENSE_MAGIC];
            fate_play_jingle(COMUSIC_25_USING_GOLD_FEATHERS);
            fate_message("A golden aura surrounds you.", MSG_COLOR_GOLD);
            break;
        case SPELL_TURBO_TALON:
            gHasteTimer = 20.0f + (f32)gDerived.skill[SKILL_DEFENSE_MAGIC];
            fate_play_jingle(COMUSIC_57_TURBO_TRAINERS);
            break;
        case SPELL_PET_FRENZY:
            if (!fate_pet_present()) {
                fate_message("Your pet isn't here.", MSG_COLOR_BAD);
                return FALSE;
            }
            gPetFrenzyTimer = 15.0f + (f32)gDerived.skill[SKILL_CHARM_MAGIC];
            fate_play_sfx(SFX_17_JINJO_WHISTLE);
            fate_message("Your pet goes into a frenzy!", MSG_COLOR_PET);
            break;
        case SPELL_MUMBOS_FEAR: {
            Actor *found[24];
            s32 n, i;
            fate_find_enemies_near(pos, 700.0f, found, &n, 24);
            for (i = 0; i < n; i++) {
                FateEnemy *e = fate_enemy_get(found[i]->marker);
                if (e != NULL) {
                    e->fearTimer = 8.0f;
                }
            }
            fate_play_sfx(SFX_1_MUMBO_UMENAKA);
            fate_message(n > 0 ? "Mumbo's curse withers your foes!" : "The curse finds no one.", MSG_COLOR_FAME);
            break;
        }
        case SPELL_WARP_HOME:
            if (!fate_warp_to_town()) {
                return FALSE;
            }
            break;
        case SPELL_DISCERN: {
            s32 n = fate_identify_all();
            char msg[48];
            msg[0] = '\0';
            fate_strcat(msg, "Identified ");
            fate_strcat_int(msg, n);
            fate_strcat(msg, " items.");
            fate_message(msg, MSG_COLOR_FAME);
            fate_play_sfx(SFX_A6_MAGICAL_FINISH);
            break;
        }
    }

    gFateMana -= (f32)cost;
    sCooldowns[spell] = (f32)d->cooldownTenths / 10.0f;
    fate_hud_mark_dirty();
    return TRUE;
}

void fate_spells_frame(f32 dt) {
    s32 i;
    for (i = 0; i < SPELL_COUNT; i++) {
        if (sCooldowns[i] > 0.0f) {
            sCooldowns[i] -= dt;
        }
    }
    if (gAuraTimer > 0.0f) {
        gAuraTimer -= dt;
        if (gAuraTimer <= 0.0f) {
            fate_message("Your golden aura fades.", MSG_COLOR_DEFAULT);
        }
    }
    if (gHasteTimer > 0.0f) {
        gHasteTimer -= dt;
    }
}

/* ------------------------------------------------------------------ */
/* Movement speed: haste, Swift boots, "of Haste" jewelry              */
/* ------------------------------------------------------------------ */

extern f32 baphysics_target_horizontal_velocity;

RECOMP_HOOK_RETURN("baphysics_set_target_horizontal_velocity") void fate_hook_speed(void) {
    s32 bonus;
    if (!gFate.created || gFateLoadedSlot < 0) {
        return;
    }
    bonus = gDerived.moveSpeed + (gHasteTimer > 0.0f ? 50 : 0);
    if (bonus > 0 && player_getTransformation() == TRANSFORM_1_BANJO) {
        baphysics_target_horizontal_velocity *= 1.0f + (f32)bonus / 100.0f;
    }
}
