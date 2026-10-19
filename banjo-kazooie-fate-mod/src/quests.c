#include "fate.h"

/*
 * Fate's town quests. Grove's residents post requests on a board of three.
 * Accept up to three at a time, complete them in the worlds, then return to
 * Grove to collect gold, experience, fame and sometimes an item.
 */

/* Monsters that live in each world, used to pick slay and champion targets. */
typedef struct {
    u8 level;
    u16 markers[4];
} LevelMonsters;

static const LevelMonsters sLevelMonsters[] = {
    { LEVEL_1_MUMBOS_MOUNTAIN,     { 0x005, 0x004, 0x003, 0 } },
    { LEVEL_2_TREASURE_TROVE_COVE, { 0x013, 0x015, 0, 0 } },
    { LEVEL_3_CLANKERS_CAVERN,     { 0x029, 0x0DD, 0, 0 } },
    { LEVEL_4_BUBBLEGLOOP_SWAMP,   { 0x0C1, 0x0C5, 0x0C2, 0 } },
    { LEVEL_7_GOBIS_VALLEY,        { 0x253, 0x0AD, 0, 0 } },
    { LEVEL_A_MAD_MONSTER_MANSION, { 0x096, 0x127, 0, 0 } },
    { LEVEL_9_RUSTY_BUCKET_BAY,    { 0x21A, 0x0C9, 0x065, 0 } },
    { LEVEL_8_CLICK_CLOCK_WOOD,    { 0x1E2, 0x1C5, 0x1B2, 0 } },
    { LEVEL_6_LAIR,                { 0x1EA, 0x295, 0x1F1, 0 } },
};

static const LevelMonsters *monsters_for(s32 level) {
    s32 i;
    for (i = 0; i < (s32)(sizeof(sLevelMonsters) / sizeof(sLevelMonsters[0])); i++) {
        if (sLevelMonsters[i].level == level) {
            return &sLevelMonsters[i];
        }
    }
    return NULL;
}

static s32 pick_monster(const LevelMonsters *lm) {
    s32 n = 0;
    while (n < 4 && lm->markers[n] != 0) {
        n++;
    }
    return n ? lm->markers[fate_rand_range(0, n - 1)] : 0x005;
}

static bool level_entered(s32 level) {
    switch (level) {
        case LEVEL_1_MUMBOS_MOUNTAIN:     return TRUE;
        case LEVEL_2_TREASURE_TROVE_COVE: return fileProgressFlag_get(FILEPROG_B2_HAS_ENTERED_TTC);
        case LEVEL_3_CLANKERS_CAVERN:     return fileProgressFlag_get(FILEPROG_B8_HAS_ENTERED_CC);
        case LEVEL_4_BUBBLEGLOOP_SWAMP:   return fileProgressFlag_get(FILEPROG_B1_HAS_ENTERED_BGS);
        case LEVEL_5_FREEZEEZY_PEAK:      return fileProgressFlag_get(FILEPROG_B6_HAS_ENTERED_FP);
        case LEVEL_7_GOBIS_VALLEY:        return fileProgressFlag_get(FILEPROG_B3_HAS_ENTERED_GV);
        case LEVEL_A_MAD_MONSTER_MANSION: return fileProgressFlag_get(FILEPROG_B7_HAS_ENTERED_MMM);
        case LEVEL_9_RUSTY_BUCKET_BAY:    return fileProgressFlag_get(FILEPROG_B4_HAS_ENTERED_RBB);
        case LEVEL_8_CLICK_CLOCK_WOOD:    return fileProgressFlag_get(FILEPROG_B5_HAS_ENTERED_CCW);
        case LEVEL_6_LAIR:                return fileProgressFlag_get(FILEPROG_97_ENTERED_LAIR_TEXT);
        default:                          return FALSE;
    }
}

static const u8 sQuestLevels[] = {
    LEVEL_1_MUMBOS_MOUNTAIN, LEVEL_2_TREASURE_TROVE_COVE, LEVEL_3_CLANKERS_CAVERN, LEVEL_4_BUBBLEGLOOP_SWAMP,
    LEVEL_5_FREEZEEZY_PEAK, LEVEL_7_GOBIS_VALLEY, LEVEL_A_MAD_MONSTER_MANSION, LEVEL_9_RUSTY_BUCKET_BAY,
    LEVEL_8_CLICK_CLOCK_WOOD, LEVEL_6_LAIR,
};

static s32 pick_level(void) {
    u8 options[16];
    s32 n = 0;
    s32 i;
    for (i = 0; i < (s32)sizeof(sQuestLevels); i++) {
        if (level_entered(sQuestLevels[i])) {
            options[n++] = sQuestLevels[i];
        }
    }
    if (n == 0) {
        return LEVEL_1_MUMBOS_MOUNTAIN;
    }
    /* Favour the frontier: the most recently opened worlds come up more often. */
    if (n > 2 && fate_chance(5000)) {
        return options[fate_rand_range(n - 2, n - 1)];
    }
    return options[fate_rand_range(0, n - 1)];
}

static void make_quest(FateQuest *q) {
    s32 tries;
    s32 depth;
    s32 k10; /* reward multiplier x10 */

    bzero(q, sizeof(*q));
    for (tries = 0; tries < 12; tries++) {
        s32 level = pick_level();
        s32 r = fate_rand_range(0, 99);
        const LevelMonsters *lm = monsters_for(level);

        q->level = (u8)level;
        q->giver = (u8)fate_rand_range(0, gQuestGiverCount - 1);
        q->seed = (u16)fate_rand();
        if (r < 35 && lm != NULL) {
            q->type = QUEST_SLAY;
            q->target = (u16)pick_monster(lm);
            q->required = (u16)fate_rand_range(4, 8);
            k10 = 10;
        } else if (r < 55 && lm != NULL) {
            q->type = QUEST_CHAMPION;
            q->target = (u16)pick_monster(lm);
            q->required = 1;
            k10 = 15;
        } else if (r < 75 && lm != NULL) {
            q->type = QUEST_FETCH;
            q->seed &= 0xFF;
            q->required = 1;
            k10 = 12;
        } else if (r < 90) {
            s32 remaining = 100 - itemscore_noteScores_get(level);
            if (level == LEVEL_6_LAIR || remaining < 10) {
                continue;
            }
            q->type = QUEST_NOTES;
            q->required = (u16)fate_mini(remaining, fate_rand_range(10, 25));
            k10 = 9;
        } else {
            s32 remaining = 10 - jiggyscore_leveltotal(level);
            if (level == LEVEL_6_LAIR || remaining < 1) {
                continue;
            }
            q->type = QUEST_JIGGIES;
            q->required = (u16)fate_mini(remaining, fate_rand_range(1, 2));
            k10 = 13;
        }
        depth = fate_maxi(1, fate_depth_for_level(level));
        q->rewardGold = (u32)((80 + 60 * depth) * k10 / 10);
        q->rewardXp = (u32)((150 + 120 * depth) * k10 / 10);
        q->rewardFame = (u16)((20 + 6 * depth) * k10 / 10);
        q->rewardItem = fate_chance(3500);
        q->state = QSTATE_ACTIVE;
        return;
    }
    /* Fallback that is always possible. */
    q->type = QUEST_SLAY;
    q->level = LEVEL_1_MUMBOS_MOUNTAIN;
    q->target = 0x005;
    q->required = 5;
    q->rewardGold = 120;
    q->rewardXp = 250;
    q->rewardFame = 25;
    q->state = QSTATE_ACTIVE;
}

void fate_quests_refresh_board(void) {
    s32 i;
    for (i = 0; i < QUEST_BOARD_SIZE; i++) {
        if (gFate.board[i].type == QUEST_NONE) {
            make_quest(&gFate.board[i]);
        }
    }
}

bool fate_quest_accept(s32 boardIndex) {
    s32 i;
    if (boardIndex < 0 || boardIndex >= QUEST_BOARD_SIZE || gFate.board[boardIndex].type == QUEST_NONE) {
        return FALSE;
    }
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        if (gFate.quests[i].type == QUEST_NONE) {
            gFate.quests[i] = gFate.board[boardIndex];
            gFate.quests[i].progress = 0;
            gFate.quests[i].state = QSTATE_ACTIVE;
            gFate.board[boardIndex].type = QUEST_NONE;
            make_quest(&gFate.board[boardIndex]);
            fate_message("Quest accepted. Check your quest log (D-Pad Up).", MSG_COLOR_QUEST);
            fate_play_jingle(COMUSIC_0_DING_A);
            return TRUE;
        }
    }
    fate_message("You can only carry three quests at once.", MSG_COLOR_BAD);
    return FALSE;
}

bool fate_quest_turn_in(s32 index, bool keepArtifact) {
    FateQuest *q;
    char msg[96];

    if (index < 0 || index >= MAX_ACTIVE_QUESTS) {
        return FALSE;
    }
    q = &gFate.quests[index];
    if (q->type == QUEST_NONE || q->state != QSTATE_READY) {
        return FALSE;
    }
    if (!fate_in_town()) {
        fate_message("Return to Grove to collect your reward.", MSG_COLOR_BAD);
        return FALSE;
    }
    msg[0] = '\0';
    fate_strcat(msg, "Quest complete! +");
    fate_strcat_int(msg, (s32)q->rewardGold);
    fate_strcat(msg, " gold");
    fate_message(msg, MSG_COLOR_QUEST);
    fate_banner("QUEST COMPLETE!", MSG_COLOR_QUEST);
    fate_add_gold((s32)q->rewardGold);
    fate_gain_xp((s32)q->rewardXp, TRUE);
    fate_gain_fame(q->rewardFame);
    if (q->rewardItem) {
        FateItem item;
        char name[64];
        fate_item_generate(&item, fate_maxi(1, fate_depth_for_level(q->level)) + 1, RARITY_MAGIC + (fate_chance(2500) ? 1 : 0), gDerived.magicFind);
        fate_identify(&item);
        fate_item_name(&item, name, sizeof(name));
        msg[0] = '\0';
        if (fate_inventory_add(&item)) {
            fate_strcat(msg, "Reward: ");
        } else {
            fate_strcat(msg, "Reward (pack full, sold): ");
            fate_add_gold(fate_item_sell_price(&item));
        }
        fate_strcat(msg, name);
        fate_message(msg, fate_rarity_color(item.rarity));
    }
    fate_play_jingle(COMUSIC_3B_MINIGAME_VICTORY);
    gFate.questsCompleted++;
    bzero(q, sizeof(*q));
    return TRUE;
}

void fate_quest_abandon(s32 index) {
    if (index >= 0 && index < MAX_ACTIVE_QUESTS) {
        bzero(&gFate.quests[index], sizeof(FateQuest));
    }
}

static const char *plural_monster(s32 marker) {
    const FateRosterEntry *r = fate_roster_find(marker);
    return r ? r->name : "monster";
}

void fate_quest_title(const FateQuest *q, char *out, s32 cap) {
    char name[64];
    out[0] = '\0';
    switch (q->type) {
        case QUEST_SLAY:
            fate_strcat(out, "Slay ");
            fate_strcat_int(out, q->required);
            fate_strcat(out, " ");
            fate_strcat(out, plural_monster(q->target));
            fate_strcat(out, "s");
            break;
        case QUEST_CHAMPION:
            fate_champion_name(q->seed, q->target, name, sizeof(name));
            fate_strcat(out, "Hunt ");
            fate_strcat(out, name);
            break;
        case QUEST_FETCH:
            fate_strcat(out, "Recover ");
            fate_strcat(out, gArtifactNames[(q->seed & 0xFF) % gArtifactNameCount]);
            break;
        case QUEST_NOTES:
            fate_strcat(out, "Gather ");
            fate_strcat_int(out, q->required);
            fate_strcat(out, " notes");
            break;
        case QUEST_JIGGIES:
            fate_strcat(out, "Find ");
            fate_strcat_int(out, q->required);
            fate_strcat(out, q->required == 1 ? " jiggy" : " jiggies");
            break;
        default:
            fate_strcat(out, "(no quest)");
            return;
    }
    fate_strcat(out, " in ");
    fate_strcat(out, fate_level_name(q->level));
}

void fate_quest_describe(const FateQuest *q, char *out, s32 cap) {
    out[0] = '\0';
    if (q->type == QUEST_NONE) {
        return;
    }
    fate_strcat(out, "From ");
    fate_strcat(out, gQuestGivers[q->giver % gQuestGiverCount]);
    fate_strcat(out, ".  ");
    if (q->state == QSTATE_READY) {
        fate_strcat(out, "DONE - return to Grove!");
    } else if (q->type == QUEST_FETCH) {
        fate_strcat(out, "Defeat monsters there to find it.");
    } else if (q->type == QUEST_CHAMPION) {
        fate_strcat(out, "It lurks among the ");
        fate_strcat(out, plural_monster(q->target));
        fate_strcat(out, "s.");
    } else {
        fate_strcat(out, "Progress ");
        fate_strcat_int(out, q->progress);
        fate_strcat(out, "/");
        fate_strcat_int(out, q->required);
    }
    fate_strcat(out, "\nReward: ");
    fate_strcat_int(out, (s32)q->rewardGold);
    fate_strcat(out, " gold, ");
    fate_strcat_int(out, (s32)q->rewardXp);
    fate_strcat(out, " xp, ");
    fate_strcat_int(out, q->rewardFame);
    fate_strcat(out, " fame");
    if (q->rewardItem) {
        fate_strcat(out, ", and an item");
    }
}

static void quest_progress(FateQuest *q, s32 amount) {
    char msg[96];
    char title[96];
    if (q->state != QSTATE_ACTIVE) {
        return;
    }
    q->progress = (u16)fate_mini(q->required, q->progress + amount);
    fate_quest_title(q, title, sizeof(title));
    msg[0] = '\0';
    if (q->progress >= q->required) {
        q->state = QSTATE_READY;
        fate_strcat(msg, "Quest objective complete: ");
        fate_strcat(msg, title);
        fate_play_jingle(COMUSIC_2D_PUZZLE_SOLVED_FANFARE);
    } else {
        fate_strcat(msg, "Quest: ");
        fate_strcat_int(msg, q->progress);
        fate_strcat(msg, "/");
        fate_strcat_int(msg, q->required);
    }
    fate_message(msg, MSG_COLOR_QUEST);
}

void fate_quest_on_kill(Actor *actor, FateEnemy *enemy) {
    s32 level = level_get();
    s32 i;
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        FateQuest *q = &gFate.quests[i];
        if (q->type == QUEST_NONE || q->level != level || q->state != QSTATE_ACTIVE) {
            continue;
        }
        switch (q->type) {
            case QUEST_SLAY: {
                const FateRosterEntry *want = fate_roster_find(q->target);
                const FateRosterEntry *got = fate_roster_find(actor->marker->id);
                /* Flipped/alternate variants share a name and count too. */
                if (want != NULL && got != NULL && (q->target == actor->marker->id || want->name == got->name)) {
                    quest_progress(q, 1);
                }
                break;
            }
            case QUEST_CHAMPION:
                if (enemy->questChampion == i + 1) {
                    quest_progress(q, 1);
                }
                break;
            case QUEST_FETCH:
                q->seed += 0x100; /* count kills in the high byte so the artifact name stays stable */
                if (fate_chance(2500) || (q->seed >> 8) >= 6) {
                    char msg[96];
                    msg[0] = '\0';
                    fate_strcat(msg, "You found ");
                    fate_strcat(msg, gArtifactNames[(q->seed & 0xFF) % gArtifactNameCount]);
                    fate_strcat(msg, "!");
                    fate_message(msg, 0xD08CFFFF);
                    quest_progress(q, 1);
                }
                break;
            default:
                break;
        }
    }
}

void fate_quest_on_note(enum level_e level) {
    s32 i;
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        FateQuest *q = &gFate.quests[i];
        if (q->type == QUEST_NOTES && q->level == level) {
            quest_progress(q, 1);
        }
    }
}

void fate_quest_on_jiggy(enum level_e level) {
    s32 i;
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        FateQuest *q = &gFate.quests[i];
        if (q->type == QUEST_JIGGIES && q->level == level) {
            quest_progress(q, 1);
        }
    }
}

void fate_quest_on_level_enter(enum level_e level) {
    if (level == LEVEL_B_SPIRAL_MOUNTAIN) {
        fate_quests_refresh_board();
    }
}

s32 fate_quest_champion_for(enum level_e level, s32 markerId) {
    s32 i;
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        const FateQuest *q = &gFate.quests[i];
        if (q->type == QUEST_CHAMPION && q->level == level && q->state == QSTATE_ACTIVE) {
            const FateRosterEntry *want = fate_roster_find(q->target);
            const FateRosterEntry *got = fate_roster_find(markerId);
            if (want != NULL && got != NULL && want->name == got->name) {
                return i;
            }
        }
    }
    return -1;
}
