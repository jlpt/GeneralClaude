#include "fate.h"

/* ------------------------------------------------------------------ */
/* Names                                                               */
/* ------------------------------------------------------------------ */

const char *gStatNames[STAT_COUNT] = { "Strength", "Dexterity", "Vitality", "Magic" };
const char *gStatShort[STAT_COUNT] = { "STR", "DEX", "VIT", "MAG" };

const char *gSkillNames[SKILL_COUNT] = {
    "Claw Swipe",
    "Forward Roll",
    "Rat-a-tat Rap",
    "Beak Barge",
    "Beak Buster",
    "Beak Bomb",
    "Egg Firing",
    "Critical Strike",
    "Shield",
    "Attack Magic",
    "Defense Magic",
    "Charm Magic",
};

const char *gSlotNames[SLOT_COUNT] = {
    "Banjo's Weapon",
    "Kazooie's Weapon",
    "Helm",
    "Armor",
    "Backpack Shield",
    "Boots",
    "Amulet",
    "Ring",
    "Ring",
};

const char *gRarityNames[RARITY_COUNT] = { "Normal", "Magic", "Unique", "Elite", "Legendary", "Artifact" };

const char *gConsumableNames[CONS_COUNT] = {
    "Health Potion",
    "Greater Health Potion",
    "Mana Potion",
    "Greater Mana Potion",
    "Scroll of Return",
    "Scroll of Identify",
};

const u16 gConsumablePrices[CONS_COUNT] = { 25, 90, 25, 90, 60, 30 };

/* Format strings are appended after a signed value, e.g. "+12" + "% Damage". */
const char *gEffectFormats[FX_COUNT] = {
    "",
    "% Damage",
    " Damage",
    " Strength",
    " Dexterity",
    " Vitality",
    " Magic",
    " to All Attributes",
    " Health",
    " Mana",
    "% Armor",
    " Armor",
    "% Critical Strike Chance",
    "% Block Chance",
    "% Dodge Chance",
    "% Life Stolen per Hit",
    "% Gold Found",
    "% Magic Find",
    "% Experience",
    " Health per 10 Seconds",
    " Mana per 10 Seconds",
    "% Egg Damage",
    "% Beak Damage",
    "% Paw Damage",
    "% Pet Damage",
    "% Pet Health",
    "% Spell Power",
    "% Movement Speed",
    " to All Skills",
    " Damage Reflected",
    " Fire Damage",
    " Damage Absorbed",
};

/* ------------------------------------------------------------------ */
/* Item bases                                                          */
/* ------------------------------------------------------------------ */

#define BW KIND_BANJO_WEAPON
#define KW KIND_KAZOOIE_WEAPON

const FateItemBase gItemBases[] = {
    /* 0 */ { "(empty)", KIND_NONE, 0, 0, 0, 0, 0, 0, 0, 0 },

    /* Banjo's weapons: damage range applies to claw, roll and stomps */
    /* 1 */  { "Honey Claws",        BW, 1,   2,  5,   20, 10,  0,  0, 0 },
    /* 2 */  { "Wooden Club",        BW, 1,   3,  6,   25, 12,  0,  0, 0 },
    /* 3 */  { "Spiked Claws",       BW, 2,   4,  8,   60, 14,  0,  0, 0 },
    /* 4 */  { "Stone Hammer",       BW, 2,   5,  9,   70, 16,  0,  0, 0 },
    /* 5 */  { "Bear Claws",         BW, 3,   6, 11,  120, 18,  0,  0, 0 },
    /* 6 */  { "Iron Mace",          BW, 3,   7, 12,  140, 20,  0,  0, 0 },
    /* 7 */  { "War Hammer",         BW, 4,   8, 15,  220, 24,  0,  0, 0 },
    /* 8 */  { "Battle Axe",         BW, 5,  10, 18,  320, 28,  0,  0, 0 },
    /* 9 */  { "Grublin Cleaver",    BW, 5,   9, 20,  340, 26, 12,  0, 0 },
    /* 10 */ { "Morning Star",       BW, 6,  12, 22,  480, 32,  0,  0, 0 },
    /* 11 */ { "Great Axe",          BW, 7,  15, 27,  650, 36,  0,  0, 0 },
    /* 12 */ { "Rune Claws",         BW, 8,  17, 30,  900, 38, 20,  0, 0 },
    /* 13 */ { "Maul of Mayhem",     BW, 9,  20, 36, 1200, 44,  0,  0, 0 },
    /* 14 */ { "Dragon Claws",       BW, 10, 24, 42, 1700, 50, 24,  0, 0 },
    /* 15 */ { "Wizard's Staff",     BW, 3,   4,  9,  150, 10,  0, 20, 0 },
    /* 16 */ { "Mumbo's Staff",      BW, 7,   9, 18,  800, 14,  0, 40, 0 },

    /* Kazooie's weapons: damage range applies to pecks, barges, busters, bombs and eggs */
    /* 17 */ { "Beak Cap",           KW, 1,   2,  4,   20,  0, 10,  0, 0 },
    /* 18 */ { "Egg Sling",          KW, 1,   2,  5,   25,  0, 12,  0, 0 },
    /* 19 */ { "Beak Spike",         KW, 2,   3,  7,   60,  0, 14,  0, 0 },
    /* 20 */ { "Sharpened Beak",     KW, 3,   5,  9,  120,  0, 18,  0, 0 },
    /* 21 */ { "Repeating Sling",    KW, 3,   4, 10,  130,  0, 20,  0, 0 },
    /* 22 */ { "Beak Lance",         KW, 4,   7, 13,  220, 14, 22,  0, 0 },
    /* 23 */ { "Feathered Blade",    KW, 5,   8, 16,  320,  0, 26,  0, 0 },
    /* 24 */ { "Iron Beak",          KW, 6,  10, 19,  480, 18, 30,  0, 0 },
    /* 25 */ { "Golden Sling",       KW, 7,  12, 22,  650,  0, 34,  0, 0 },
    /* 26 */ { "Breegull Halberd",   KW, 8,  14, 26,  900, 24, 38,  0, 0 },
    /* 27 */ { "Thunder Beak",       KW, 9,  17, 31, 1200,  0, 42,  0, 0 },
    /* 28 */ { "Phoenix Sling",      KW, 10, 20, 37, 1700,  0, 48,  0, 0 },

    /* Helms */
    /* 29 */ { "Woolly Cap",         KIND_HELM, 1,   2, 0,   15,  0,  0,  0, 0 },
    /* 30 */ { "Leather Cap",        KIND_HELM, 1,   3, 0,   20,  0,  0,  0, 0 },
    /* 31 */ { "Grublin Hood",       KIND_HELM, 2,   5, 0,   50,  0,  0,  0, 0 },
    /* 32 */ { "Iron Helm",          KIND_HELM, 3,   8, 0,  110, 16,  0,  0, 0 },
    /* 33 */ { "Horned Helm",        KIND_HELM, 4,  11, 0,  180, 20,  0,  0, 0 },
    /* 34 */ { "Knight's Helm",      KIND_HELM, 6,  16, 0,  380, 26,  0,  0, 0 },
    /* 35 */ { "Crown of Thorns",    KIND_HELM, 8,  22, 0,  700, 32,  0,  0, 0 },
    /* 36 */ { "Wizard's Hat",       KIND_HELM, 3,   4, 0,  140,  0,  0, 18, 0 },
    /* 37 */ { "Witch's Hat",        KIND_HELM, 9,  18, 0, 1000,  0,  0, 36, 0 },

    /* Armor */
    /* 38 */ { "Patched Shorts",     KIND_ARMOR, 1,   3, 0,   20,  0,  0,  0, 0 },
    /* 39 */ { "Padded Vest",        KIND_ARMOR, 1,   5, 0,   30,  0,  0,  0, 0 },
    /* 40 */ { "Leather Jerkin",     KIND_ARMOR, 2,   8, 0,   70,  0,  0,  0, 0 },
    /* 41 */ { "Studded Leather",    KIND_ARMOR, 3,  12, 0,  140, 14,  0,  0, 0 },
    /* 42 */ { "Chain Shirt",        KIND_ARMOR, 4,  16, 0,  240, 18,  0,  0, 0 },
    /* 43 */ { "Scale Mail",         KIND_ARMOR, 5,  21, 0,  360, 24,  0,  0, 0 },
    /* 44 */ { "Plate Mail",         KIND_ARMOR, 7,  30, 0,  700, 32,  0,  0, 0 },
    /* 45 */ { "Honeycomb Plate",    KIND_ARMOR, 9,  40, 0, 1200, 38,  0,  0, 0 },
    /* 46 */ { "Mystic Robe",        KIND_ARMOR, 4,   8, 0,  260,  0,  0, 24, 0 },

    /* Backpack shields */
    /* 47 */ { "Old Backpack",       KIND_SHIELD, 1,   2, 0,   15,  0,  0,  0, 0 },
    /* 48 */ { "Wooden Buckler",     KIND_SHIELD, 1,   4, 0,   25,  0,  0,  0, 0 },
    /* 49 */ { "Reinforced Backpack",KIND_SHIELD, 2,   6, 0,   60,  0,  0,  0, 0 },
    /* 50 */ { "Kite Shield",        KIND_SHIELD, 4,  11, 0,  200, 18,  0,  0, 0 },
    /* 51 */ { "Tower Shield",       KIND_SHIELD, 6,  17, 0,  420, 26,  0,  0, 0 },
    /* 52 */ { "Clanker's Plating",  KIND_SHIELD, 8,  24, 0,  800, 32,  0,  0, 0 },
    /* 53 */ { "Grunty-Proof Pack",  KIND_SHIELD, 10, 32, 0, 1400, 38,  0,  0, 0 },

    /* Boots */
    /* 54 */ { "Sandals",            KIND_BOOTS, 1,   1, 0,   10,  0,  0,  0, 0 },
    /* 55 */ { "Leather Boots",      KIND_BOOTS, 1,   3, 0,   20,  0,  0,  0, 0 },
    /* 56 */ { "Wading Boots",       KIND_BOOTS, 3,   6, 0,  120,  0,  0,  0, 0 },
    /* 57 */ { "Turbo Trainers",     KIND_BOOTS, 5,   7, 0,  300,  0, 20,  0, 0 },
    /* 58 */ { "Iron Greaves",       KIND_BOOTS, 6,  12, 0,  400, 24,  0,  0, 0 },
    /* 59 */ { "Wizard's Slippers",  KIND_BOOTS, 4,   3, 0,  250,  0,  0, 20, 0 },
    /* 60 */ { "Snow Boots",         KIND_BOOTS, 8,  15, 0,  700,  0,  0,  0, 0 },

    /* Amulets */
    /* 61 */ { "Shark Tooth Necklace", KIND_AMULET, 1, 0, 0,  40, 0, 0, 0, 0 },
    /* 62 */ { "Jinjo Charm",          KIND_AMULET, 3, 0, 0, 120, 0, 0, 0, 0 },
    /* 63 */ { "Silver Amulet",        KIND_AMULET, 5, 0, 0, 260, 0, 0, 0, 0 },
    /* 64 */ { "Gold Amulet",          KIND_AMULET, 8, 0, 0, 500, 0, 0, 0, 0 },
    /* 65 */ { "Starry Pendant",       KIND_AMULET, 10, 0, 0, 900, 0, 0, 0, 0 },

    /* Rings */
    /* 66 */ { "Copper Ring",          KIND_RING, 1, 0, 0,  35, 0, 0, 0, 0 },
    /* 67 */ { "Silver Ring",          KIND_RING, 3, 0, 0, 110, 0, 0, 0, 0 },
    /* 68 */ { "Gold Ring",            KIND_RING, 6, 0, 0, 280, 0, 0, 0, 0 },
    /* 69 */ { "Mystic Band",          KIND_RING, 9, 0, 0, 600, 0, 0, 0, 0 },

    /* Quest artifacts */
    /* 70 */ { "Artifact",             KIND_ARTIFACT, 0, 0, 0, 100, 0, 0, 0, 0 },
};
const s32 gItemBaseCount = sizeof(gItemBases) / sizeof(gItemBases[0]);

/* ------------------------------------------------------------------ */
/* Affixes                                                             */
/* ------------------------------------------------------------------ */

#define M_W   (KINDBIT(KIND_BANJO_WEAPON) | KINDBIT(KIND_KAZOOIE_WEAPON))
#define M_A   (KINDBIT(KIND_HELM) | KINDBIT(KIND_ARMOR) | KINDBIT(KIND_SHIELD) | KINDBIT(KIND_BOOTS))
#define M_J   (KINDBIT(KIND_AMULET) | KINDBIT(KIND_RING))
#define M_ALL (M_W | M_A | M_J)

const FateAffix gAffixes[] = {
    /* 0 */ { "", 0, FX_NONE, 0, 0, 0, 0, 0 },
    /* prefixes */
    /* 1 */  { "Sharp",          0, FX_DMG_PCT,     M_W,                             5, 15,  8, 1 },
    /* 2 */  { "Brutal",         0, FX_DMG_FLAT,    M_W,                             1,  3,  6, 1 },
    /* 3 */  { "Burning",        0, FX_FIRE_DMG,    M_W | M_J,                       1,  4,  5, 2 },
    /* 4 */  { "Vampiric",       0, FX_LIFE_STEAL,  M_W | M_J,                       2,  4,  1, 4 },
    /* 5 */  { "Mighty",         0, FX_STR,         M_ALL,                           2,  5,  4, 1 },
    /* 6 */  { "Nimble",         0, FX_DEX,         M_ALL,                           2,  5,  4, 1 },
    /* 7 */  { "Sturdy",         0, FX_VIT,         M_A | M_J,                       2,  5,  4, 1 },
    /* 8 */  { "Mystic",         0, FX_MAG,         M_A | M_J | M_W,                 2,  5,  4, 1 },
    /* 9 */  { "Reinforced",     0, FX_ARMOR_PCT,   M_A,                            10, 25,  8, 1 },
    /* 10 */ { "Honeyed",        0, FX_HP_REGEN,    M_A | M_J,                       2,  6,  4, 2 },
    /* 11 */ { "Lucky",          0, FX_MAGIC_FIND,  M_J | M_A,                       5, 15,  5, 2 },
    /* 12 */ { "Golden",         0, FX_GOLD_FIND,   M_ALL,                          10, 25,  8, 1 },
    /* 13 */ { "Swift",          0, FX_MOVE_SPEED,  KINDBIT(KIND_BOOTS),             5, 10,  2, 3 },
    /* 14 */ { "Jinjo-Blessed",  0, FX_PET_DMG,     M_J,                            10, 25,  8, 2 },
    /* 15 */ { "Mumbo's",        0, FX_SPELL_PWR,   M_W | M_J | KINDBIT(KIND_HELM), 10, 20,  8, 2 },
    /* 16 */ { "Breegull's",     0, FX_EGG_DMG,     KINDBIT(KIND_KAZOOIE_WEAPON) | M_J, 10, 25, 8, 1 },
    /* 17 */ { "Bear's",         0, FX_PAW_DMG,     KINDBIT(KIND_BANJO_WEAPON) | M_J,   10, 25, 8, 1 },
    /* 18 */ { "Spiked",         0, FX_THORNS,      M_A,                             2,  6,  5, 2 },
    /* 19 */ { "Masterful",      0, FX_ALL_SKILLS,  M_W | M_J,                       1,  1,  1, 8 },
    /* 20 */ { "Hardened",       0, FX_DMG_REDUCE,  M_A,                             1,  2,  3, 3 },
    /* suffixes */
    /* 21 */ { "of the Bear",     1, FX_STR,        M_ALL,                           3,  6,  4, 1 },
    /* 22 */ { "of the Breegull", 1, FX_DEX,        M_ALL,                           3,  6,  4, 1 },
    /* 23 */ { "of Vitality",     1, FX_VIT,        M_ALL,                           3,  6,  4, 1 },
    /* 24 */ { "of Wizardry",     1, FX_MAG,        M_ALL,                           3,  6,  4, 1 },
    /* 25 */ { "of Health",       1, FX_HP,         M_A | M_J,                       8, 20, 15, 1 },
    /* 26 */ { "of the Mind",     1, FX_MANA,       M_A | M_J,                       5, 12,  8, 1 },
    /* 27 */ { "of Precision",    1, FX_CRIT,       M_W | M_J,                       2,  4,  2, 2 },
    /* 28 */ { "of Blocking",     1, FX_BLOCK,      KINDBIT(KIND_SHIELD),            4,  8,  2, 1 },
    /* 29 */ { "of Evasion",      1, FX_DODGE,      KINDBIT(KIND_BOOTS) | KINDBIT(KIND_ARMOR) | M_J, 2, 5, 2, 2 },
    /* 30 */ { "of Learning",     1, FX_XP,         M_J | KINDBIT(KIND_HELM),        3,  8,  3, 2 },
    /* 31 */ { "of Fortune",      1, FX_GOLD_FIND,  M_ALL,                          10, 20,  6, 1 },
    /* 32 */ { "of Plenty",       1, FX_MAGIC_FIND, M_ALL,                           5, 12,  5, 2 },
    /* 33 */ { "of the Hive",     1, FX_HP_REGEN,   M_ALL,                           3,  6,  4, 2 },
    /* 34 */ { "of Slaughter",    1, FX_DMG_PCT,    M_W,                            10, 20,  8, 3 },
    /* 35 */ { "of the Jinjo",    1, FX_PET_HP,     M_J | M_A,                      10, 25,  8, 2 },
    /* 36 */ { "of Haste",        1, FX_MOVE_SPEED, KINDBIT(KIND_BOOTS) | M_J,       4,  8,  2, 3 },
    /* 37 */ { "of Thorns",       1, FX_THORNS,     M_A,                             3,  8,  5, 2 },
    /* 38 */ { "of Clarity",      1, FX_MANA_REGEN, M_J | KINDBIT(KIND_HELM),        2,  5,  3, 2 },
    /* 39 */ { "of the Witch",    1, FX_SPELL_PWR,  M_J | KINDBIT(KIND_HELM) | M_W,  8, 16,  6, 3 },
    /* 40 */ { "of the Champion", 1, FX_ALL_STATS,  M_J | M_A | M_W,                 1,  3,  2, 6 },
};
const s32 gAffixCount = sizeof(gAffixes) / sizeof(gAffixes[0]);

/* ------------------------------------------------------------------ */
/* Uniques                                                             */
/* ------------------------------------------------------------------ */

const FateUnique gUniques[] = {
    /* 0 */  { "", 0, 0, { 0, 0, 0 }, { 0, 0, 0 } },
    /* 1 */  { "Bottles' Spectacles",     36, 2, { FX_MAG, FX_XP, FX_MANA },               {  8, 10,  20 } },
    /* 2 */  { "Mumbo's Skull Staff",     16, 6, { FX_SPELL_PWR, FX_MAG, FX_MANA_REGEN },  { 40, 12,   6 } },
    /* 3 */  { "Broomstick of Gruntilda",  7, 5, { FX_DMG_PCT, FX_FIRE_DMG, FX_LIFE_STEAL },{ 40,  8,   4 } },
    /* 4 */  { "Clanker's Gold Tooth",    68, 3, { FX_GOLD_FIND, FX_MAGIC_FIND, FX_VIT },   { 80, 25,   5 } },
    /* 5 */  { "Tooty's Hair Bow",        62, 2, { FX_ALL_STATS, FX_HP, FX_XP },            {  4, 40,   8 } },
    /* 6 */  { "Bigbutt's Horns",         33, 3, { FX_STR, FX_ARMOR_PCT, FX_THORNS },       { 10, 40,  10 } },
    /* 7 */  { "Conga's Orange Sling",    21, 2, { FX_EGG_DMG, FX_DEX, FX_CRIT },           { 60,  8,   4 } },
    /* 8 */  { "Nipper's Claws",           5, 2, { FX_PAW_DMG, FX_DMG_FLAT, FX_CRIT },      { 50,  5,   5 } },
    /* 9 */  { "Boggy's Belly Plate",     44, 6, { FX_HP, FX_ARMOR_PCT, FX_DMG_REDUCE },    { 80, 30,   4 } },
    /* 10 */ { "Tiptup's Shell",          51, 5, { FX_BLOCK, FX_ARMOR_PCT, FX_VIT },        { 15, 50,   8 } },
    /* 11 */ { "Gobi's Hump Pack",        49, 4, { FX_HP_REGEN, FX_MANA, FX_BLOCK },        { 12, 30,   8 } },
    /* 12 */ { "Wozza's Snow Boots",      60, 6, { FX_MOVE_SPEED, FX_DODGE, FX_DEX },       { 15,  8,   8 } },
    /* 13 */ { "Jinjonator's Fist",       14, 9, { FX_DMG_PCT, FX_ALL_SKILLS, FX_STR },     { 60,  2,  12 } },
    /* 14 */ { "Kazooie's Lucky Feather", 23, 4, { FX_BEAK_DMG, FX_DODGE, FX_CRIT },        { 50,  6,   6 } },
    /* 15 */ { "Mr. Vile's Fang",         67, 4, { FX_LIFE_STEAL, FX_DMG_PCT, FX_PET_DMG }, {  8, 20,  30 } },
    /* 16 */ { "Brentilda's Locket",      65, 7, { FX_HP_REGEN, FX_SPELL_PWR, FX_MANA_REGEN }, { 15, 25, 8 } },
    /* 17 */ { "Cheato's Bookmark",       66, 1, { FX_MAG, FX_SPELL_PWR, FX_XP },           { 10, 20,  12 } },
    /* 18 */ { "Captain Blubber's Hook",   9, 3, { FX_GOLD_FIND, FX_DMG_PCT, FX_THORNS },   { 50, 30,   6 } },
    /* 19 */ { "Loggo's Golden Plunger",   2, 1, { FX_DMG_PCT, FX_MAGIC_FIND, FX_HP },      { 35, 40,  20 } },
    /* 20 */ { "Twinkly's Star",          64, 5, { FX_PET_HP, FX_PET_DMG, FX_ALL_STATS },   { 60, 40,   3 } },
    /* 21 */ { "Klungo's Spectacles",     37, 8, { FX_MAG, FX_MANA, FX_SPELL_PWR },         { 15, 50,  30 } },
    /* 22 */ { "Dingpot's Ladle",          4, 2, { FX_FIRE_DMG, FX_DMG_PCT, FX_HP_REGEN },  {  6, 25,   6 } },
    /* 23 */ { "Turbo Trainers of Legend",57, 5, { FX_MOVE_SPEED, FX_DEX, FX_DODGE },       { 25, 10,   6 } },
};
const s32 gUniqueCount = sizeof(gUniques) / sizeof(gUniques[0]);

/* ------------------------------------------------------------------ */
/* Gems                                                                */
/* ------------------------------------------------------------------ */

const FateGem gGems[] = {
    /* 0 */  { "",                FX_NONE,       0,   0 },
    /* 1 */  { "Chipped Ruby",    FX_STR,        3,  40 },
    /* 2 */  { "Ruby",            FX_STR,        6, 150 },
    /* 3 */  { "Chipped Emerald", FX_DEX,        3,  40 },
    /* 4 */  { "Emerald",         FX_DEX,        6, 150 },
    /* 5 */  { "Chipped Topaz",   FX_VIT,        3,  40 },
    /* 6 */  { "Topaz",           FX_VIT,        6, 150 },
    /* 7 */  { "Chipped Sapphire",FX_MAG,        3,  40 },
    /* 8 */  { "Sapphire",        FX_MAG,        6, 150 },
    /* 9 */  { "Honey Amber",     FX_HP_REGEN,   4, 120 },
    /* 10 */ { "Jiggy Shard",     FX_ALL_STATS,  2, 200 },
    /* 11 */ { "Diamond",         FX_DMG_PCT,   10, 260 },
    /* 12 */ { "Golden Note",     FX_GOLD_FIND, 15, 120 },
};
const s32 gGemCount = sizeof(gGems) / sizeof(gGems[0]);

/* ------------------------------------------------------------------ */
/* Fish                                                                */
/* ------------------------------------------------------------------ */

const FateFishDef gFishDefs[FISH_COUNT] = {
    { "Gloop Guppy",       1, 40, PET_FORM_ORANGE_JINJO,  4 },
    { "Blubber Snapper",   2, 30, PET_FORM_BLUE_JINJO,    6 },
    { "Clanker Carp",      3, 25, PET_FORM_GREEN_JINJO,   8 },
    { "Swamp Sucker",      4, 25, PET_FORM_PINK_JINJO,   10 },
    { "Freezeezy Trout",   5, 20, PET_FORM_BLUE_JINJO,   12 },
    { "Sandybutt Eel",     6, 18, PET_FORM_ORANGE_JINJO, 14 },
    { "Spooky Pike",       7, 15, PET_FORM_GREEN_JINJO,  16 },
    { "Rusty Catfish",     8, 15, PET_FORM_PINK_JINJO,   18 },
    { "Seasonal Koi",      9, 12, PET_FORM_ORANGE_JINJO, 22 },
    { "Bigbutt Bass",      3,  5, PET_FORM_BIGBUTT,      30 },
    { "Dogfish",           1,  6, PET_FORM_REVERT,        5 },
    { "Old Boot",          1, 20, PET_FORM_NONE,          1 },
};

const char *gFishSizeNames[FISH_SIZE_COUNT] = { "Small", "Medium", "Large", "Flawless" };

/* ------------------------------------------------------------------ */
/* Spells                                                              */
/* ------------------------------------------------------------------ */

const FateSpellDef gSpellDefs[SPELL_COUNT] = {
    { "Egg Storm",    "Exploding eggs blast every foe around you.",                SKILL_ATTACK_MAGIC,  10, 12, 150,  30 },
    { "Thunder Peck", "Lightning strikes the nearest foe for heavy damage.",        SKILL_ATTACK_MAGIC,  18, 18, 400,  20 },
    { "Honey Heal",   "Restores a large amount of health.",                         SKILL_DEFENSE_MAGIC, 10, 15, 120,  50 },
    { "Golden Aura",  "Halves the damage you take for 20 seconds.",                 SKILL_DEFENSE_MAGIC, 22, 25, 500, 300 },
    { "Turbo Talon",  "Run much faster for 20 seconds.",                            SKILL_DEFENSE_MAGIC, 16, 15, 300, 250 },
    { "Pet Frenzy",   "Your pet attacks three times as fast and heals for 15 s.",   SKILL_CHARM_MAGIC,   14, 20, 350, 300 },
    { "Mumbo's Fear", "Curses nearby foes so they wither away over 8 seconds.",     SKILL_CHARM_MAGIC,   26, 30, 700, 150 },
    { "Warp Home",    "Opens a portal back to Grove.",                              SKILL_DEFENSE_MAGIC, 12, 20, 250, 100 },
    { "Discern",      "Identifies every item you and your pet carry.",              SKILL_CHARM_MAGIC,    8, 10, 100,  20 },
};

/* ------------------------------------------------------------------ */
/* Pet forms                                                           */
/* ------------------------------------------------------------------ */

const FatePetFormDef gPetForms[PET_FORM_COUNT] = {
    { "Yellow Jinjo",  0,   0,   0, 0 },
    { "Orange Jinjo", 60,   0,   0, 0 },
    { "Blue Jinjo",    0,  80,   0, 0 },
    { "Green Jinjo",  10,   0,  60, 0 },
    { "Pink Jinjo",    0,  20,   0, 3 },
    { "Bigbutt",     100, 100, -20, 0 },
};

/* ------------------------------------------------------------------ */
/* Monster roster (marker id -> name, threat tier)                     */
/* ------------------------------------------------------------------ */

#define TOWN ROSTERF_TOWN
#define BRK  ROSTERF_BREAKABLE
#define VAN  0x04  /* keep vanilla hits-to-kill (minigames, wonderwing-only foes) */

const FateRosterEntry gRoster[] = {
    { 0x005, 1, 0,    "Grublin" },
    { 0x004, 2, 0,    "Ticker" },
    { 0x003, 5, 0,    "Bigbutt" },
    { 0x013, 2, 0,    "Snippet" },
    { 0x16B, 2, 0,    "Snippet" },
    { 0x0DD, 4, 0,    "Black Snippet" },
    { 0x0DE, 4, 0,    "Black Snippet" },
    { 0x015, 2, 0,    "Yum-Yum" },
    { 0x014, 8, 0,    "Snacker" },
    { 0x065, 3, 0,    "Shrapnel" },
    { 0x029, 3, 0,    "Grille Chompa" },
    { 0x1CF, 3, 0,    "Grille Chompa" },
    { 0x069, 2, 0,    "Chump" },
    { 0x173, 2, 0,    "Chump" },
    { 0x0C1, 2, 0,    "Flibbit" },
    { 0x0C5, 4, 0,    "Yellow Flibbit" },
    { 0x0C2, 2, 0,    "Buzzbomb" },
    { 0x0B2, 1, 0,    "Snowball" },
    { 0x0AD, 5, 0,    "Slappa" },
    { 0x253, 2, 0,    "Scarab" },
    { 0x219, 3, VAN,  "Mum-Mum" },
    { 0x218, 3, VAN,  "Limbo" },
    { 0x099, 3, VAN,  "Tee-Hee" },
    { 0x296, 3, VAN,  "Tee-Hee" },
    { 0x096, 5, 0,    "Ripper" },
    { 0x297, 6, 0,    "Giant Ripper" },
    { 0x127, 2, 0,    "Bat" },
    { 0x254, 3, VAN,  "Portrait Chompa" },
    { 0x1D1, 3, VAN,  "Portrait Chompa" },
    { 0x21A, 3, 0,    "Seaman Grublin" },
    { 0x0C9, 3, 0,    "Flotsam" },
    { 0x1B7, 2, VAN,  "Boom Box" },
    { 0x1E2, 3, 0,    "Grublin Hood" },
    { 0x1C5, 4, 0,    "Whipcrack" },
    { 0x1B2, 3, 0,    "Clucker" },
    { 0x1D0, 3, 0,    "Clucker" },
    { 0x1AE, 1, VAN,  "Zubba" },
    { 0x1EA, 2, 0,    "Red Gruntling" },
    { 0x295, 3, 0,    "Blue Gruntling" },
    { 0x1F1, 4, 0,    "Black Gruntling" },
    { 0x1E6, 1, TOWN, "Topper" },
    { 0x1E7, 1, TOWN, "Bawl" },
    { 0x1E8, 1, TOWN, "Colliwobble" },
    { 0x12A, 1, TOWN, "Topper" },
    { 0x129, 1, TOWN, "Bawl" },
    { 0x128, 1, TOWN, "Colliwobble" },
    { 0x050, 1, BRK,  "Beehive" },
};
const s32 gRosterCount = sizeof(gRoster) / sizeof(gRoster[0]);

/* ------------------------------------------------------------------ */
/* Champions, artifacts, fame                                          */
/* ------------------------------------------------------------------ */

const char *gChampionPrefixes[] = {
    "Grim", "Vile", "Dread", "Feral", "Cursed", "Rotten", "Savage", "Gloomy",
    "Wretched", "Mighty", "Ancient", "Furious", "Hexed", "Spiteful", "Ghastly", "Rabid",
};
const s32 gChampionPrefixCount = sizeof(gChampionPrefixes) / sizeof(gChampionPrefixes[0]);

const char *gChampionTitles[] = {
    "the Fierce", "the Cruel", "the Unyielding", "the Devourer", "Bonecrusher", "the Hungry",
    "of the Swamp", "Grunty's Pet", "the Sly", "the Wicked", "Skullsplitter", "the Undying",
};
const s32 gChampionTitleCount = sizeof(gChampionTitles) / sizeof(gChampionTitles[0]);

const char *gChampionModNames[CHAMP_COUNT] = { "", "Tough", "Brutal", "Regenerating", "Vampiric", "Armored" };

const char *gArtifactNames[] = {
    "Bottles' Lost Glasses",
    "Mumbo's Spare Skull",
    "Tooty's Hair Bow",
    "Brentilda's Diary",
    "Cheato's Torn Page",
    "Jinjo Elder's Crown",
    "Captain Blubber's Gold",
    "Gobi's Water Flask",
    "Boggy's Fish Supper",
    "Grunty's Spell Ingredient",
    "Klungo's Potion",
    "Dingpot's Ladle",
    "Loggo's Golden Plunger",
    "Tiptup's Choir Sheet",
    "Twinkly Box Key",
    "Ancient Honeycomb",
};
const s32 gArtifactNameCount = sizeof(gArtifactNames) / sizeof(gArtifactNames[0]);

const char *gFameTitles[] = {
    "Unknown",
    "Honey Forager",
    "Note Gatherer",
    "Grublin Bopper",
    "Jiggy Seeker",
    "Mumbo's Friend",
    "Hero of Spiral Mountain",
    "Breegull Rider",
    "Scourge of Gruntlings",
    "Witch Hunter",
    "Lair Delver",
    "Champion of Grove",
    "Legend of the Lair",
    "Bane of Gruntilda",
    "Bear of Legend",
};
const s32 gFameTitleCount = sizeof(gFameTitles) / sizeof(gFameTitles[0]);

const char *gQuestGivers[] = {
    "Bottles", "Tooty", "Mumbo Jumbo", "Brentilda", "Cheato", "Captain Blubber", "the Jinjo Elder", "Gobi",
};
const s32 gQuestGiverCount = sizeof(gQuestGivers) / sizeof(gQuestGivers[0]);
