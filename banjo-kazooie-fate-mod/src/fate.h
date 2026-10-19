#ifndef FATE_H
#define FATE_H

/*
 * Banjo-Kazooie: Fate Edition
 *
 * Turns Banjo-Kazooie: Recompiled into a WildTangent "Fate"-style action RPG:
 * levels, attributes and skills, fame, randomized loot with rarities, a pet
 * that fights and sells loot, fishing, quests from the townsfolk of Grove
 * (Spiral Mountain), shops, spells, champion monsters, a death bargain and
 * retirement with heirlooms.
 */

#include "modding.h"
#include "recomputils.h"
#include "recompconfig.h"
#include "game.h"

#ifdef FATE_DEV
#define FATE_TRACE(...) recomp_printf("[Fate trace] " __VA_ARGS__)
#else
#define FATE_TRACE(...) do { } while (0)
#endif

#define FATE_SAVE_MAGIC   0x46415445 /* 'FATE' */
#define FATE_SAVE_VERSION 1

/* ------------------------------------------------------------------ */
/* Character                                                           */
/* ------------------------------------------------------------------ */

enum fate_stat {
    STAT_STR,
    STAT_DEX,
    STAT_VIT,
    STAT_MAG,
    STAT_COUNT
};

/* Fate's weapon skills map onto Banjo and Kazooie's moves. */
enum fate_skill {
    SKILL_CLAW,           /* Claw Swipe (Fate: Sword) */
    SKILL_ROLL,           /* Forward Roll (Club) */
    SKILL_PECK,           /* Rat-a-tat Rap (Spear) */
    SKILL_BARGE,          /* Beak Barge (Axe) */
    SKILL_BUSTER,         /* Beak Buster and stomps (Hammer) */
    SKILL_BOMB,           /* Beak Bomb (Polearm) */
    SKILL_EGG,            /* Egg Firing (Bow) */
    SKILL_CRIT,           /* Critical Strike */
    SKILL_SHIELD,         /* Shield (backpack blocking) */
    SKILL_ATTACK_MAGIC,
    SKILL_DEFENSE_MAGIC,
    SKILL_CHARM_MAGIC,
    SKILL_COUNT
};

#define FATE_MAX_LEVEL      99
#define FATE_SKILL_MAX      20
#define STAT_POINTS_PER_LEVEL  5
#define SKILL_POINTS_PER_LEVEL 1
#define SKILL_POINTS_PER_FAME  4

/* ------------------------------------------------------------------ */
/* Items                                                               */
/* ------------------------------------------------------------------ */

enum fate_slot {
    SLOT_BANJO_WEAPON,    /* claws, clubs, hammers, axes for Banjo's paws */
    SLOT_KAZOOIE_WEAPON,  /* beak spikes, lances and egg slings for Kazooie */
    SLOT_HELM,
    SLOT_ARMOR,
    SLOT_SHIELD,          /* Banjo's backpack doubles as his shield */
    SLOT_BOOTS,
    SLOT_AMULET,
    SLOT_RING1,
    SLOT_RING2,
    SLOT_COUNT
};

enum fate_item_kind {
    KIND_NONE,
    KIND_BANJO_WEAPON,
    KIND_KAZOOIE_WEAPON,
    KIND_HELM,
    KIND_ARMOR,
    KIND_SHIELD,
    KIND_BOOTS,
    KIND_AMULET,
    KIND_RING,
    KIND_ARTIFACT,        /* quest items */
    KIND_COUNT
};
#define KINDBIT(k) (1u << (k))

enum fate_rarity {
    RARITY_NORMAL,
    RARITY_MAGIC,
    RARITY_UNIQUE,
    RARITY_ELITE,
    RARITY_LEGENDARY,
    RARITY_ARTIFACT,
    RARITY_COUNT
};

/* Item effects, used by affixes, uniques and gems. */
enum fate_effect {
    FX_NONE,
    FX_DMG_PCT,
    FX_DMG_FLAT,
    FX_STR,
    FX_DEX,
    FX_VIT,
    FX_MAG,
    FX_ALL_STATS,
    FX_HP,
    FX_MANA,
    FX_ARMOR_PCT,
    FX_ARMOR_FLAT,
    FX_CRIT,
    FX_BLOCK,
    FX_DODGE,
    FX_LIFE_STEAL,
    FX_GOLD_FIND,
    FX_MAGIC_FIND,
    FX_XP,
    FX_HP_REGEN,
    FX_MANA_REGEN,
    FX_EGG_DMG,
    FX_BEAK_DMG,
    FX_PAW_DMG,
    FX_PET_DMG,
    FX_PET_HP,
    FX_SPELL_PWR,
    FX_MOVE_SPEED,
    FX_ALL_SKILLS,
    FX_THORNS,
    FX_FIRE_DMG,
    FX_DMG_REDUCE,
    FX_COUNT
};

#define ITEMF_IDENTIFIED 0x01
#define ITEMF_HEIRLOOM   0x02
#define ITEMF_CURSED     0x04
#define ITEMF_SOCKETS_SHIFT 4
#define ITEMF_SOCKETS_MASK  0x30

typedef struct {
    u8 base;              /* index into gItemBases, 0 = empty */
    u8 rarity;
    u8 ilvl;              /* item level, scales affixes */
    u8 flags;
    u8 affix[3];          /* affix ids, 0 = none; for uniques affix[0] is the unique id */
    u8 roll[3];           /* 0-255 roll within the affix range */
    u8 gem[3];            /* socketed gem ids, 0 = empty */
    u8 enchant;           /* heirloom generation (each adds +25% to enchantments) */
} FateItem;

typedef struct {
    const char *name;
    u8 kind;
    u8 tier;              /* first depth it can drop at */
    u16 a;                /* weapons: min damage; armor: armor value */
    u16 b;                /* weapons: max damage */
    u16 price;
    u8 reqStr;
    u8 reqDex;
    u8 reqMag;
    u8 pad;
} FateItemBase;

typedef struct {
    const char *name;
    u8 isSuffix;
    u8 effect;
    u16 kindMask;
    s16 min;              /* magnitude at item level 1 */
    s16 max;
    u8 perLevel;          /* extra magnitude per 10 item levels, x10 */
    u8 minIlvl;
} FateAffix;

typedef struct {
    const char *name;
    u8 base;
    u8 minDepth;
    u8 effect[3];
    s16 value[3];
} FateUnique;

typedef struct {
    const char *name;
    u8 effect;
    s16 value;
    u16 price;
} FateGem;

#define INV_SIZE      30
#define PET_BAG_SIZE  20
#define PET_EQUIP_COUNT 2

/* ------------------------------------------------------------------ */
/* Consumables, fish, spells                                           */
/* ------------------------------------------------------------------ */

enum fate_consumable {
    CONS_HP_POTION,
    CONS_HP_POTION_GREATER,
    CONS_MANA_POTION,
    CONS_MANA_POTION_GREATER,
    CONS_TOWN_PORTAL,
    CONS_IDENTIFY,
    CONS_COUNT
};

enum fate_fish {
    FISH_GLOOP_GUPPY,
    FISH_BLUBBER_SNAPPER,
    FISH_CLANKER_CARP,
    FISH_SWAMP_SUCKER,
    FISH_FREEZEEZY_TROUT,
    FISH_SANDYBUTT_EEL,
    FISH_SPOOKY_PIKE,
    FISH_RUSTY_CATFISH,
    FISH_SEASONAL_KOI,
    FISH_BIGBUTT_BASS,
    FISH_DOGFISH,
    FISH_OLD_BOOT,
    FISH_COUNT
};

enum fate_fish_size {
    FISH_SIZE_SMALL,
    FISH_SIZE_MEDIUM,
    FISH_SIZE_LARGE,
    FISH_SIZE_FLAWLESS,
    FISH_SIZE_COUNT
};

/* Fish counts are tracked per type and size: fish[type * 4 + size]. */
#define FISH_SLOTS (FISH_COUNT * FISH_SIZE_COUNT)

typedef struct {
    const char *name;
    u8 minDepth;
    u8 weight;            /* catch weight at its depth */
    u8 petForm;           /* pet form when fed, 0xFF = none, 0xFE = revert */
    u8 basePrice;
} FateFishDef;

enum fate_spell {
    SPELL_EGG_STORM,      /* Attack: area blast around Banjo */
    SPELL_THUNDER_PECK,   /* Attack: heavy strike on the nearest foe */
    SPELL_HONEY_HEAL,     /* Defense: heal */
    SPELL_GOLDEN_AURA,    /* Defense: damage reduction */
    SPELL_TURBO_TALON,    /* Defense: haste */
    SPELL_PET_FRENZY,     /* Charm: pet attacks fast and heals */
    SPELL_MUMBOS_FEAR,    /* Charm: nearby foes take heavy damage over time */
    SPELL_WARP_HOME,      /* Utility: return to Grove */
    SPELL_DISCERN,        /* Utility: identify everything */
    SPELL_COUNT
};

typedef struct {
    const char *name;
    const char *desc;
    u8 school;            /* SKILL_ATTACK_MAGIC / DEFENSE / CHARM */
    u8 reqMag;
    u16 manaCost;
    u16 price;
    u16 cooldownTenths;
} FateSpellDef;

/* ------------------------------------------------------------------ */
/* Pet                                                                 */
/* ------------------------------------------------------------------ */

enum fate_pet_species {
    PET_SPECIES_JINJO,
    PET_SPECIES_BIGBUTT,
    PET_SPECIES_NONE,
    PET_SPECIES_COUNT
};

enum fate_pet_form {
    PET_FORM_YELLOW_JINJO,
    PET_FORM_ORANGE_JINJO,
    PET_FORM_BLUE_JINJO,
    PET_FORM_GREEN_JINJO,
    PET_FORM_PINK_JINJO,
    PET_FORM_BIGBUTT,
    PET_FORM_COUNT
};

#define PET_FORM_NONE   0xFF
#define PET_FORM_REVERT 0xFE

typedef struct {
    const char *name;
    s16 dmgPct;           /* bonus damage % */
    s16 hpPct;            /* bonus hp % */
    s16 speedPct;         /* bonus attack speed % */
    s16 healsOwner;       /* heals Banjo this much per attack */
} FatePetFormDef;

/* ------------------------------------------------------------------ */
/* Quests                                                              */
/* ------------------------------------------------------------------ */

enum fate_quest_type {
    QUEST_NONE,
    QUEST_SLAY,           /* slay N of a monster type in a world */
    QUEST_CHAMPION,       /* hunt a named champion in a world */
    QUEST_FETCH,          /* recover an artifact from a world */
    QUEST_NOTES,          /* collect N notes in a world */
    QUEST_JIGGIES,        /* collect N jiggies in a world */
    QUEST_TYPE_COUNT
};

enum fate_quest_state {
    QSTATE_ACTIVE,
    QSTATE_READY,         /* objective done, turn in at Grove */
};

typedef struct {
    u8 type;
    u8 giver;
    u8 level;             /* enum level_e */
    u8 state;
    u16 target;           /* marker id for slay/champion */
    u16 progress;
    u16 required;
    u16 seed;             /* champion / artifact name seed */
    u32 rewardGold;
    u32 rewardXp;
    u16 rewardFame;
    u8 rewardItem;
    u8 pad;
} FateQuest;

#define MAX_ACTIVE_QUESTS 3
#define QUEST_BOARD_SIZE  3

/* ------------------------------------------------------------------ */
/* Save data (one per game file, stored in its own save file)          */
/* ------------------------------------------------------------------ */

typedef struct {
    u32 magic;
    u16 version;
    u16 size;
    u32 checksum;

    /* character */
    u8  created;
    u8  level;
    u8  generation;       /* retirement lineage, 0 for the first hero */
    u8  nightmare;        /* post-game depth tier */
    u16 statPoints;
    u16 skillPoints;
    u16 stats[STAT_COUNT];
    u8  skills[SKILL_COUNT];
    u32 xp;
    u32 fame;
    u32 gold;
    u16 hp;
    u16 mana;
    u32 kills;
    u32 championKills;
    u16 deaths;
    u8  retired;
    u8  selectedSpell;
    u32 spellsKnown;
    u32 fameBought;       /* fame purchased from Tooty, raises her prices */

    /* stuff */
    u16 consumables[CONS_COUNT];
    u8  fish[FISH_SLOTS];
    u8  gems[16];
    u8  poleTier;         /* 0 = no fishing pole */
    u8  pad0[3];
    FateItem equip[SLOT_COUNT];
    FateItem inv[INV_SIZE];

    /* pet */
    u8  petSpecies;
    u8  petLevel;
    u8  petForm;          /* current form */
    u8  petFormPermanent; /* form granted by a flawless fish */
    u32 petXp;
    u16 petHp;
    u16 petFormSeconds;   /* remaining time of a temporary form */
    FateItem petEquip[PET_EQUIP_COUNT];
    FateItem petBag[PET_BAG_SIZE];

    /* quests */
    FateQuest quests[MAX_ACTIVE_QUESTS];
    FateQuest board[QUEST_BOARD_SIZE];
    u16 questsCompleted;
    u16 shopSeed;

    /* travel */
    u8  portalMap;        /* map the last Return to Grove left from */
    u8  portalExit;
    u8  droppedGoldMap;   /* map where gold was dropped on death */
    u8  pad1;
    u32 droppedGold;
    u16 droppedGoldSeconds;
    u16 pad2;

    u32 rngState;
} FateSave;

/* Global (cross-file) data: the heirloom handed down on retirement. */
typedef struct {
    u32 magic;
    u16 version;
    u16 size;
    u32 checksum;
    u8  generation;
    u8  hasHeirloom;
    u8  pad[2];
    FateItem heirloom;
    char ancestorTitle[24];
} FateGlobalSave;

/* ------------------------------------------------------------------ */
/* Runtime state                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    u8 init;
    u8 champion;          /* champion modifier, 0 = normal monster */
    u8 questChampion;     /* 1 + quest index if this is a quest target */
    u8 rosterIdx;
    s32 hp;
    s32 maxHp;
    f32 regenAccum;
    f32 fearTimer;
    u16 nameSeed;
    u8 killed;
    u8 pad;
} FateEnemy;

enum fate_champion_mod {
    CHAMP_NONE,
    CHAMP_TOUGH,          /* lots of extra health */
    CHAMP_BRUTAL,         /* hits much harder */
    CHAMP_REGEN,          /* regenerates */
    CHAMP_VAMPIRIC,       /* heals when it hurts Banjo */
    CHAMP_ARMORED,        /* takes reduced damage */
    CHAMP_COUNT
};

typedef struct {
    u16 marker;
    u8 tier;              /* rough threat, scales hp and xp */
    u8 flags;
    const char *name;
} FateRosterEntry;

#define ROSTERF_BREAKABLE 0x01 /* containers like beehives: loot only */
#define ROSTERF_TOWN      0x02 /* Spiral Mountain training foes */

/* Derived character numbers, recomputed whenever gear or stats change. */
typedef struct {
    s32 stat[STAT_COUNT];
    s32 skill[SKILL_COUNT];
    s32 maxHp;
    s32 maxMana;
    s32 armor;
    s32 dmgPct;
    s32 dmgFlat;
    s32 pawMin, pawMax;
    s32 beakMin, beakMax;
    s32 eggPct, beakPct, pawPct;
    s32 critChance;       /* x100 (percent * 100) */
    s32 critMult;         /* percent, 200 = double damage */
    s32 block;            /* percent */
    s32 dodge;            /* percent */
    s32 lifeSteal;        /* percent */
    s32 goldFind;
    s32 magicFind;
    s32 xpBonus;
    s32 hpRegen;          /* hp per 10 s */
    s32 manaRegen;
    s32 petDmgPct;
    s32 petHpPct;
    s32 spellPct;
    s32 moveSpeed;
    s32 thorns;
    s32 fireDmg;
    s32 dmgReduce;
} FateDerived;

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */

extern FateSave gFate;
extern FateGlobalSave gFateGlobal;
extern FateDerived gDerived;
extern s32 gFateLoadedSlot;
extern f32 gFateHp;
extern f32 gFateMana;
extern bool gFateMenuOpen;
extern f32 gFateGameTime;

/* ------------------------------------------------------------------ */
/* Data tables (data.c)                                                */
/* ------------------------------------------------------------------ */

extern const FateItemBase gItemBases[];
extern const s32 gItemBaseCount;
extern const FateAffix gAffixes[];
extern const s32 gAffixCount;
extern const FateUnique gUniques[];
extern const s32 gUniqueCount;
extern const FateGem gGems[];
extern const s32 gGemCount;
extern const FateFishDef gFishDefs[FISH_COUNT];
extern const char *gFishSizeNames[FISH_SIZE_COUNT];
extern const FateSpellDef gSpellDefs[SPELL_COUNT];
extern const FatePetFormDef gPetForms[PET_FORM_COUNT];
extern const FateRosterEntry gRoster[];
extern const s32 gRosterCount;
extern const char *gStatNames[STAT_COUNT];
extern const char *gStatShort[STAT_COUNT];
extern const char *gSkillNames[SKILL_COUNT];
extern const char *gSlotNames[SLOT_COUNT];
extern const char *gRarityNames[RARITY_COUNT];
extern const char *gConsumableNames[CONS_COUNT];
extern const u16 gConsumablePrices[CONS_COUNT];
extern const char *gChampionPrefixes[];
extern const s32 gChampionPrefixCount;
extern const char *gChampionTitles[];
extern const s32 gChampionTitleCount;
extern const char *gChampionModNames[CHAMP_COUNT];
extern const char *gArtifactNames[];
extern const s32 gArtifactNameCount;
extern const char *gFameTitles[];
extern const s32 gFameTitleCount;
extern const char *gQuestGivers[];
extern const s32 gQuestGiverCount;
extern const char *gEffectFormats[FX_COUNT];

/* ------------------------------------------------------------------ */
/* util.c                                                              */
/* ------------------------------------------------------------------ */

void *memset(void *dst, int value, unsigned int size);
u32 fate_rand(void);
s32 fate_rand_range(s32 lo, s32 hi); /* inclusive */
f32 fate_randf(void);
bool fate_chance(s32 percentTimes100);
void fate_seed(u32 seed);
s32 fate_strlen(const char *s);
char *fate_strcpy(char *dst, const char *src);
char *fate_strcat(char *dst, const char *src);
char *fate_strcat_int(char *dst, s32 value);
char *fate_strcat_signed(char *dst, s32 value);
s32 fate_clampi(s32 v, s32 lo, s32 hi);
f32 fate_clampf(f32 v, f32 lo, f32 hi);
s32 fate_mini(s32 a, s32 b);
s32 fate_maxi(s32 a, s32 b);
u32 fate_checksum(const void *data, u32 size);
f32 fate_dist2d(f32 a[3], f32 b[3]);
f32 fate_dist3d(f32 a[3], f32 b[3]);
void fate_play_sfx(enum sfx_e sfx);
void fate_play_sfx_pitch(enum sfx_e sfx, f32 pitch, s32 volume);
void fate_play_jingle(enum comusic_e music);

/* ------------------------------------------------------------------ */
/* character.c                                                         */
/* ------------------------------------------------------------------ */

void fate_new_character(void);
void fate_recalc(void);
u32 fate_xp_for_level(s32 level);        /* total xp needed to reach level */
void fate_gain_xp(s32 amount, bool showMessage);
void fate_gain_fame(s32 amount);
s32 fate_fame_level(void);
u32 fate_fame_for_level(s32 fameLevel);
const char *fate_fame_title(void);
void fate_add_gold(s32 amount);
bool fate_spend_gold(s32 amount);
bool fate_raise_stat(s32 stat);
bool fate_raise_skill(s32 skill);
s32 fate_depth(void);
s32 fate_depth_for_level(enum level_e level);
const char *fate_level_name(enum level_e level);
bool fate_in_town(void);
bool fate_gameplay_active(void);
void fate_heal_full(void);
void fate_update_regen(f32 dt);

/* ------------------------------------------------------------------ */
/* items.c                                                             */
/* ------------------------------------------------------------------ */

bool fate_item_empty(const FateItem *item);
void fate_item_clear(FateItem *item);
const FateItemBase *fate_item_base(const FateItem *item);
bool fate_item_identified(const FateItem *item);
s32 fate_item_sockets(const FateItem *item);
void fate_item_name(const FateItem *item, char *out, s32 cap);
void fate_item_describe(const FateItem *item, char *out, s32 cap);
s32 fate_item_effect_total(const FateItem *item, s32 effect);
s32 fate_item_value(const FateItem *item);
s32 fate_item_sell_price(const FateItem *item);
s32 fate_item_buy_price(const FateItem *item);
s32 fate_item_fame_req(const FateItem *item);
bool fate_item_can_equip(const FateItem *item, char *reason, s32 cap);
s32 fate_slot_for_kind(s32 kind, s32 preferred);
bool fate_item_generate(FateItem *out, s32 depth, s32 forceRarity, s32 magicFind);
void fate_item_generate_base(FateItem *out, s32 base, s32 rarity, s32 ilvl);
bool fate_inventory_add(const FateItem *item);
bool fate_pet_bag_add(const FateItem *item);
s32 fate_inventory_free_slots(void);
bool fate_equip_from_inventory(s32 invIndex);
bool fate_unequip(s32 slot);
void fate_identify(FateItem *item);
s32 fate_identify_all(void);
u32 fate_rarity_color(s32 rarity);

/* ------------------------------------------------------------------ */
/* combat.c                                                            */
/* ------------------------------------------------------------------ */

void fate_combat_init(void);
void fate_combat_on_map_load(void);
FateEnemy *fate_enemy_get(ActorMarker *marker);
const FateRosterEntry *fate_roster_find(s32 markerId);
void fate_enemy_name(ActorMarker *marker, char *out, s32 cap);
s32 fate_damage_enemy(Actor *actor, s32 damage, bool fromPlayer, bool canCrit, s32 *outCrit);
void fate_combat_frame(f32 dt);
void fate_on_enemy_killed(Actor *actor, FateEnemy *enemy);
s32 fate_monster_damage_scale(void);
void fate_find_enemies_near(f32 pos[3], f32 radius, Actor **out, s32 *count, s32 max);
Actor *fate_find_nearest_enemy(f32 pos[3], f32 radius);
void fate_drop_loot(f32 pos[3], s32 depth, bool champion);

extern Actor *gFateTargetActor;
extern f32 gFateTargetTimer;
extern s32 gFateLastDamage;
extern bool gFateLastCrit;

/* ------------------------------------------------------------------ */
/* health.c                                                            */
/* ------------------------------------------------------------------ */

void fate_health_sync_to_bk(void);
void fate_health_frame(f32 dt);
void fate_drink_health_potion(void);
void fate_drink_mana_potion(void);
void fate_bargain_resolve(s32 choice);
extern bool gFateBargainPending;
extern bool gFateBargainOpen;

/* ------------------------------------------------------------------ */
/* pet.c                                                               */
/* ------------------------------------------------------------------ */

void fate_pet_register_actors(void);
void fate_pet_frame(f32 dt);
void fate_pet_on_map_load(void);
s32 fate_pet_max_hp(void);
s32 fate_pet_damage(void);
void fate_pet_gain_xp(s32 amount);
bool fate_pet_send_to_town(void);
bool fate_pet_feed_fish(s32 fishType, s32 size);
bool fate_pet_present(void);
const char *fate_pet_status(void);
const char *fate_pet_name(void);
f32 fate_pet_town_seconds(void);
void fate_pet_heal_full(void);
extern f32 gPetFrenzyTimer;

/* ------------------------------------------------------------------ */
/* fishing.c                                                           */
/* ------------------------------------------------------------------ */

bool fate_can_fish(void);
void fate_fishing_start(void);
void fate_fishing_frame(f32 dt);
bool fate_fishing_active(void);
const char *fate_fishing_prompt(void);
s32 fate_fish_price(s32 type, s32 size);
void fate_fish_name(s32 type, s32 size, char *out, s32 cap);

/* ------------------------------------------------------------------ */
/* quests.c                                                            */
/* ------------------------------------------------------------------ */

void fate_quests_refresh_board(void);
bool fate_quest_accept(s32 boardIndex);
bool fate_quest_turn_in(s32 index, bool keepArtifact);
void fate_quest_abandon(s32 index);
void fate_quest_describe(const FateQuest *q, char *out, s32 cap);
void fate_quest_title(const FateQuest *q, char *out, s32 cap);
void fate_quest_on_kill(Actor *actor, FateEnemy *enemy);
void fate_quest_on_note(enum level_e level);
void fate_quest_on_jiggy(enum level_e level);
void fate_quest_on_level_enter(enum level_e level);
s32 fate_quest_champion_for(enum level_e level, s32 markerId);
void fate_champion_name(u16 seed, s32 markerId, char *out, s32 cap);

/* ------------------------------------------------------------------ */
/* town.c                                                              */
/* ------------------------------------------------------------------ */

#define SHOP_STOCK 8
extern FateItem gShopStock[SHOP_STOCK];
void fate_shop_restock(void);
bool fate_shop_buy_item(s32 index);
bool fate_shop_buy_consumable(s32 cons);
bool fate_shop_buy_spell(s32 spell);
bool fate_shop_buy_pole(void);
bool fate_shop_buy_fish(s32 type);
bool fate_shop_sell_inventory(s32 invIndex);
s32 fate_shop_sell_all_fish(void);
s32 fate_shop_sell_junk(void);
s32 fate_identify_cost(const FateItem *item);
bool fate_shop_identify(s32 invIndex);
s32 fate_enchant_cost(const FateItem *item);
const char *fate_enchant(FateItem *item);
s32 fate_socket_cost(const FateItem *item);
const char *fate_add_socket(FateItem *item);
s32 fate_fame_cost(void);
bool fate_buy_fame(void);
void fate_healer(void);
bool fate_socket_gem(FateItem *item, s32 gem);
s32 fate_pole_price(void);

/* ------------------------------------------------------------------ */
/* spells.c                                                            */
/* ------------------------------------------------------------------ */

bool fate_cast_spell(s32 spell);
void fate_spells_frame(f32 dt);
bool fate_spell_known(s32 spell);
s32 fate_spell_mana_cost(s32 spell);
f32 fate_spell_cooldown(s32 spell);
void fate_cycle_spell(void);
extern f32 gAuraTimer;
extern f32 gHasteTimer;

/* ------------------------------------------------------------------ */
/* travel.c                                                            */
/* ------------------------------------------------------------------ */

bool fate_can_warp(void);
bool fate_warp_to_town(void);
bool fate_warp_to_portal(void);
void fate_travel_on_map_load(enum map_e map, s32 exit);

/* ------------------------------------------------------------------ */
/* save.c                                                              */
/* ------------------------------------------------------------------ */

void fate_save_init(void);
void fate_save_slot(s32 slot);
void fate_load_slot(s32 slot);
void fate_clear_slot(s32 slot);
void fate_save_global(void);
void fate_load_global(void);
void fate_retire(s32 equipSlot);
bool fate_can_retire(void);

/* ------------------------------------------------------------------ */
/* messages (ui_hud.c)                                                 */
/* ------------------------------------------------------------------ */

#define MSG_COLOR_DEFAULT 0xE8E0D0FF
#define MSG_COLOR_GOLD    0xF2C744FF
#define MSG_COLOR_XP      0x9EE06BFF
#define MSG_COLOR_FAME    0xE58CF2FF
#define MSG_COLOR_BAD     0xFF6B5BFF
#define MSG_COLOR_PET     0x7FD8FFFF
#define MSG_COLOR_QUEST   0xFFB347FF

void fate_message(const char *text, u32 color);
void fate_banner(const char *text, u32 color);
void fate_hud_init(void);
void fate_hud_frame(void);
void fate_hud_mark_dirty(void);

/* ------------------------------------------------------------------ */
/* menus (ui_menu.c)                                                   */
/* ------------------------------------------------------------------ */

enum fate_menu_tab {
    TAB_CHARACTER,
    TAB_INVENTORY,
    TAB_PET,
    TAB_SPELLS,
    TAB_QUESTS,
    TAB_GROVE,
    TAB_COUNT
};

void fate_menu_init(void);
void fate_menu_open(s32 tab);
void fate_menu_close(void);
void fate_menu_frame(void);
void fate_bargain_open(void);
void fate_menu_request_rebuild(void);

#endif
