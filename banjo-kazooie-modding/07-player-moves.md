# 7. Player Moves and Abilities

[← Models](06-models-textures-animation.md) · [Index](README.md) · [Next: Text and dialog →](08-text-and-dialog.md)

Banjo and Kazooie are driven by a state machine. Every move (walk, jump,
backflip, Beak Barge, swimming, each transformation's moves) is a **state**
(`enum bs_e`, "Banjo state") with four callbacks. Changing moves means
changing states, the transitions between them, or the physics constants they
use.

## 7.1 How the state machine works

| Piece | Where |
|---|---|
| State IDs | `enum bs_e` in `include/enums.h` (`BS_1_IDLE` … `BS_A5_WONDERWING_UNKA5`) |
| Callback storage | `bsState D_8037C700[166]` in `src/core2/bsList.c` (init/update/end/interrupt) |
| State table | `bsMethodsTable[]` in `src/core2/bsmethods.c`, copied into the list by `bsmethods_setMethods()` |
| Move implementations | `src/core2/bs/*.c` (`jump.c`, `bFlip.c`, `bBarge.c`, `swim.c`, `croc.c`, `bee*.c`, …) |
| Prototypes | `include/bs_funcs.h` |
| Switching | `bs_setState(next)`: 0 means "stay". `bs_getState()`, `bs_getPrevState()`, `bs_getNextState()`. |
| Input | `src/core2/bakey.c`: `bakey_pressed(BUTTON_x)` is true on the first frame only, `bakey_held()` counts frames held, `bakey_released()` counts frames since release (non-zero = button is up), `bakey_newlyReleased()` is the first frame up. `bastick_*` reads the analog stick. |
| Input + ability combos | `bainput_should_*()` in `src/core2/bainput.c` |
| Ability gates | `can_*()` in `src/core2/playerutils.c` |
| Physics | `baphysics_*` in `src/core2/ba/physics.c` |
| Animation | `baanim_getAnimCtrlPtr()` + `anctrl_*` |

Each table row:

```c
{BS_12_BFLIP, bsbflip_init, bsbflip_update, bsbflip_end, func_80296608},
//  state      on enter      every frame     on exit      external interrupts
```

The **interrupt** callback handles events pushed by the world through
`bs_checkInterrupt(BS_INTR_*)`: getting hurt, touching wading boots or turbo
trainers, landing on a surf object, and so on. Reuse the handler of a similar
move: `func_802B5350` for grounded moves, `func_80296608` for airborne or
attack moves, `func_80296590` for "ow" and transformation-locked moves.

A typical `update` collects a candidate next state and commits it at the end.
From `bsjump_update()` in `src/core2/bs/jump.c`:

```c
if(bainput_should_feathery_flap()) sp34 = BS_10_BFLAP;
if(bainput_should_peck())          sp34 = BS_11_BPECK;
if(bainput_should_beak_bust())     sp34 = BS_F_BBUSTER;
if(player_isStable()){ func_8029C5E8(); sp34 = BS_20_LANDING; }
if(velocity[1] < 0.0f && player_inWater()) sp34 = BS_4C_LANDING_IN_WATER;
bs_setState(sp34);
```

Later checks win, so the order of the `if`s sets priority.

## 7.2 Tuning existing moves (quick wins)

Most feel changes are a single constant:

| Move | File | Variable | Vanilla |
|---|---|---|---|
| Gravity (default) | `src/core2/ba/physics.c` | `baphysics_default_gravity` | `-2700.0f` |
| Terminal velocity | `src/core2/ba/physics.c` | `baphysics_default_terminal_velocity` | `-4000.0f` |
| Jump: initial vertical speed | `src/core2/bs/jump.c` | `D_80364CD0` | `710.0f` |
| Jump: gravity while A is held | `src/core2/bs/jump.c` | `D_80364CD4` | `-1350.0f` |
| Jump from water | `src/core2/bs/jump.c` | `D_80364CE4` | `610.0f` |
| Backflip height | `src/core2/bs/bFlip.c` | `D_80364A20` (vertical speed) | `920.0f` |
| Backflip gravity | `src/core2/bs/bFlip.c` | `D_80364A24` | `-1200.0f` |
| Backflip horizontal range | `src/core2/bs/bFlip.c` | `D_80364A2C` / `D_80364A30` | `80` / `200` |
| Run speed (max) | `src/core2/bs/walk.c` | `bsWalkMaxFastWalkVelocity` | `500.0f` |
| Walk speed thresholds | `src/core2/bs/walk.c` | `bsWalk*VelocityThreshold` | 80 / 150 / 225 |
| Talon Trot speed | `src/core2/bs/bTrot.c` | `gBsbTrotMaxWalkVelocity` | `700.0f` |
| Turbo Trainers speed | `src/core2/bs/bTrot.c` | `gBsbTrotMaxTurboVelocity` | `1000.0f` |
| Talon Trot jump | `src/core2/bs/bTrot.c` | `gBsbTrotInitialJumpVelocity`, `gBsbTrotGravity` | `693.5`, `-1200` |
| Swim speed | `src/core2/bs/swim.c` | `bsSwimHorzVelocityMax` | `300.0f` |
| Shock Spring Jump | `src/core2/bs/bShock.c` | `D_80364A70` / `D_80364A74` | `1250` / `-1200` |
| Feathery Flap | `src/core2/bs/bFlap.c` | `D_80364A10` | `280.0f` |

Some are declared `const f32`. You can still edit their initializers.

## 7.3 Unlocking or locking abilities

Abilities are bits in one `u32 learnedAbilities` (`src/core2/abilityprogress.c`),
saved with the file. `enum ability_e` (`include/core2/abilityprogress.h`) runs
from `ABILITY_0_BARGE` to `ABILITY_13_1ST_NOTEDOOR`, so **bits 0x14–0x1F are
free** for your own moves.

| Want | Do |
|---|---|
| Start with every move | Add `ability_setAllLearned(0xFFFFF);` at the end of `clearScoreStates()` (`src/core2/code_956B0.c`), which resets state for a fresh game |
| Learn a move from code | `ability_setLearned(ABILITY_9_FLIGHT, TRUE)` |
| Check a move | `ability_hasLearned()` or the `can_*()` helpers |
| Remove a requirement | Edit the `can_*()` helper, e.g. make `can_dive()` ignore `ABILITY_F_DIVE` |
| New learnable move | Add `ABILITY_14_MY_MOVE`, gate your input on `ability_hasLearned(ABILITY_14_MY_MOVE)`, and have Bottles (or anything else) call `ability_setLearned()` |

"Used" abilities (`usedAbilities`, `enum ability_used_e`) are a separate set
that drives the first-time tutorial dialogs (`ability_setUsedWithDialog()`).

## 7.4 Adding a new move: worked example (double jump)

The complete, compiling code is in
[`examples/bsdoublejump.c`](examples/bsdoublejump.c). The steps:

### 1. Pick a state ID

`D_8037C700` has 166 slots (`0x00`–`0xA5`), and these IDs are **unused** by
vanilla: `0x19`, `0x47`, `0x5C`, `0x5D`, `0x65`, `0x66`, `0x83`, `0x84`,
`0x90`. Name one in `enum bs_e`:

```c
    BS_18_FLY_KNOCKBACK,
    BS_19_DOUBLE_JUMP,              // new
    BS_1A_WONDERWING_ENTER = 0x1A,
```

To go beyond 166 states, enlarge `D_8037C700[166]` and the `0xa6` loop in
`bsList_clearAll()` (`src/core2/bsList.c`).

### 2. Write init / update / end

```c
void bsdoublejump_init(void) {
    AnimCtrl *aCtrl = baanim_getAnimCtrlPtr();
    sBsDoubleJumpUsed = TRUE;
    anctrl_reset(aCtrl);
    anctrl_setIndex(aCtrl, ASSET_4B_ANIM_BSBFLIP_ENTER);   // reuse the backflip spin
    anctrl_setDuration(aCtrl, 1.2f);
    anctrl_setPlaybackType(aCtrl, ANIMCTRL_ONCE);
    anctrl_start(aCtrl, "bsdoublejump.c", __LINE__);
    code_14420_setUpdateTypes(1, YAW_STATE_1_DEFAULT, 3, BA_PHYSICS_AIRBORN);
    baphysics_set_vertical_velocity(800.0f);
    baphysics_set_gravity(-1350.0f);
    func_8030E4E4(SFX_33_BANJO_AHOO);
}
```

`update` mirrors `bsjump_update`: air control (`func_802B6FA8()`), flap, beak
buster, landing, water. `end` restores gravity.

### 3. Register it

```c
// src/core2/bsmethods.c, bsMethodsTable[], before {0}
{BS_19_DOUBLE_JUMP, bsdoublejump_init, bsdoublejump_update, bsdoublejump_end, func_80296608},
```

and declare the three functions in `include/bs_funcs.h`.

### 4. Add the transition

In `bsjump_update()`, after the feathery-flap line:

```c
if (bsdoublejump_canStart())       // A pressed again, not used yet, still airborne
    sp34 = BS_19_DOUBLE_JUMP;
```

and reset the "used" flag in `bsjump_init()`.

### 5. Gate it (optional)

Make `bsdoublejump_canStart()` also require
`ability_hasLearned(ABILITY_14_DOUBLE_JUMP)` and teach it somewhere.

## 7.5 Other things you can change

| Goal | Where to look |
|---|---|
| Change which button does what | `src/core2/bainput.c` (`bainput_should_*`) and the transition blocks in each `bs/*.c` |
| Make a move available to a transformation | The transformation's state files: `bs/ant.c` (termite), `bs/pumpkin.c`, `bs/croc.c`, `bs/walrus.c`, `bs/bee*.c` |
| Fall damage | `src/core2/ba/falldamage.c` |
| Invulnerability frames | `src/core2/ba/iFrame.c` |
| Egg / feather costs | `func_80346C10(&next, -1, BS_x, ITEM_D_EGGS, 0)` calls in the transition code check and consume items. See `bs/crouch.c`. |
| Health, lives, air | [chapter 10](10-game-rules-and-tweaks.md) |
| Climbing | `src/core2/climb.c`, `bs/climb.c` |
| First-person camera | `src/core2/nc/ba/` |
| Transformation costs | `__transformation_getCost()` in `src/core2/ch/mumbo.c` |

## 7.6 Pitfalls

- **Every state must keep calling `bs_setState()` in `update`**, even with 0.
  Don't return early before the landing and water checks, or Banjo can get
  stuck in the air or underwater.
- Reset any gravity, terminal velocity or stick-zone changes in `end`, as
  `bsbflip_end()` does. The next state inherits the physics settings.
- Use `anctrl_start(aCtrl, "file.c", __LINE__)`. The strings are debug info,
  but keep the call.
- The player's model visibility tweaks (`modelAppendages_setKazooiesUpperHalfVisibility`)
  must be undone on exit, or Kazooie stays half-visible.
