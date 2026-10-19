/*
 * bsdoublejump.c - example new player move for the Banjo-Kazooie decomp
 *
 * Adds a double jump: press A again while in a normal jump. From the double
 * jump you can still Feathery Flap, Beak Buster, etc.
 *
 * Wiring (see chapter 7 of the guide):
 *   1. include/enums.h, enum bs_e: name the unused slot 0x19
 *          BS_18_FLY_KNOCKBACK,
 *          BS_19_DOUBLE_JUMP,          // new (0x19 was unused)
 *          BS_1A_WONDERWING_ENTER = 0x1A,
 *   2. Put this code in src/core2/bs/doublejump.inc.c and #include it at the
 *      end of src/core2/bs/jump.c (or list a new .c in the yaml).
 *   3. src/core2/bsmethods.c, bsMethodsTable[] (before the {0} terminator):
 *          {BS_19_DOUBLE_JUMP, bsdoublejump_init, bsdoublejump_update, bsdoublejump_end, func_80296608},
 *      func_80296608 is the interrupt handler the other airborne moves use.
 *   4. src/core2/bs/jump.c: near the top add the prototypes
 *          void bsdoublejump_resetUsed(void);
 *          bool bsdoublejump_canStart(void);
 *      in bsjump_init() add      bsdoublejump_resetUsed();
 *      in bsjump_update(), after the feathery flap check, add
 *          if (bsdoublejump_canStart())
 *              sp34 = BS_19_DOUBLE_JUMP;
 *      bsMethodsTable references the bsdoublejump_* functions, so declare
 *      them in include/bs_funcs.h too (that's where the other moves are).
 *
 * This file compiles with the decomp's IDO flags against commit e8d31fa5.
 */
#include <ultra64.h>
#include "functions.h"
#include "variables.h"

#include "core2/ba/anim.h"
#include "core2/yaw.h"
#include "core2/ba/physics.h"

#ifndef BS_19_DOUBLE_JUMP
#define BS_19_DOUBLE_JUMP 0x19
#endif

/* Defined in core2 but not declared in a header */
extern void func_802B6FA8(void);            /* src/core2/bs/walk.c: stick -> target horizontal speed */
extern void func_8029C5E8(void);            /* src/core2/code_14420.c: used by moves when landing */
extern void code_14420_setUpdateTypes(enum baanim_update_type_e, enum yaw_state_e, s32, BaPhysicsType);
extern bool player_isStable(void);          /* src/core2/playerutils.c */
extern bool player_isFallTumbling(void);    /* src/core2/playerutils.c */
extern int  bakey_released(s32 button);     /* src/core2/bakey.c */
extern int  bainput_should_feathery_flap(void); /* src/core2/bainput.c */
extern int  bainput_should_beak_bust(void);     /* src/core2/bainput.c */

#define DOUBLE_JUMP_VELOCITY   800.0f    /* normal jump is 710 (D_80364CD0 in jump.c) */
#define DOUBLE_JUMP_GRAVITY  -1350.0f    /* same as the held-A jump gravity */

static u8 sBsDoubleJumpUsed;

void bsdoublejump_resetUsed(void) {
    sBsDoubleJumpUsed = FALSE;
}

/* Called from bsjump_update(): second A press while still airborne */
bool bsdoublejump_canStart(void) {
    return bakey_pressed(BUTTON_A) && !sBsDoubleJumpUsed && !player_isStable();
}

void bsdoublejump_init(void) {
    AnimCtrl *aCtrl = baanim_getAnimCtrlPtr();

    sBsDoubleJumpUsed = TRUE;

    /* Reuse the backflip's spin animation */
    anctrl_reset(aCtrl);
    anctrl_setIndex(aCtrl, ASSET_4B_ANIM_BSBFLIP_ENTER);
    anctrl_setDuration(aCtrl, 1.2f);
    anctrl_setPlaybackType(aCtrl, ANIMCTRL_ONCE);
    anctrl_start(aCtrl, "bsdoublejump.c", __LINE__);

    /* Same update types bsjump_init() uses for an airborne state */
    code_14420_setUpdateTypes(1, YAW_STATE_1_DEFAULT, 3, BA_PHYSICS_AIRBORN);
    baphysics_set_vertical_velocity(DOUBLE_JUMP_VELOCITY);
    baphysics_set_gravity(DOUBLE_JUMP_GRAVITY);
    func_8030E4E4(SFX_33_BANJO_AHOO);
}

void bsdoublejump_update(void) {
    enum bs_e next_state = 0;
    f32 velocity[3];

    func_802B6FA8();    /* keep air control like a normal jump */
    baphysics_get_velocity(velocity);

    if (bakey_released(BUTTON_A) && velocity[1] > 0.0f) {
        baphysics_reset_gravity();      /* short hop if A is let go early */
    }

    if (player_isFallTumbling())
        next_state = BS_3D_FALL_TUMBLING;

    if (bainput_should_feathery_flap())
        next_state = BS_10_BFLAP;

    if (bainput_should_beak_bust())
        next_state = BS_F_BBUSTER;

    if (player_isStable()) {
        func_8029C5E8();
        next_state = BS_20_LANDING;
    }

    if (velocity[1] < 0.0f && player_inWater())
        next_state = BS_4C_LANDING_IN_WATER;

    bs_setState(next_state);    /* 0 = stay in this state */
}

void bsdoublejump_end(void) {
    baphysics_reset_gravity();
}
