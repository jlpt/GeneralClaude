; ---------------------------------------------------------------------------
;  retro-llm for the SNES: the same 2-layer int8 transformer as the NES build,
;  running on the 65816 with the PPU's signed 16x8 multiplier ($211B/$211C).
;
;  LoROM + FastROM, 32 KB. All state lives in the first 8 KB of WRAM.
;  Bit-exact with train/ref.py.
; ---------------------------------------------------------------------------
.p816
.smart -

.include "gen/model.inc"

.macro A8
        sep #$20
        .a8
.endmacro
.macro A16
        rep #$20
        .a16
.endmacro

D       = 32
F       = 64
NH      = 4
HD      = 8
CTX     = 32
NV      = 36
LEFT    = 2
COLS    = 28

TOKLOG  = $7E2000               ; long addresses, read by the test harness
MAGIC   = $7E3000
SEED    = $7E3004
NTOK    = $7E3006
TEMPLOG = $7E3008

BTN_A     = $0080
BTN_START = $1000
BTN_UP    = $0800
BTN_DOWN  = $0400

; ---------------------------------------------------------------------------
.segment "ZEROPAGE"
mv_in:    .res 2
mv_w:     .res 2
mv_nin:   .res 2
mv_nout:  .res 2
nout2:    .res 2
mv_stride:.res 2
mv_i:     .res 2
mv_nz:    .res 2
mv_cm:    .res 2
mv_ch:    .res 2
sh_cnt:   .res 2
rnd:      .res 2
t0:       .res 2
t1:       .res 2
t2:       .res 2
t3:       .res 2
ptr:      .res 2
layer:    .res 2
lbase:    .res 2
kvoff:    .res 2
head:     .res 2
slot:     .res 2
nvalid:   .res 2
nv2:      .res 2
tok:      .res 2
mval:     .res 2
esum:     .res 2
rlo:      .res 2
quot:     .res 2
mask:     .res 2
rng:      .res 2
frame:    .res 2
buttons:  .res 2
prevbtn:  .res 2
pressed:  .res 2
temp_idx: .res 2
ppu_cmd:  .res 2                ; 0 idle, 1 put tile, 2 clear BG1 row, 3 status
ppu_addr: .res 2
ppu_data: .res 2
scroll_y: .res 2
cur_row:  .res 2
cur_col:  .res 2
nmi_on:   .res 2
logpos:   .res 2

.segment "BSS"
KC:     .res 2*1024             ; K cache [layer][channel][slot]
VC:     .res 2*1024             ; V cache [layer][slot][channel]
XV:     .res D
QV:     .res D
AT:     .res D
HV:     .res F
TV:     .res F
ACCL:   .res 2*F                ; 32-bit accumulators (lo words / hi words)
ACCH:   .res 2*F
SC:     .res 2*CTX              ; scores (16-bit signed)
EW:     .res 2*CTX              ; exp weights (16-bit)
PV:     .res CTX                ; probabilities 0..127
EV:     .res NV+1
LG:     .res NV

; ---------------------------------------------------------------------------
.segment "CODE"

reset:
        sei
        clc
        xce                     ; native mode
        jml $800000 + fast_reset ; run from the FastROM mirror
fast_reset:
        rep #$38
        .a16
        .i16
        ldx #$1FFF
        txs
        lda #$0000
        tcd
        A8
        lda #$80
        pha
        plb                     ; DB = $80
        lda #$01
        sta $420D               ; FastROM
        lda #$80
        sta $2100               ; forced blank
        stz $4200
        ldx #$2101
@pr:    stz a:$0000,x
        inx
        cpx #$2134
        bne @pr
        A16
        ldx #0
@cr:    stz a:$0000,x
        inx
        inx
        cpx #$1F00
        bne @cr
        A8
        ; ---- PPU setup
        lda #$80
        sta $2115               ; VRAM increment after high byte
        stz $2105               ; BG mode 0
        stz $2107               ; BG1 map @ word $0000, 32x32
        lda #$08
        sta $2108               ; BG2 map @ word $0800
        lda #$11
        sta $210B               ; BG1/BG2 tiles @ word $1000
        lda #$03
        sta $212C               ; main screen: BG1 + BG2
        stz $210D
        stz $210D
        stz $210F
        stz $210F
        lda #$FF
        sta $210E
        stz $210E               ; BG1 (status) VOFS = -1, never touched again:
                                ; $210D/$210E share the mode-7 write latch with
                                ; the multiplier ($211B), so the NMI must not
                                ; write them. Text scrolls on BG2 instead.
        ; palette
        stz $2121
        lda #$42
        sta $2122
        lda #$20
        sta $2122               ; 0: dark blue
        lda #$9F
        sta $2122
        lda #$23
        sta $2122               ; 1: yellow (BG1 status)
        lda #33
        sta $2121
        lda #$FF
        sta $2122
        lda #$7F
        sta $2122               ; 33: white (BG2 text)
        ; font -> VRAM word $1000 + tile*8 (tile = ASCII)
        A16
        lda #$1000 + $20*8
        sta $2116
        ldx #0
@fn:    lda FONT,x
        and #$00FF
        sta $2118
        inx
        cpx #64*8
        bne @fn
        jsr clear_text
        ; BG1 map (status layer): spaces
        lda #$0000
        sta $2116
        ldx #1024
        lda #$0020
@b2:    sta $2118
        dex
        bne @b2
        jsr draw_title
        lda #2
        sta temp_idx
        lda #$FF
        sta scroll_y
        jsr status_now
        A8
        lda scroll_y
        sta $2110
        stz $2110
        lda #$81
        sta $4200               ; NMI + auto joypad
        A16
        lda #1
        sta nmi_on
        jsr wait_frame
        A8
        lda #$0F
        sta $2100               ; screen on
        A16

@title: jsr wait_frame
        jsr read_pressed
        and #BTN_START
        beq @title
        lda frame
        bne :+
        inc a
:       sta rng
        jsr start_session

main_loop:
        .a16
        jsr read_pressed
        sta pressed
        and #BTN_UP
        beq @noup
        lda temp_idx
        cmp #3
        beq @noup
        inc temp_idx
        jsr status_cmd
@noup:  lda pressed
        and #BTN_DOWN
        beq @nodn
        lda temp_idx
        beq @nodn
        dec temp_idx
        jsr status_cmd
@nodn:  lda pressed
        and #BTN_A
        beq @noa
        jsr start_session
@noa:   jsr forward
        jsr sample
        sta tok
        jsr log_token
        lda tok
        jsr putc
        lda slot
        inc a
        and #CTX-1
        sta slot
        lda nvalid
        cmp #CTX
        beq :+
        inc nvalid
:       jmp main_loop

; ---------------------------------------------------------------------------
start_session:
        .a16
        stz slot
        stz logpos
        lda #1
        sta nvalid
        stz tok
        lda #0
        sta f:NTOK
        lda rng
        sta f:SEED
        lda temp_idx
        sta f:TEMPLOG
        lda #('L' | ('L' << 8))
        sta f:MAGIC
        lda #('M' | ('!' << 8))
        sta f:MAGIC+2
        jsr wait_frame
        stz nmi_on
        A8
        lda #$80
        sta $2100
        A16
        jsr clear_text
        lda #26
        sta cur_row
        stz cur_col
        lda #$FF
        sta scroll_y
        lda #1
        sta nmi_on
        jsr wait_frame
        A8
        lda #$0F
        sta $2100
        A16
        rts

log_token:
        .a16
        lda logpos
        and #$0FFF
        tax
        A8
        lda tok
        sta f:TOKLOG,x
        A16
        inc logpos
        lda f:NTOK
        inc a
        sta f:NTOK
        rts

; ---------------------------------------------------------------------------
;  forward(tok) -> LG
; ---------------------------------------------------------------------------
forward:
        .a16
        lda tok
        and #$00FF
        asl a
        asl a
        asl a
        asl a
        asl a
        tax
        ldy #0
        A8
@emb:   lda EMB,x
        sta XV,y
        inx
        iny
        cpy #D
        bne @emb
        A16
        stz layer
@layer: lda layer
        xba
        asl a
        asl a
        sta kvoff               ; layer * $400
        asl a
        asl a
        asl a
        clc
        adc #LAYERS
        sta lbase               ; layer * $2000

        ; q
        lda #XV
        ldx #$0000
        jsr mv_dd
        ldx #0
        jsr shift_of
        jsr requant
        ldx #D-2
@cq:    lda TV,x
        sta QV,x
        dex
        dex
        bpl @cq
        ; k -> KC[layer][c][slot]
        lda #XV
        ldx #$0400
        jsr mv_dd
        ldx #1
        jsr shift_of
        jsr requant
        lda #KC
        clc
        adc kvoff
        adc slot
        sta ptr
        ldx #0
        ldy #0
@ck:    A8
        lda TV,x
        sta (ptr),y
        A16
        tya
        clc
        adc #CTX
        tay
        inx
        cpx #D
        bne @ck
        ; v -> VC[layer][slot][c]
        lda #XV
        ldx #$0800
        jsr mv_dd
        ldx #2
        jsr shift_of
        jsr requant
        lda slot
        asl a
        asl a
        asl a
        asl a
        asl a
        clc
        adc #VC
        adc kvoff
        sta ptr
        ldy #D-2
@cv:    lda TV,y
        sta (ptr),y
        dey
        dey
        bpl @cv

        stz head
@head:  jsr attend_head
        inc head
        lda head
        cmp #NH
        bne @head

        ; x += att Wo
        lda #AT
        ldx #$0C00
        jsr mv_dd
        ldx #3
        jsr shift_of
        jsr requant
        jsr add_tv_to_x
        ; h = relu(x W1)
        lda #XV
        sta mv_in
        lda #D
        sta mv_nin
        lda #F
        sta mv_nout
        sta mv_stride
        lda lbase
        clc
        adc #$1000
        sta mv_w
        jsr matvec
        ldx #4
        jsr shift_of
        jsr requant
        ldx #0
        A8
@relu:  lda TV,x
        bpl :+
        lda #0
:       sta HV,x
        inx
        cpx #F
        bne @relu
        A16
        ; x += h W2
        lda #HV
        sta mv_in
        lda #F
        sta mv_nin
        lda #D
        sta mv_nout
        sta mv_stride
        lda lbase
        clc
        adc #$1800
        sta mv_w
        jsr matvec
        ldx #5
        jsr shift_of
        jsr requant
        jsr add_tv_to_x

        inc layer
        lda layer
        cmp #2
        beq :+
        jmp @layer
:       ; logits
        lda #XV
        sta mv_in
        lda #D
        sta mv_nin
        lda #NV
        sta mv_nout
        sta mv_stride
        lda #WOUT
        sta mv_w
        jsr matvec
        lda #SHIFT_OUT
        jsr requant
        ldx #0
        A8
@lg:    lda TV,x
        sta LG,x
        inx
        cpx #NV
        bne @lg
        A16
        rts

; A = shift of matrix X in this layer
shift_of:
        .a16
        lda layer
        beq :+
        txa
        clc
        adc #6
        tax
:       lda SHIFTS,x
        and #$00FF
        rts

; 32x32 matvec: in = A, weights = lbase + X
mv_dd:
        .a16
        sta mv_in
        txa
        clc
        adc lbase
        sta mv_w
        lda #D
        sta mv_nin
        sta mv_nout
        sta mv_stride
        jmp matvec

add_tv_to_x:
        .a16
        ldx #0
        A8
@l:     lda XV,x
        clc
        adc TV,x
        bvc @ok
        lda XV,x
        bmi @neg
        lda #127
        bra @st
@neg:   lda #$81
        bra @st
@ok:    cmp #$80
        bne @st
        lda #$81
@st:    sta XV,x
        inx
        cpx #D
        bne @l
        A16
        rts

; ---------------------------------------------------------------------------
attend_head:
        .a16
        lda head
        asl a
        asl a
        asl a
        clc
        adc #QV
        sta mv_in
        lda #HD
        sta mv_nin
        lda nvalid
        sta mv_nout
        asl a
        sta nv2
        lda #CTX
        sta mv_stride
        lda head
        xba
        clc
        adc kvoff
        adc #KC
        sta mv_w
        jsr matvec
        lda #SCORE_SHIFT
        jsr acc_shift

        ; scores = acc - ALIBI[head][(slot - j) & 31]
        lda head
        asl a
        asl a
        asl a
        asl a
        asl a
        sta t2
        ldy #0
@sc:    tya
        eor #$FFFF
        sec
        adc slot
        and #CTX-1
        clc
        adc t2
        tax
        lda ALIBI,x
        and #$00FF
        sta t3
        tya
        asl a
        tax
        lda ACCL,x
        sec
        sbc t3
        sta SC,x
        iny
        cpy nvalid
        bne @sc

        ; max (signed)
        lda SC
        sta mval
        ldx #2
        cpx nv2
        beq @maxd
@mx:    lda SC,x
        sec
        sbc mval
        bvc :+
        eor #$8000
:       bmi @nomx
        lda SC,x
        sta mval
@nomx:  inx
        inx
        cpx nv2
        bne @mx
@maxd:
        ; e = EXP_ATT[min(m - s, 255)], esum
        stz esum
        ldx #0
@ex:    lda mval
        sec
        sbc SC,x
        cmp #256
        bcc :+
        lda #255
:       phx
        tax
        lda EXP_ATT,x
        and #$00FF
        plx
        sta EW,x
        clc
        adc esum
        sta esum
        inx
        inx
        cpx nv2
        bne @ex

        ; p = min(127, (e << 7) / esum)
        ldx #0
@pr:    lda EW,x
        beq @pz
        sta rlo
        stz quot
        ldy #7
@dv:    asl rlo
        lda rlo
        sec
        sbc esum
        bcc @dn
        sta rlo
@dn:    rol quot
        dey
        bne @dv
        lda quot
@pz:    phx
        pha
        txa
        lsr a
        tax
        pla
        A8
        sta PV,x
        A16
        plx
        inx
        inx
        cpx nv2
        bne @pr

        ; o = sum p * V >> 7
        lda #PV
        sta mv_in
        lda nvalid
        sta mv_nin
        lda #HD
        sta mv_nout
        lda #CTX
        sta mv_stride
        lda head
        asl a
        asl a
        asl a
        clc
        adc kvoff
        adc #VC
        sta mv_w
        jsr matvec
        lda #7
        jsr requant
        lda head
        asl a
        asl a
        asl a
        tay
        ldx #0
        A8
@co:    lda TV,x
        sta AT,y
        iny
        inx
        cpx #HD
        bne @co
        A16
        rts

; ---------------------------------------------------------------------------
;  matvec: ACC[j] = sum_i in[i] * W[i*stride + j]  (32-bit signed)
;  hardware: M7A ($211B, 16-bit signed) * M7B ($211C, 8-bit signed) -> $2134
; ---------------------------------------------------------------------------
matvec:
        .a16
        lda mv_nout
        asl a
        sta nout2
        ldx #0
@clr:   stz ACCL,x
        stz ACCH,x
        inx
        inx
        cpx nout2
        bne @clr
        stz mv_nz
        stz mv_i
@row:   A8
        ldy mv_i
        lda (mv_in),y
        beq @skip
        sta $211B               ; multiplicand low
        asl a
        lda #0
        adc #$FF
        eor #$FF                ; sign extension
        sta $211B               ; multiplicand high
        ldy #0
        ldx #0
@mac:   lda (mv_w),y
        sta $211C
        A16
        lda $2134
        eor #$8000              ; bias: accumulate unsigned
        clc
        adc ACCL,x
        sta ACCL,x
        bcc :+
        inc ACCH,x
:       A8
        inx
        inx
        iny
        cpy mv_nout
        bne @mac
        inc mv_nz
@skip:  A16
        lda mv_w
        clc
        adc mv_stride
        sta mv_w
        inc mv_i
        lda mv_i
        cmp mv_nin
        bne @row
        ; remove the bias: ACC -= nz * $8000
        lda mv_nz
        lsr a
        sta mv_ch
        lda #0
        ror a
        sta mv_cm
        ldx #0
@cor:   lda ACCL,x
        sec
        sbc mv_cm
        sta ACCL,x
        lda ACCH,x
        sbc mv_ch
        sta ACCH,x
        inx
        inx
        cpx nout2
        bne @cor
        rts

; ACC = (ACC + 2^(s-1)) >> s, A = s
acc_shift:
        .a16
        and #$00FF
        sta sh_cnt
        beq @done
        tay
        lda #1
@b:     dey
        beq :+
        asl a
        bra @b
:       sta rnd
        ldx #0
@el:    lda ACCL,x
        clc
        adc rnd
        sta t0
        lda ACCH,x
        adc #0
        sta t1
        ldy sh_cnt
@sh:    lda t1
        cmp #$8000
        ror t1
        ror t0
        dey
        bne @sh
        lda t0
        sta ACCL,x
        lda t1
        sta ACCH,x
        inx
        inx
        cpx nout2
        bne @el
@done:  rts

; TV[j] = clamp8(ACC[j] >> A)
requant:
        .a16
        jsr acc_shift
        ldx #0
        ldy #0
@el:    lda ACCH,x
        bmi @neg
        bne @psat
        lda ACCL,x
        cmp #128
        bcs @psat
        bra @st
@neg:   cmp #$FFFF
        bne @nsat
        lda ACCL,x
        cmp #$FF81
        bcc @nsat
        bra @st
@psat:  lda #127
        bra @st
@nsat:  lda #$FF81
@st:    A8
        sta TV,y
        A16
        inx
        inx
        iny
        cpy mv_nout
        bne @el
        rts

; ---------------------------------------------------------------------------
sample:
        .a16
        ldx #1
        A8
        lda LG
        sta t0
@mx:    lda LG,x
        sec
        sbc t0
        bvc :+
        eor #$80
:       bmi @no
        lda LG,x
        sta t0
@no:    inx
        cpx #NV
        bne @mx
        A16
        lda temp_idx
        xba
        clc
        adc #EXPS
        sta ptr
        stz esum
        ldx #0
@ex:    A8
        lda t0
        sec
        sbc LG,x
        A16
        and #$00FF
        tay
        lda (ptr),y
        and #$00FF
        A8
        sta EV,x
        A16
        clc
        adc esum
        sta esum
        inx
        cpx #NV
        bne @ex
        ; mask = 2^bitlen(E) - 1
        lda esum
        sta rlo
        stz mask
@mk:    lda rlo
        beq @draw
        lsr rlo
        sec
        rol mask
        bra @mk
@draw:  jsr rng_next
        and mask
        cmp esum
        bcs @draw
        sta rlo
        stz t1
        ldx #0
@cum:   lda EV,x
        and #$00FF
        clc
        adc t1
        sta t1
        lda rlo
        cmp t1
        bcc @hit
        inx
        cpx #NV
        bne @cum
        ldx #NV-1
@hit:   txa
        rts

; 16-bit xorshift (7, 9, 8) -> A
rng_next:
        .a16
        lda rng
        asl a
        asl a
        asl a
        asl a
        asl a
        asl a
        asl a
        eor rng
        sta rng
        lsr a
        lsr a
        lsr a
        lsr a
        lsr a
        lsr a
        lsr a
        lsr a
        lsr a
        eor rng
        sta rng
        xba
        and #$FF00
        eor rng
        sta rng
        rts

; ---------------------------------------------------------------------------
;  text output
; ---------------------------------------------------------------------------
putc:
        .a16
        cmp #0
        bne @ch
        jmp newline
@ch:    tax
        lda cur_col
        cmp #COLS
        bcc @fits
        phx
        jsr newline
        plx
        cpx #1                  ; drop a space that starts a wrapped line
        bne @fits
        rts
@fits:  lda VOCAB_ASCII,x
        and #$00FF
        sta t2
        lda cur_row
        asl a
        asl a
        asl a
        asl a
        asl a
        clc
        adc #$0800 + LEFT
        adc cur_col
        sta t0
        jsr wait_cmd
        lda t0
        sta ppu_addr
        lda t2
        sta ppu_data
        lda #1
        sta ppu_cmd
        inc cur_col
        rts

newline:
        .a16
        stz cur_col
        lda cur_row
        inc a
        and #31
        sta cur_row
        inc a
        and #31                 ; the row that scrolls in at the bottom
        asl a
        asl a
        asl a
        asl a
        asl a
        ora #$0800
        sta t0
        jsr wait_cmd
        lda t0
        sta ppu_addr
        lda #2
        sta ppu_cmd
        jsr wait_cmd
        lda cur_row             ; top row = cur_row - 26
        sec
        sbc #26
        and #31
        asl a
        asl a
        asl a
        dec a
        and #$00FF
        sta scroll_y
        rts

wait_cmd:
        .a16
        lda ppu_cmd
        bne wait_cmd
        rts

wait_frame:
        .a16
        lda frame
:       cmp frame
        beq :-
        rts

read_pressed:
        .a16
        lda buttons
        tax
        eor prevbtn
        and buttons
        stx prevbtn
        rts

status_cmd:
        .a16
        jsr wait_cmd
        lda #3
        sta ppu_cmd
        rts

; write the status line directly (forced blank only)
status_now:
        .a16
        lda #$0000 + 1*32 + 21
        sta $2116
        lda temp_idx
        asl a
        asl a
        asl a
        tax
        ldy #8
:       lda STATUS_TXT,x
        and #$00FF
        sta $2118
        inx
        dey
        bne :-
        rts

clear_text:
        .a16
        lda #$0800
        sta $2116
        ldx #1024
        lda #$0020
:       sta $2118
        dex
        bne :-
        rts

draw_title:
        .a16
        ldy #0
@str:   lda TITLE_TAB,y
        and #$00FF
        cmp #$FF
        beq @done
        asl a
        asl a
        asl a
        asl a
        asl a
        sta t0
        lda TITLE_TAB+1,y
        and #$00FF
        clc
        adc #$0800
        adc t0
        sta $2116
        lda TITLE_TAB+2,y
        tax
@ch:    lda a:$0000,x
        and #$00FF
        beq @end
        sta $2118
        inx
        bra @ch
@end:   iny
        iny
        iny
        iny
        bra @str
@done:  rts

; ---------------------------------------------------------------------------
nmi:
        jml $800000 + nmi_fast
nmi_fast:
        rep #$30
        .a16
        .i16
        pha
        phx
        phy
        phb
        phd
        lda #$0000
        tcd
        A8
        lda #$80
        pha
        plb
        lda $4210
        A16
        lda nmi_on
        beq @joy
        lda ppu_cmd
        beq @scroll
        cmp #3
        beq @status
        lda ppu_addr
        sta $2116
        lda ppu_cmd
        cmp #1
        bne @clear
        lda ppu_data
        sta $2118
        bra @cmd_done
@clear: ldx #32
        lda #$0020
:       sta $2118
        dex
        bne :-
        bra @cmd_done
@status:jsr status_now
@cmd_done:
        stz ppu_cmd
@scroll:
        A8
        lda scroll_y
        sta $2110
        stz $2110
        A16
@joy:   A8
:       lda $4212
        and #$01
        bne :-
        A16
        lda $4218
        sta buttons
        inc frame
        pld
        plb
        ply
        plx
        pla
        rti

irq_stub:
        rti

; ---------------------------------------------------------------------------
.segment "RODATA"
.include "gen/tables.inc"

STATUS_TXT: .byte "TEMP 0.5", "TEMP 0.7", "TEMP .85", "TEMP 1.0"

TITLE_TAB:
        .byte 6, 7
        .word str1
        .byte 8, 7
        .word str2
        .byte 12, 3
        .word str3
        .byte 13, 3
        .word str4
        .byte 14, 3
        .word str5
        .byte 15, 3
        .word str6
        .byte 18, 3
        .word str6b
        .byte 21, 4
        .word str7
        .byte 22, 4
        .word str8
        .byte 25, 10
        .word str9
        .byte $FF
str1:   .byte "RETRO-LLM  FOR THE SNES", 0
str2:   .byte "   A TINY TRANSFORMER", 0
str3:   .byte "2 LAYERS, 4 HEADS, D=32", 0
str4:   .byte "18688 INT8 PARAMETERS", 0
str5:   .byte "32-CHAR CONTEXT (KV CACHE)", 0
str6:   .byte "TRAINED ON SHAKESPEARE", 0
str6b:  .byte "MATH ON THE PPU MULTIPLIER", 0
str7:   .byte "UP/DOWN: TEMPERATURE", 0
str8:   .byte "A: NEW STORY", 0
str9:   .byte "PRESS START", 0

.segment "HEADER"
        .byte "RETRO-LLM SNES       "   ; 21 bytes
        .byte $30                       ; LoROM, FastROM
        .byte $00                       ; ROM only
        .byte $05                       ; 32 KB
        .byte $00                       ; no SRAM
        .byte $01                       ; North America
        .byte $00
        .byte $00                       ; version
        .word $FFFF                     ; checksum complement (patched)
        .word $0000                     ; checksum (patched)

.segment "VECTORS"
        .word 0, 0, irq_stub, irq_stub, irq_stub, nmi, 0, irq_stub
        .word 0, 0, irq_stub, 0, irq_stub, irq_stub, reset, irq_stub
