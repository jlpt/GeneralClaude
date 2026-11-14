; ---------------------------------------------------------------------------
;  retro-llm for the NES: a 2-layer int8 transformer that writes Shakespeare.
;
;  ca65 / ld65 (cc65 suite). Mapper 1 (MMC1) with 8 KB battery WRAM, used for
;  the KV cache. All math is integer and bit-exact with train/ref.py.
;
;  Multiply: quarter squares, a*b = f(a+b) - f(b-a), f(n) = n*n/4.
;  The inner loop lives in RAM and is self-modified per input row.
; ---------------------------------------------------------------------------

.import __RAMCODE_LOAD__, __RAMCODE_RUN__, __RAMCODE_SIZE__
.include "gen/model.inc"        ; weights, shifts, model constants

D       = 32
F       = 64
NH      = 4
HD      = 8
CTX     = 32
NV      = 36

LEFT    = 2                     ; text column offset in the nametable
COLS    = 24                    ; text columns

PPUCTRL   = $2000
PPUMASK   = $2001
PPUSTATUS = $2002
OAMADDR   = $2003
PPUSCROLL = $2005
PPUADDR   = $2006
PPUDATA   = $2007
OAMDMA    = $4014
JOY1      = $4016

BTN_A      = $80
BTN_B      = $40
BTN_SELECT = $20
BTN_START  = $10
BTN_UP     = $08
BTN_DOWN   = $04

; ---------------------------------------------------------------------------
.segment "HEADER"
    .byte "NES", $1A
    .byte 2                     ; 2 x 16 KB PRG
    .byte 1                     ; 1 x 8 KB CHR
    .byte $12                   ; mapper 1, battery-backed PRG RAM
    .byte $00
    .res 8, 0

; ---------------------------------------------------------------------------
.segment "ZEROPAGE"
mv_in:      .res 2              ; matvec input vector
mv_w:       .res 2              ; matvec weight row pointer
mv_nin:     .res 1
mv_nout:    .res 1
mv_stride:  .res 1
mv_i:       .res 1
mv_nz:      .res 1
mv_tmp:     .res 1
mv_cm:      .res 1
mv_ch:      .res 1
sh_cnt:     .res 1
rnd0:       .res 1
rnd1:       .res 1
t0:         .res 1
t1:         .res 1
t2:         .res 1
ptr:        .res 2
ptr2:       .res 2
layer:      .res 1
lbase:      .res 1              ; high byte of current layer's weights
kvoff:      .res 1              ; layer * 4 (high byte offset into KV cache)
head:       .res 1
slot:       .res 1
nvalid:     .res 1
tok:        .res 1
cnt:        .res 1
mlo:        .res 1
mhi:        .res 1
elo:        .res 1
ehi:        .res 1
rlo:        .res 1
rhi:        .res 1
quot:       .res 1
mask_lo:    .res 1
mask_hi:    .res 1
rng_lo:     .res 1
rng_hi:     .res 1
frame:      .res 2
buttons:    .res 1
prevbtn:    .res 1
pressed:    .res 1
temp_idx:   .res 1
ppu_cmd:    .res 1              ; 0 idle, 1 put char, 2 clear row
ppu_addr:   .res 2
ppu_data:   .res 1
scroll_y:   .res 1
cur_row:    .res 1
cur_col:    .res 1
nmi_on:     .res 1
logpos:     .res 2

.segment "OAM"
oam:        .res 256

.segment "BSS"
XV:     .res D                  ; residual stream (signed int8)
QV:     .res D
AT:     .res D
HV:     .res F
TV:     .res F                  ; requantised matvec output
ACC0:   .res F                  ; 24-bit accumulators, split by byte
ACC1:   .res F
ACC2:   .res F
SCL:    .res CTX                ; attention scores (lo, hi ^ $80)
SCH:    .res CTX
PV:     .res CTX                ; attention probabilities 0..127
EV:     .res NV                 ; sampling weights
LG:     .res NV                 ; logits

.segment "WRAM"
KC:     .res 2*1024             ; $6000 K cache [layer][channel][slot] (biased +128)
VC:     .res 2*1024             ; $6800 V cache [layer][slot][channel] (biased +128)
TOKLOG: .res $800               ; $7000 ring log of generated tokens
MAGIC:  .res 4                  ; "LLM!" once running (for the test harness)
SEED:   .res 2
NTOK:   .res 2                  ; tokens generated since the seed was set
TEMPLOG:.res 1

; ---------------------------------------------------------------------------
.segment "TABLES"               ; page aligned
.include "gen/tables.inc"

; ---------------------------------------------------------------------------
.segment "RAMCODE"
; inner loop of matvec:  ACC[x] += a * W[x]  for x in 0..nout-1
; (a is baked into the four table operands, W row address into mvk_w)
mvk_inner:
        ldx #0
mvk_loop:
mvk_w:  ldy $FFFF,x             ; biased weight b' = b + 128
mvk_qla:lda QLO,y               ; f(a + b)
        sec
mvk_qlb:sbc QLO,y               ; - f(b - a)
        sta mv_tmp
mvk_qha:lda QHI,y
mvk_qhb:sbc QHI,y
        eor #$80                ; bias product by +$8000 (unsigned accumulate)
        tay
        lda mv_tmp
        clc
        adc ACC0,x
        sta ACC0,x
        tya
        adc ACC1,x
        sta ACC1,x
        bcc mvk_nc
        inc ACC2,x
mvk_nc: inx
        cpx mv_nout
        bne mvk_loop
        rts

; ---------------------------------------------------------------------------
.segment "CODE"

reset:
        sei
        cld
        ldx #$40
        stx $4017
        ldx #$FF
        txs
        inx
        stx PPUCTRL
        stx PPUMASK
        stx $4010
        ; MMC1: reset shift register, then control = $0E
        ;       (vertical mirroring, 16K PRG with $C000 fixed, 8K CHR)
        lda #$80
        sta $8000
        lda #$0E
        jsr mmc1_ctrl
        lda #$00
        jsr mmc1_prg            ; bank 0 at $8000, WRAM enabled
        lda #$00
        jsr mmc1_chr0
@vb1:   bit PPUSTATUS
        bpl @vb1
        ; clear internal RAM
        lda #0
        tax
@clr:   sta $00,x
        sta $0100,x
        sta $0300,x
        sta $0400,x
        sta $0500,x
        sta $0600,x
        sta $0700,x
        inx
        bne @clr
        lda #$FF
@clroam:sta oam,x
        inx
        bne @clroam
        ; copy the RAM code
        ldx #0
@cp:    lda __RAMCODE_LOAD__,x
        sta __RAMCODE_RUN__,x
        inx
        cpx #<__RAMCODE_SIZE__
        bne @cp
@vb2:   bit PPUSTATUS
        bpl @vb2

        jsr load_palette
        jsr clear_nametable
        jsr draw_title
        lda #2
        sta temp_idx
        jsr update_status
        lda #0
        sta scroll_y
        lda #1
        sta nmi_on
        lda #$80
        sta PPUCTRL             ; NMI on
        jsr wait_frame
        lda #%00011110          ; show bg + sprites, no left clipping
        sta PPUMASK

        ; wait for START, seed RNG from the frame counter
@title: jsr wait_frame
        jsr read_pressed
        and #BTN_START
        beq @title
        lda frame
        sta rng_lo
        lda frame+1
        sta rng_hi
        ora rng_lo
        bne @seeded
        inc rng_lo              ; seed must be non-zero
@seeded:
        jsr start_session

main_loop:
        jsr read_pressed
        sta pressed
        and #BTN_UP
        beq @noup
        lda temp_idx
        cmp #3
        beq @noup
        inc temp_idx
        jsr update_status
@noup:  lda pressed
        and #BTN_DOWN
        beq @nodn
        lda temp_idx
        beq @nodn
        dec temp_idx
        jsr update_status
@nodn:  lda pressed
        and #BTN_A
        beq @noa
        jsr start_session       ; new story, keep the RNG running
@noa:
        jsr forward
        jsr sample
        sta tok
        jsr log_token
        jsr putc
        ; advance the KV ring
        lda slot
        clc
        adc #1
        and #CTX-1
        sta slot
        lda nvalid
        cmp #CTX
        beq @full
        inc nvalid
@full:  jmp main_loop

; ---------------------------------------------------------------------------
; reset the model state and clear the screen
start_session:
        lda #0
        sta slot
        sta logpos
        sta logpos+1
        sta NTOK
        sta NTOK+1
        lda #1
        sta nvalid
        lda #0                  ; NEWLINE token
        sta tok
        lda rng_lo
        sta SEED
        lda rng_hi
        sta SEED+1
        lda temp_idx
        sta TEMPLOG
        lda #'L'
        sta MAGIC
        sta MAGIC+1
        lda #'M'
        sta MAGIC+2
        lda #'!'
        sta MAGIC+3
        ; blank the screen with rendering off
        jsr wait_frame
        lda #0
        sta nmi_on
        sta PPUMASK
        jsr clear_nametable
        lda #28
        sta cur_row
        lda #0
        sta cur_col
        lda #0                  ; ((28 + 2) mod 30) * 8
        sta scroll_y
        lda #1
        sta nmi_on
        jsr wait_frame
        lda #%00011110
        sta PPUMASK
        rts

log_token:
        lda logpos
        sta ptr
        lda logpos+1
        and #$07
        clc
        adc #>TOKLOG
        sta ptr+1
        ldy #0
        lda tok
        sta (ptr),y
        inc logpos
        bne :+
        inc logpos+1
:       inc NTOK
        bne :+
        inc NTOK+1
:       rts

; ---------------------------------------------------------------------------
;  forward(tok): one transformer step, leaves int8 logits in LG
; ---------------------------------------------------------------------------
forward:
        ; x = emb[tok]
        lda tok
        lsr a
        lsr a
        lsr a
        sta ptr+1               ; (tok*32) >> 8
        lda tok
        asl a
        asl a
        asl a
        asl a
        asl a
        sta ptr                 ; (tok*32) & $FF
        lda ptr
        clc
        adc #<EMB
        sta ptr
        lda ptr+1
        adc #>EMB
        sta ptr+1
        ldy #D-1
@emb:   lda (ptr),y
        sta XV,y
        dey
        bpl @emb

        lda #0
        sta layer
@layer:
        lda layer
        asl a
        asl a
        sta kvoff               ; layer * $400 -> high byte offset 4
        asl a
        asl a
        asl a
        clc
        adc #>LAYERS            ; layer * $2000
        sta lbase

        ; ---- q = x Wq
        lda #0
        jsr mv_x_d              ; weights at lbase + $00
        ldx #0
        jsr shift_of
        jsr requant
        ldx #D-1
@cq:    lda TV,x
        sta QV,x
        dex
        bpl @cq

        ; ---- k = x Wk  -> KC[layer][c][slot]
        lda #$04
        jsr mv_x_d
        ldx #1
        jsr shift_of
        jsr requant
        lda slot
        sta ptr
        lda #>KC
        clc
        adc kvoff
        sta ptr+1
        ldx #0
        ldy #0
@ck:    lda TV,x
        eor #$80
        sta (ptr),y
        lda ptr
        clc
        adc #CTX
        sta ptr
        bcc :+
        inc ptr+1
:       inx
        cpx #D
        bne @ck

        ; ---- v = x Wv  -> VC[layer][slot][c]
        lda #$08
        jsr mv_x_d
        ldx #2
        jsr shift_of
        jsr requant
        jsr vc_slot_ptr         ; ptr = VC + layer*$400 + slot*32
        ldy #D-1
@cv:    lda TV,y
        eor #$80
        sta (ptr),y
        dey
        bpl @cv

        ; ---- attention, one head at a time
        lda #0
        sta head
@head:
        jsr attend_head
        inc head
        lda head
        cmp #NH
        bne @head

        ; ---- x += att Wo
        lda #<AT
        sta mv_in
        lda #>AT
        sta mv_in+1
        lda #$0C
        jsr mv_d_d
        ldx #3
        jsr shift_of
        jsr requant
        jsr add_tv_to_x

        ; ---- h = relu(x W1)
        lda #<XV
        sta mv_in
        lda #>XV
        sta mv_in+1
        lda #D
        sta mv_nin
        lda #F
        sta mv_nout
        sta mv_stride
        lda #0
        sta mv_w
        lda lbase
        clc
        adc #$10
        sta mv_w+1
        jsr matvec
        ldx #4
        jsr shift_of
        jsr requant
        ldx #F-1
@relu:  lda TV,x
        bpl :+
        lda #0
:       sta HV,x
        dex
        bpl @relu

        ; ---- x += h W2
        lda #<HV
        sta mv_in
        lda #>HV
        sta mv_in+1
        lda #F
        sta mv_nin
        lda #D
        sta mv_nout
        sta mv_stride
        lda #0
        sta mv_w
        lda lbase
        clc
        adc #$18
        sta mv_w+1
        jsr matvec
        ldx #5
        jsr shift_of
        jsr requant
        jsr add_tv_to_x

        inc layer
        lda layer
        cmp #2
        beq @out
        jmp @layer

@out:   ; ---- logits = x Wout
        lda #<XV
        sta mv_in
        lda #>XV
        sta mv_in+1
        lda #D
        sta mv_nin
        lda #NV
        sta mv_nout
        sta mv_stride
        lda #<WOUT
        sta mv_w
        lda #>WOUT
        sta mv_w+1
        jsr matvec
        lda #SHIFT_OUT
        jsr requant
        ldx #NV-1
@lg:    lda TV,x
        sta LG,x
        dex
        bpl @lg
        rts

; A = shift for matrix X of the current layer
shift_of:
        lda layer
        beq :+
        txa
        clc
        adc #6
        tax
:       lda SHIFTS,x
        rts

; matvec: in = XV (32), weights at lbase + A*256 (32x32)
mv_x_d:
        tax
        lda #<XV
        sta mv_in
        lda #>XV
        sta mv_in+1
        txa
; matvec: in = mv_in (32), weights at lbase + A*256 (32x32)
mv_d_d:
        clc
        adc lbase
        sta mv_w+1
        lda #0
        sta mv_w
        lda #D
        sta mv_nin
        sta mv_nout
        sta mv_stride
        jmp matvec

vc_slot_ptr:
        lda slot
        lsr a
        lsr a
        lsr a                   ; slot*32 >> 8
        clc
        adc #>VC
        adc kvoff
        sta ptr+1
        lda slot
        asl a
        asl a
        asl a
        asl a
        asl a
        sta ptr                 ; VC is page aligned
        rts

add_tv_to_x:
        ldx #D-1
@l:     lda XV,x
        clc
        adc TV,x
        bvc @ok                 ; no signed overflow: still need clamp to +-127
        lda XV,x                ; overflow: saturate in the direction of x
        bmi @neg
        lda #127
        bne @st
@neg:   lda #$81
        bne @st
@ok:    cmp #$80
        bne @st
        lda #$81                ; -128 -> -127
@st:    sta XV,x
        dex
        bpl @l
        rts

; ---------------------------------------------------------------------------
attend_head:
        ; scores[j] = (q_h . K_h[j]) rshift 7  -  slope_h * dist(j)
        lda head
        asl a
        asl a
        asl a                   ; head * 8
        clc
        adc #<QV
        sta mv_in
        lda #>QV
        adc #0
        sta mv_in+1
        lda #HD
        sta mv_nin
        lda nvalid
        sta mv_nout
        lda #CTX
        sta mv_stride
        lda #0
        sta mv_w
        lda #>KC
        clc
        adc kvoff
        adc head                ; + head*8*32 = head*$100
        sta mv_w+1
        jsr matvec
        lda #SCORE_SHIFT
        jsr acc_shift

        ; ALiBi row for this head
        lda head
        asl a
        asl a
        asl a
        asl a
        asl a
        clc
        adc #<ALIBI
        sta ptr
        lda #>ALIBI
        adc #0
        sta ptr+1

        ldx #0
@sc:    txa
        eor #$FF
        sec
        adc slot                ; slot - j
        and #CTX-1
        tay                     ; dist
        lda ACC0,x
        sec
        sbc (ptr),y
        sta SCL,x
        lda ACC1,x
        sbc #0
        eor #$80
        sta SCH,x
        inx
        cpx nvalid
        bne @sc

        ; max (biased-unsigned compare)
        lda SCL
        sta mlo
        lda SCH
        sta mhi
        ldx #1
        cpx nvalid
        beq @maxd
@mx:    lda SCL,x
        cmp mlo
        lda SCH,x
        sbc mhi
        bcc @nomx
        lda SCL,x
        sta mlo
        lda SCH,x
        sta mhi
@nomx:  inx
        cpx nvalid
        bne @mx
@maxd:
        ; e[j] = EXP_ATT[min(m - s, 255)], E = sum e
        lda #0
        sta elo
        sta ehi
        ldx #0
@ex:    lda mlo
        sec
        sbc SCL,x
        tay
        lda mhi
        sbc SCH,x
        beq :+
        ldy #255
:       lda EXP_ATT,y
        sta PV,x                ; temporarily hold e in PV
        clc
        adc elo
        sta elo
        bcc :+
        inc ehi
:       inx
        cpx nvalid
        bne @ex

        ; p[j] = min(127, (e << 7) / E)
        ldx #0
@pr:    lda PV,x
        beq @pz
        sta rlo
        lda #0
        sta rhi
        sta quot
        ldy #7
@dv:    asl rlo
        rol rhi
        lda rlo
        sec
        sbc elo
        sta t0
        lda rhi
        sbc ehi
        bcc @dn
        sta rhi
        lda t0
        sta rlo
@dn:    rol quot                ; carry = 1 if subtracted
        dey
        bne @dv
        lda quot
@pz:    sta PV,x
        inx
        cpx nvalid
        bne @pr

        ; o_h = sum_j p[j] * V[j]_h  >> 7
        lda #<PV
        sta mv_in
        lda #>PV
        sta mv_in+1
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
        sta mv_w                ; VC + layer*$400 + head*8 (VC page aligned)
        lda #>VC
        clc
        adc kvoff
        sta mv_w+1
        jsr matvec
        lda #7
        jsr requant
        lda head
        asl a
        asl a
        asl a
        tay
        ldx #0
@co:    lda TV,x
        sta AT,y
        iny
        inx
        cpx #HD
        bne @co
        rts

; ---------------------------------------------------------------------------
;  matvec: ACC[j] = sum_i in[i] * W[i][j]   (24-bit signed)
;  mv_in, mv_nin, mv_w (row 0), mv_stride, mv_nout
; ---------------------------------------------------------------------------
matvec:
        ldx #0
        lda #0
@clr:   sta ACC0,x
        sta ACC1,x
        sta ACC2,x
        inx
        cpx mv_nout
        bne @clr
        sta mv_nz
        sta mv_i
@row:   ldy mv_i
        lda (mv_in),y
        beq @skip               ; zero input: whole row contributes nothing
        tax
        eor #$80                ; 128 + a
        sta mvk_qla+1
        sta mvk_qha+1
        txa
        eor #$FF
        clc
        adc #$81                ; 128 - a
        sta mvk_qlb+1
        sta mvk_qhb+1
        lda mv_w
        sta mvk_w+1
        lda mv_w+1
        sta mvk_w+2
        jsr mvk_inner
        inc mv_nz
@skip:  lda mv_w
        clc
        adc mv_stride
        sta mv_w
        bcc :+
        inc mv_w+1
:       inc mv_i
        lda mv_i
        cmp mv_nin
        bne @row
        ; remove the +$8000 bias of every accumulated row
        lda mv_nz
        lsr a
        sta mv_ch
        lda #0
        ror a
        sta mv_cm
        ldx #0
@cor:   lda ACC1,x
        sec
        sbc mv_cm
        sta ACC1,x
        lda ACC2,x
        sbc mv_ch
        sta ACC2,x
        inx
        cpx mv_nout
        bne @cor
        rts

; ACC[j] = (ACC[j] + 2^(s-1)) >> s   (A = s, arithmetic, in place)
acc_shift:
        sta sh_cnt
        tay
        beq @done
        lda #0
        sta rnd0
        sta rnd1
        sec                     ; build 1 << (s-1)
@b:     rol rnd0
        rol rnd1
        clc
        dey
        bne @b
        ldx #0
@el:    lda ACC0,x
        clc
        adc rnd0
        sta t0
        lda ACC1,x
        adc rnd1
        sta t1
        lda ACC2,x
        adc #0
        sta t2
        ldy sh_cnt
@sh:    lda t2
        cmp #$80
        ror t2
        ror t1
        ror t0
        dey
        bne @sh
        lda t0
        sta ACC0,x
        lda t1
        sta ACC1,x
        lda t2
        sta ACC2,x
        inx
        cpx mv_nout
        bne @el
@done:  rts

; TV[j] = clamp8(ACC[j] >> A)
requant:
        jsr acc_shift
        ldx #0
@el:    lda ACC2,x
        bmi @neg
        ora ACC1,x
        bne @psat
        lda ACC0,x
        bpl @st
@psat:  lda #127
        bne @st
@neg:   and ACC1,x
        cmp #$FF
        bne @nsat
        lda ACC0,x
        cmp #$81
        bcs @st
@nsat:  lda #$81
@st:    sta TV,x
        inx
        cpx mv_nout
        bne @el
        rts

; ---------------------------------------------------------------------------
;  sample(): temperature sampling from LG -> A = token
; ---------------------------------------------------------------------------
sample:
        ; lmax (signed)
        lda LG
        sta t0
        ldx #1
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
        ; table pointer
        lda #<EXPS
        sta ptr
        lda #>EXPS
        clc
        adc temp_idx
        sta ptr+1
        lda #0
        sta elo
        sta ehi
        ldx #0
@ex:    lda t0
        sec
        sbc LG,x                ; 0..254
        tay
        lda (ptr),y
        sta EV,x
        clc
        adc elo
        sta elo
        bcc :+
        inc ehi
:       inx
        cpx #NV
        bne @ex
        ; mask = 2^bitlen(E) - 1
        lda elo
        sta rlo
        lda ehi
        sta rhi
        lda #0
        sta mask_lo
        sta mask_hi
@mk:    lda rlo
        ora rhi
        beq @mkd
        lsr rhi
        ror rlo
        sec
        rol mask_lo
        rol mask_hi
        jmp @mk
@mkd:
@draw:  jsr rng_next
        lda rng_lo
        and mask_lo
        sta rlo
        lda rng_hi
        and mask_hi
        sta rhi
        lda rlo                 ; r < E ?
        cmp elo
        lda rhi
        sbc ehi
        bcs @draw
        ; walk the cumulative sum
        lda #0
        sta t1
        sta t2
        ldx #0
@cum:   lda EV,x
        clc
        adc t1
        sta t1
        bcc :+
        inc t2
:       lda rlo                 ; r < acc ?
        cmp t1
        lda rhi
        sbc t2
        bcc @hit
        inx
        cpx #NV
        bne @cum
        ldx #NV-1
@hit:   txa
        rts

; 16-bit xorshift (7, 9, 8)
rng_next:
        lda rng_lo
        lsr a
        sta t0                  ; lo >> 1
        lda rng_hi
        lsr a
        lda #0
        ror a                   ; (hi & 1) << 7
        ora t0
        eor rng_hi
        sta rng_hi              ; hi ^= (x << 7) >> 8
        lda rng_lo
        lsr a
        lda #0
        ror a
        eor rng_lo
        sta rng_lo              ; lo ^= lo << 7
        lda rng_hi
        lsr a
        eor rng_lo
        sta rng_lo              ; x ^= x >> 9
        eor rng_hi
        sta rng_hi              ; x ^= x << 8
        rts

; ---------------------------------------------------------------------------
;  text output (via the NMI command slot)
; ---------------------------------------------------------------------------
putc:
        cmp #0
        bne @ch
        jmp newline
@ch:    tax
        lda cur_col
        cmp #COLS
        bcc @fits
        txa
        pha
        jsr newline
        pla
        tax
        cpx #1                  ; skip a space at the start of a wrapped line
        bne @fits
        rts
@fits:  lda VOCAB_ASCII,x
        sta t2
        jsr row_addr
        lda ptr
        clc
        adc #LEFT
        adc cur_col
        sta t0
        lda ptr+1
        adc #0
        sta t1
        jsr wait_cmd
        lda t0
        sta ppu_addr
        lda t1
        sta ppu_addr+1
        lda t2
        sta ppu_data
        lda #1
        sta ppu_cmd
        inc cur_col
        rts

newline:
        lda #0
        sta cur_col
        ldx cur_row
        inx
        cpx #30
        bne :+
        ldx #0
:       stx cur_row
        ; clear the row after it (the one about to scroll in at the bottom)
        inx
        cpx #30
        bne :+
        ldx #0
:       txa
        pha
        jsr row_addr_x
        jsr wait_cmd
        lda ptr
        sta ppu_addr
        lda ptr+1
        sta ppu_addr+1
        lda #2
        sta ppu_cmd
        jsr wait_cmd
        ; scroll so that row cur_row+2 is at the top
        pla
        tax
        inx
        cpx #30
        bne :+
        ldx #0
:       txa
        asl a
        asl a
        asl a
        sta scroll_y
        rts

; ptr = $2000 + cur_row*32
row_addr:
        ldx cur_row
row_addr_x:
        txa
        lsr a
        lsr a
        lsr a
        clc
        adc #$20
        sta ptr+1
        txa
        asl a
        asl a
        asl a
        asl a
        asl a
        sta ptr
        rts

wait_cmd:
        lda ppu_cmd
        bne wait_cmd
        rts

wait_frame:
        lda frame
@w:     cmp frame
        beq @w
        rts

; A = newly pressed buttons
read_pressed:
        lda buttons
        tax
        eor prevbtn
        and buttons
        stx prevbtn
        rts

; status sprites: "TEMP" / "0.85" in the top right corner
update_status:
        ldx #0
        ldy #0
@lp:    lda status_y,x
        sta oam,y
        lda temp_idx
        asl a
        asl a
        clc
        adc #4
        sta t0                  ; offset of the value string
        cpx #4
        bcc @top
        txa
        sec
        sbc #4
        clc
        adc t0
        sta t1
        stx t2
        ldx t1
        lda status_txt,x
        ldx t2
        jmp @tile
@top:   lda status_txt,x
@tile:  sta oam+1,y
        lda #0
        sta oam+2,y
        lda status_x,x
        sta oam+3,y
        iny
        iny
        iny
        iny
        inx
        cpx #8
        bne @lp
        rts

status_y:   .byte 15,15,15,15, 24,24,24,24
status_x:   .byte 216,224,232,240, 216,224,232,240
status_txt: .byte "TEMP", "0.50", "0.70", "0.85", "1.00"

; ---------------------------------------------------------------------------
load_palette:
        bit PPUSTATUS
        lda #$3F
        sta PPUADDR
        lda #$00
        sta PPUADDR
        ldx #0
@l:     lda palette,x
        sta PPUDATA
        inx
        cpx #32
        bne @l
        rts

palette:
        .byte $0F,$30,$10,$00, $0F,$30,$10,$00, $0F,$30,$10,$00, $0F,$30,$10,$00
        .byte $0F,$28,$16,$00, $0F,$28,$16,$00, $0F,$28,$16,$00, $0F,$28,$16,$00

; full nametable + attributes = space / 0 (rendering must be off)
clear_nametable:
        bit PPUSTATUS
        lda #$20
        sta PPUADDR
        lda #$00
        sta PPUADDR
        lda #' '
        ldy #4
        ldx #0
@l:     cpy #1
        bne :+
        cpx #$C0
        bne :+
        lda #0                  ; attribute table
:       sta PPUDATA
        inx
        bne @l
        dey
        bne @l
        rts

draw_title:
        ldy #0
@str:   lda title_tab,y         ; row
        cmp #$FF
        beq @done
        tax
        jsr row_addr_x
        lda ptr
        clc
        adc title_tab+1,y       ; column
        sta ptr
        bit PPUSTATUS
        lda ptr+1
        sta PPUADDR
        lda ptr
        sta PPUADDR
        lda title_tab+2,y
        sta ptr2
        lda title_tab+3,y
        sta ptr2+1
        tya
        pha
        ldy #0
@ch:    lda (ptr2),y
        beq @end
        sta PPUDATA
        iny
        bne @ch
@end:   pla
        clc
        adc #4
        tay
        bne @str
@done:  rts

title_tab:
        .byte 6, 5
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
        .byte 19, 4
        .word str7
        .byte 20, 4
        .word str8
        .byte 24, 9
        .word str9
        .byte $FF
str1:   .byte "RETRO-LLM  FOR THE NES", 0
str2:   .byte "A TINY TRANSFORMER", 0
str3:   .byte "2 LAYERS, 4 HEADS, D=32", 0
str4:   .byte "18688 INT8 PARAMETERS", 0
str5:   .byte "32-CHAR CONTEXT (KV CACHE)", 0
str6:   .byte "TRAINED ON SHAKESPEARE", 0
str7:   .byte "UP/DOWN: TEMPERATURE", 0
str8:   .byte "A: NEW STORY", 0
str9:   .byte "PRESS START", 0

; ---------------------------------------------------------------------------
mmc1_ctrl:
        sta $8000
        lsr a
        sta $8000
        lsr a
        sta $8000
        lsr a
        sta $8000
        lsr a
        sta $8000
        rts
mmc1_chr0:
        sta $A000
        lsr a
        sta $A000
        lsr a
        sta $A000
        lsr a
        sta $A000
        lsr a
        sta $A000
        rts
mmc1_prg:
        sta $E000
        lsr a
        sta $E000
        lsr a
        sta $E000
        lsr a
        sta $E000
        lsr a
        sta $E000
        rts

; ---------------------------------------------------------------------------
nmi:
        pha
        txa
        pha
        tya
        pha
        lda nmi_on
        beq @skip
        lda #0
        sta OAMADDR
        lda #>oam
        sta OAMDMA
        lda ppu_cmd
        beq @scroll
        bit PPUSTATUS
        lda ppu_addr+1
        sta PPUADDR
        lda ppu_addr
        sta PPUADDR
        lda ppu_cmd
        cmp #1
        bne @clear
        lda ppu_data
        sta PPUDATA
        jmp @cmd_done
@clear: lda #' '
        ldx #32
:       sta PPUDATA
        dex
        bne :-
@cmd_done:
        lda #0
        sta ppu_cmd
@scroll:
        bit PPUSTATUS
        lda #0
        sta PPUSCROLL
        lda scroll_y
        sta PPUSCROLL
        lda #$80
        sta PPUCTRL
@skip:
        ; controller
        lda #1
        sta JOY1
        lda #0
        sta JOY1
        ldx #8
:       lda JOY1
        lsr a
        rol buttons
        dex
        bne :-
        inc frame
        bne :+
        inc frame+1
:       pla
        tay
        pla
        tax
        pla
irq:    rti

.segment "VECTORS"
        .word nmi, reset, irq

.segment "CHARS"
.incbin "gen/font.chr"
