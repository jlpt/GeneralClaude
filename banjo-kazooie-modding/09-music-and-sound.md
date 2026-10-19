# 9. Music and Sound

[← Text and dialog](08-text-and-dialog.md) · [Index](README.md) · [Next: Game rules and tweaks →](10-game-rules-and-tweaks.md)

## 9.1 How music works

- Tracks are identified by `enum comusic_e` (`include/enums.h`), from
  `COMUSIC_0_DING_A` to `COMUSIC_AC_GOOD_ENDING`.
- Each track is a **MIDI asset** at `0x1516 + track_id`
  (`MUSIC_TRACK_ASSET_BASE_ID` in `src/core1/code_11AC0.c`). It's loaded with
  `assetcache_get()` and played by libultra's compressed-sequence player
  (`n_alCSeqNew`, `N_ALCSPlayer`).
- Instruments and samples come from the **soundfont** ROM segments
  (`soundfont1ctl/tbl`, `soundfont2ctl/tbl` in the splat yaml). They aren't in
  the asset table.

## 9.2 Changing a map's music

`sCore2_9B650MapMusicList[0x84]` in `src/core2/code_9B650.c`:

```c
typedef struct{
    s16 map_id;
    s16 track_id;           // enum comusic_e, -1 for none
    s16 secondary_track_id; // second track faded against the first (TTC seagulls by height), -1 for none
    s16 flags;              // bit 0: skip func_8025A9D4 when the music starts
}MapMusicInfo;

{MAP_2_MM_MUMBOS_MOUNTAIN, COMUSIC_2_MM, -1, 1},
```

Swap `track_id` to make any map play any track. `sCore2_9B650SnsMusicOverrides`
in the same file swaps a map's music when a Stop 'n' Swop item is active.

### Per-area instrument layers

Banjo-Kazooie's signature trick of re-orchestrating the same song as you walk
around is done by **muting MIDI channels by position**. It lives in
`src/core1/code_CE60.c`, a `switch (gsworld_getMap())`:

```c
case MAP_2_MM_MUMBOS_MOUNTAIN:
    if (midichannel_isPlayerInRange(-4450, 4550, 1900) || midichannel_isPlayerInRange(-6007, 6176, 620)) {
        midichannel_setChanMask(0x1cc0);          // bit n = MIDI channel n audible
    } else if (midichannel_isPlayerInRange(4511, -1888, 2000)) {
        midichannel_setChanMask(0xb0c0);
    } ...
    else {
        midichannel_setChanMaskFromWaterState(0x200, 0x103f);   // underwater mix
    }
```

`midichannel_isPlayerInRange(x, z, radius)` tests the player's XZ distance.
For a new track, give each area's instruments their own channels and add a
`case` for your map.

## 9.3 Playing music and jingles from code

| Function | Use |
|---|---|
| `coMusicPlayer_playMusic(COMUSIC_xx, volume)` | Start or restart a track or jingle (volume around 20000–32000) |
| `coMusicPlayer_playMusicWeak(COMUSIC_xx, volume)` | Start it, or just adjust the volume if it's already playing |
| `comusic_playTrack(COMUSIC_xx)` | Queue a track |
| `musicKeepsPlaying()` | Keep the current music across the next transition (used before warps) |

Example from `src/core2/code_BEF20.c` (100th note):

```c
coMusicPlayer_playMusic(COMUSIC_36_100TH_NOTE_COLLECTED, 20000);
```

## 9.4 Replacing or adding songs

1. **Replace**: overwrite `assets/midi/<0x1516 + id>.midi.bin` with another
   track in the same **N64 compressed MIDI** format (libultra `ALCSeq`). That's
   the format BK's own `.midi.bin` files are in. Community N64 MIDI tools
   convert between standard MIDI and this format. Instrument numbers refer to
   the game's soundfont, so map your MIDI's programs to BK instruments.
2. **Add**: append a MIDI asset and a matching `COMUSIC_` entry. The track ID
   and asset ID must line up (`asset = 0x1516 + id`), so new IDs start after
   `COMUSIC_AC_GOOD_ENDING`, i.e. asset `0x15C3`. That collides with anything
   else you appended at `0x15C3`, so **reserve the slots right after
   `0x15C2` for music** and put other new assets after them. `sMIDIAssets`
   (`src/core1/code_11AC0.c`) is allocated with `COMUSIC_NUM_TRACKS` entries,
   so adding enum values before `COMUSIC_NUM_TRACKS` grows it automatically.
3. Point the map music table (9.2) at it.

## 9.5 Sound effects

Sound effects use `enum sfx_e` (`SFX_2_CLAW_SWIPE`, `SFX_29_GRUBLIN_NYAHAHA`, …).

| Function | Use |
|---|---|
| `func_8030E6D4(SFX_xx)` | Play a non-positional sound |
| `func_8030E760(SFX_xx, pitch, volume)` | Non-positional with pitch (1.0 = normal) and volume (≤ 32767) |
| `func_8030E4E4(SFX_xx)` | Another non-positional variant, used by player moves |
| `sfx_playFadeShorthandDefault(SFX_xx, vol, sample_rate, pos, near, far)` | **Positional**: full volume within `near`, silent beyond `far` (macro in `functions.h`) |
| `sfxsource_createSfxsourceAndReturnIndex()` + `sfxsource_set*` | A persistent emitter attached to an actor, for loops and repeated sounds. See `src/core2/ch/beehive.c`. |

The sample rate argument doubles as pitch (Grublins use 25000 for their
laugh; lower it for a deeper voice). That's how one sample becomes several
characters' voices.

## 9.6 New samples and instruments

New sounds need the soundfont banks (`.ctl` = bank metadata, `.tbl` = ADPCM
sample data) rebuilt with an N64 audio bank tool, then dropped in as the
`soundfont*` binaries splat extracted. That's the most involved audio change.
Most mods get by with re-pitched existing samples and new MIDI arrangements.
