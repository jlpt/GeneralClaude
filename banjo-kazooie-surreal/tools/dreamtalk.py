#!/usr/bin/env python3
"""
dreamtalk.py - rewrite Banjo-Kazooie dialog for the Surreal mod.

Works on the YAML files bk_asset_tool extracts (assets/dialog/XXXX.dialog). Only the
spoken lines are replaced, in order; every control entry (close box, choices, triggers,
speaker portraits) is kept, so each conversation still plays out the same way.

usage: dreamtalk.py <pristine assets dir> <output assets dir>
"""
import os, re, sys

KEEP = None
JINJO = ["YIPPEE, YOU NOTICED ME! WE JINJOS ONLY EXIST WHILE SOMEBODY IS LOOKING. FIND ALL FIVE OF US ON EACH WORLD TO GET A JIGGY!"]

# asset id -> (bottom box lines, top box lines); KEEP leaves the original line alone
DREAMS = {
    # ---- opening cutscene: Gruntilda and Dingpot
    0x0E57: (["DINGPOT, DINGPOT, FULL OF RAIN,",
              "WHO DREAMT ME UP INSIDE THEIR BRAIN?",
              "A SLEEPING BEAR, FOUR HILLS AWAY,",
              "HE SNORES, AND YOU APPEAR...COUGH!",
              "THEN WHILE HE SLEEPS I'LL RULE HIS HEAD,",
              "AND HANG AN EYE ABOVE HIS BED!"], []),
    0x0E58: (["ERR...BUT HE ALSO DREAMS A GIRL...",
              "A SECOND DREAM? THAT CANNOT BE,",
              "THIS SKULL HAS ONLY ROOM FOR ME!",
              "WHY, IT'S TOOTY, BRIGHT AND SMALL,",
              "THE CLEAREST DREAM HE HAS AT ALL!",
              "NO NO NO, THE CLOCKS ARE WRONG,",
              "I'VE BEEN HIS NIGHTMARE ALL ALONG!",
              KEEP,
              "SHE'S THE ONE THING ON HIS MIND!"], []),
    0x0E59: (["WELL...WE'LL SEE WHO WAKES UP FIRST!"], []),
    0x0E5A: (["HI THERE TOOTY. IS IT TODAY AGAIN, OR STILL YESTERDAY?"],
             ["WHEN MY BIG LAZY BROTHER FALLS ASLEEP, THE ADVENTURE STARTS. HE NEVER REMEMBERS IT!"]),
    0x0E5B: ([], ["DON'T WAKE UP YET. I LIKE IT IN HERE..."]),
    0x0E5C: (["IF TOOTY SHINES MORE BRIGHT THAN ME,",
              "I'LL STEAL HER GLOW, AND GREY SHE'LL BE!"], []),
    0x0E5D: (["IS THAT AN EYE?", KEEP],
             ["IT'S ALWAYS BEEN THERE, MR. MOLE. DON'T LOOK BACK AT IT..."]),
    0x0E5E: ([], ["THAT'S NOT THE EYE. WHO IS THAT?"]),
    0x0E5F: (["COME WITH ME, MY LITTLE DREAM,",
              "NOTHING HERE IS WHAT IT SEEMS!"], []),
    0x0E60: ([], ["LET ME GO, YOU'RE NOT EVEN REAL!"]),
    0x0E61: (["DON'T PINCH YOURSELF, MY LITTLE BEAR,",
              "YOU CAN'T WAKE UP FROM ANYWHERE!"], []),
    0x0E62: (["OH NO, SHE'S GOT HER! SOMEBODY......WAKE HIM!"], []),
    0x0E63: ([], ["BANJO! WAKE UP......NO, WAIT, DON'T!"]),
    0x0E64: ([], ["YAWN...I DREAMT THE SKY WAS LOOKING AT ME."]),
    0x0E65: ([], ["IT STILL IS. LET'S GET OUTSIDE!"]),
    0x0E66: ([KEEP,
              "TURNS DREAMS TO LEAD AND LEAD TO GOLD!",
              KEEP,
              "THE STAIRS GO NOWHERE IN MY LAIR!",
              KEEP,
              "BEFORE HE WAKES AND ENDS THIS WITCH!",
              "YESSS MISSSTRESSS GRUNTY! THE CLOCKSSS ARE MELTING, SSSOON BE READY..."],
             ["LET ME GO! MY BROTHER WILL JUST DREAM",
              "OF SOMETHING ELSE!",
              "BANJO............KEEP SLEEPING!"]),

    # ---- Bottles in Spiral Mountain
    0x0DF3: (["LISTEN UP! I'M BOTTLES. I WAS A MOLE BEFORE THE SKY STARTED SPINNING.",
              "SURE IS A STRANGE LOOKING DREAM, BANJO. CAN IT TALK?",
              "THE WITCH GRUNTILDA FELL OUT OF THE EYE IN THE SKY AND GRABBED HER!",
              "SHE FLEW UP TO HER MOUNTAIN LAIR! IT WASN'T THERE YESTERDAY.",
              "NOTHING UP THERE HOLDS ITS SHAPE, SO YOU'LL PROBABLY NEED SOME TRAINING BEFORE YOU GO."],
             ["I'M BANJO, I THINK. AND THIS HERE'S MY BUDDY KAZOOIE.",
              KEEP,
              "WHY IS THE GRASS THE WRONG COLOR? AND WHERE'S MY SISTER TOOTY?",
              KEEP]),
    0x0DF4: (["HELLO AGAIN, OR FOR THE FIRST TIME! THE CAMERA CONTROLS ARE ON THE YELLOW C BUTTONS. THE CAMERA IS A LITTLE SEASICK TODAY.",
              KEEP,
              "NO...HOLD R TO KEEP THE CAMERA BEHIND YOUR FEATHERY BUTT! TRY IT OUT, I'LL SEE YOU AT THE NEXT MOLEHILL, IF IT'S STILL THERE."],
             ["ERRR...ANYONE AWAKE?", KEEP, KEEP]),
    0x0DF6: (["HMMM...GRAVITY ISN'T TRYING VERY HARD TODAY.", KEEP], [KEEP]),
    0x0DF9: (["THAT'S ALL THE JUMPS. TRY NOT TO DRIFT AWAY!"], []),
    0x0DFB: (["FANCY LEARNING TO SWIM UNDERWATER? THE WATER IS THE WRONG COLOR, BUT IT'S STILL WET.", KEEP],
             [KEEP, KEEP]),

    # ---- first meetings
    0x0D8F: (["ME MUMBO, ONLY SHAMAN STILL AWAKE. CAN HELP BEAR AND FILTHY FEATHERED ONE.",
              "MUMBO'S MAGIC TOKENS HID BY WITCH. FIND TOKENS AND MUMBO CHANGE YOUR SHAPE. SHAPE NOT REAL ANYWAY."],
             [KEEP]),
    0x0D96: (["GO AWAY! I'M DREAMING THESE HONEYCOMBS AS HARD AS I CAN!"], []),
    0x0D97: (JINJO, []), 0x0D98: (JINJO, []), 0x0D99: (JINJO, []), 0x0D9A: (JINJO, []), 0x0D9B: (JINJO, []),
    0x0D9C: (["I'M A NOTE FROM A SONG NOBODY FINISHED. THERE ARE A HUNDRED OF US ON EACH WORLD. COLLECT US TO OPEN THE NOTE DOORS!"], []),
    0x0D9D: (["ME MUMBO'S TOKEN. MUMBO DREAM OF TOKEN, SO TOKEN EXIST."], []),
    0x0D9E: (["WE'RE THE EGGS! NOBODY KNOWS WHAT'S INSIDE US. KAZOOIE CAN LEARN TO USE US AS AMMO!"], []),
    0x0D9F: (["WHEEE...I'M A RED FEATHER! I FELL OFF SOMETHING ENORMOUS. I HELP KAZOOIE FLY, WHEN SHE KNOWS HOW!"], []),
    0x0DA1: (["MMMM...I'M STICKY TASTY HONEY ENERGY! OR A VERY CONVINCING MEMORY OF IT!"], []),
    0x0DA3: (["LOOK OUT FOR ME, I'M AN EXTRA LIFE! SOMEBODY ELSE WAS USING IT."], []),

    # ---- Gruntilda's Lair
    0x0F66: (["THERE HE IS, STILL FAST ASLEEP,",
              "MY HALLS GROW LONGER WHILE YOU CREEP!"], []),
}

LINE = re.compile(r'^(\s*- \{ cmd: (0x[0-9A-Fa-f]{2}), string: ")(.*)("\}\s*)$')
ALLOWED = set(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def rewrite(text, bottom, top, uid):
    out, section, counters = [], None, {"bottom": 0, "top": 0}
    new = {"bottom": bottom, "top": top}
    for line in text.split("\n"):
        if line.startswith("bottom:"):
            section = "bottom"
        elif line.startswith("top:"):
            section = "top"
        m = LINE.match(line)
        # spoken lines have a speaker portrait command (>= 0x80) and some text
        if m and section and int(m.group(2), 16) >= 0x80 and m.group(3) != "":
            i = counters[section]
            counters[section] += 1
            if i >= len(new[section]):
                raise SystemExit("%04X: %s box has more spoken lines than the rewrite" % (uid, section))
            repl = new[section][i]
            if repl is not KEEP:
                bad = set(repl) - ALLOWED
                if bad or len(repl) > 250:
                    raise SystemExit("%04X: bad line %r (%s)" % (uid, repl, "".join(sorted(bad))))
                line = m.group(1) + repl + m.group(4)
        out.append(line)
    for section in ("bottom", "top"):
        if counters[section] != len(new[section]):
            raise SystemExit("%04X: %s box has %d spoken lines, rewrite has %d" % (uid, section, counters[section], len(new[section])))
    return "\n".join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    for uid, (bottom, top) in sorted(DREAMS.items()):
        name = "%04X.dialog" % uid
        text = open(os.path.join(src, "dialog", name)).read()
        open(os.path.join(dst, "dialog", name), "w").write(rewrite(text, bottom, top, uid))
    print("rewrote %d dialogs" % len(DREAMS))


if __name__ == "__main__":
    main()
