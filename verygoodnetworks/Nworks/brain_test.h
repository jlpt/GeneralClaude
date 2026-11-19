#pragma once

// Tests saved organisms (see brain_io.h). Started by `--test FILE...`.
//
// 1. Choice test. The organism starts in an empty world with one food and one
//    spring the same distance away, in opposite directions, at a random
//    angle. It runs until it touches one of them. Each angle is run twice:
//    once hungry (20% health, full air) and once breathless (full health, 35%
//    air). Only the hunger and breathlessness inputs differ between the two,
//    so any difference in what it picks comes from those inputs.
//
// 2. Live test. Copies of the organism live in a normal world (no breeding,
//    no refills) until they die. Run once as they are and once with the
//    hunger and breathlessness inputs frozen at their "well fed, full air"
//    values. If they need those inputs, the frozen copies die sooner.
//
// Options:
//   --trials N       angles in the choice test (default 40)
//   --distance D     food and spring distance in px (default 120)
//   --copies N       organisms in the live test (default 20)
//   --difficulty D   live-test difficulty, 0..1 (default 0.2)
//   --seed N         seed for the angles and the live world (default 1)
//   --save-best DIR  copy organisms that pass the choice test into DIR, with their scores
int RunBrainTest(int argc, char** argv);
