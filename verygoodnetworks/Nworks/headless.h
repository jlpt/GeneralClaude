#pragma once

// Runs the simulation without a window and prints progress to the console.
//   --steps N    steps to run (default 1,000,000)
//   --report N   print a line every N steps (default 50,000)
//   --seed N     random seed (default: random)
//   --free-brains, --no-vision, --no-springs, --no-seasons   turn off new features
//   --organism-sensor, --position-sensor, --slices          turn on optional features
//   --save DIR   save every living organism's brain to DIR at the end (see brain_io.h)
//   --test FILE...  test saved organisms instead of evolving (see brain_test.h)
int RunHeadless(int argc, char** argv);

bool HasFlag(int argc, char** argv, const char* flag);
