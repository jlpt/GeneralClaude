#pragma once

// Runs the simulation without a window and prints progress to the console.
//   --steps N    steps to run (default 1,000,000)
//   --report N   print a line every N steps (default 50,000)
//   --seed N     random seed (default: random)
//   --organism-sensor, --position-sensor, --slices   turn on optional features
int RunHeadless(int argc, char** argv);

bool HasFlag(int argc, char** argv, const char* flag);
