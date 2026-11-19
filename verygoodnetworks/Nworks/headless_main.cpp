// Entry point for the window-free build (nworks_headless in CMake).
#include "headless.h"

int main(int argc, char** argv) { return RunHeadless(argc, argv); }
