// Headless stand-in for raylib, used only by the test harness.
// Instead of opening a window it:
//   * runs for SHIM_FRAMES frames (default 3600 = 60 s at 60 FPS),
//   * moves the "mouse" through the waypoints in SHIM_TARGETS ("x,y;x,y;..."),
//     switching every SHIM_TARGET_PERIOD frames (default 300),
//   * presses keys on given frames: SHIM_KEYS="frame:K,frame:K" (K = one letter),
//   * writes one CSV row per frame to SHIM_CSV (default stdout):
//       frame,mouse_x,mouse_y,org_x,org_y,left,right,up,down,active_frac,fired_frac
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <string>
#include <vector>
#include "cuda_runtime.h"

struct Color { unsigned char r, g, b, a; };
struct Vector2 { float x, y; };

#define LIGHTGRAY  Color{ 200, 200, 200, 255 }
#define GRAY       Color{ 130, 130, 130, 255 }
#define DARKGRAY   Color{ 80, 80, 80, 255 }
#define YELLOW     Color{ 253, 249, 0, 255 }
#define GOLD       Color{ 255, 203, 0, 255 }
#define ORANGE     Color{ 255, 161, 0, 255 }
#define PINK       Color{ 255, 109, 194, 255 }
#define RED        Color{ 230, 41, 55, 255 }
#define MAROON     Color{ 190, 33, 55, 255 }
#define GREEN      Color{ 0, 228, 48, 255 }
#define LIME       Color{ 0, 158, 47, 255 }
#define DARKGREEN  Color{ 0, 117, 44, 255 }
#define SKYBLUE    Color{ 102, 191, 255, 255 }
#define BLUE       Color{ 0, 121, 241, 255 }
#define DARKBLUE   Color{ 0, 82, 172, 255 }
#define PURPLE     Color{ 200, 122, 255, 255 }
#define VIOLET     Color{ 135, 60, 190, 255 }
#define DARKPURPLE Color{ 112, 31, 126, 255 }
#define BEIGE      Color{ 211, 176, 131, 255 }
#define BROWN      Color{ 127, 106, 79, 255 }
#define DARKBROWN  Color{ 76, 63, 47, 255 }
#define WHITE      Color{ 255, 255, 255, 255 }
#define BLACK      Color{ 0, 0, 0, 255 }
#define BLANK      Color{ 0, 0, 0, 0 }
#define MAGENTA    Color{ 255, 0, 255, 255 }

enum { KEY_SPACE = 32, KEY_P = 80, KEY_R = 82 };

struct ShimState {
    long frame = 0, frames = 3600, targetPeriod = 300;
    std::vector<Vector2> targets{ { 400, 300 } };
    std::vector<std::pair<long, int>> keys;
    FILE* csv = stdout;
    float orgX = 0, orgY = 0;
    float motor[4] = { 0, 0, 0, 0 };
    double activeFrac = 0, firedFrac = 0;
};
inline ShimState g_shim;

inline void shimOnDeviceToHost(const void* src, size_t bytes) {
    if (bytes < 4 * sizeof(float)) return;
    if (g_shimAllocs.size() >= 3 && src != g_shimAllocs[0].ptr) {
        // A small copy back to the host = the motor-signal readout for this frame.
        std::memcpy(g_shim.motor, src, sizeof(g_shim.motor));
        const float* act = (const float*)g_shimAllocs[0].ptr;
        const int* refr = (const int*)g_shimAllocs[1].ptr;
        const int* start = (const int*)g_shimAllocs[2].ptr;
        size_t n = g_shimAllocs[0].bytes / sizeof(float);
        size_t active = 0, fired = 0;
        for (size_t i = 0; i < n; ++i) {
            if (act[i] > 0.0f) ++active;
            if (refr[i] == start[i] && start[i] > 0) ++fired;
        }
        g_shim.activeFrac = (double)active / n;
        g_shim.firedFrac = (double)fired / n;
    }
}

inline void InitWindow(int, int, const char*) {
    if (const char* s = std::getenv("SHIM_FRAMES")) g_shim.frames = std::atol(s);
    if (const char* s = std::getenv("SHIM_TARGET_PERIOD")) g_shim.targetPeriod = std::atol(s);
    if (const char* s = std::getenv("SHIM_TARGETS")) {
        g_shim.targets.clear();
        std::string str(s);
        size_t pos = 0;
        while (pos < str.size()) {
            size_t end = str.find(';', pos);
            if (end == std::string::npos) end = str.size();
            float x, y;
            if (std::sscanf(str.substr(pos, end - pos).c_str(), "%f,%f", &x, &y) == 2) g_shim.targets.push_back({ x, y });
            pos = end + 1;
        }
    }
    if (const char* s = std::getenv("SHIM_KEYS")) {
        std::string str(s);
        size_t pos = 0;
        while (pos < str.size()) {
            size_t end = str.find(',', pos);
            if (end == std::string::npos) end = str.size();
            long f; char k;
            if (std::sscanf(str.substr(pos, end - pos).c_str(), "%ld:%c", &f, &k) == 2) g_shim.keys.push_back({ f, (int)k });
            pos = end + 1;
        }
    }
    if (const char* s = std::getenv("SHIM_CSV")) g_shim.csv = std::fopen(s, "w");
    std::fprintf(g_shim.csv, "frame,mouse_x,mouse_y,org_x,org_y,left,right,up,down,active_frac,fired_frac\n");
}
inline void SetTargetFPS(int) {}
inline bool WindowShouldClose() { return g_shim.frame >= g_shim.frames; }
inline void CloseWindow() { if (g_shim.csv != stdout) std::fclose(g_shim.csv); }
inline Vector2 GetMousePosition() {
    size_t i = (size_t)(g_shim.frame / g_shim.targetPeriod) % g_shim.targets.size();
    return g_shim.targets[i];
}
inline bool IsKeyPressed(int key) {
    for (auto& k : g_shim.keys) if (k.first == g_shim.frame && k.second == key) return true;
    return false;
}
inline void BeginDrawing() {}
inline void EndDrawing() {
    Vector2 m = GetMousePosition();
    std::fprintf(g_shim.csv, "%ld,%.1f,%.1f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f\n", g_shim.frame, m.x, m.y,
        g_shim.orgX, g_shim.orgY, g_shim.motor[0], g_shim.motor[1], g_shim.motor[2], g_shim.motor[3],
        g_shim.activeFrac, g_shim.firedFrac);
    ++g_shim.frame;
}
inline void ClearBackground(Color) {}
inline void DrawText(const char*, int, int, int, Color) {}
inline void DrawFPS(int, int) {}
inline void DrawLine(int, int, int, int, Color) {}
inline void DrawCircleLines(int, int, float, Color) {}
inline void DrawRectangle(int, int, int, int, Color) {}
// The organism is the only radius-16 circle; remember where it was drawn.
inline void DrawCircle(int x, int y, float r, Color) { if (r == 16) { g_shim.orgX = (float)x; g_shim.orgY = (float)y; } }
inline bool CheckCollisionPointCircle(Vector2 p, Vector2 c, float r) {
    float dx = p.x - c.x, dy = p.y - c.y; return dx * dx + dy * dy <= r * r;
}
inline const char* TextFormat(const char* fmt, ...) {
    static char buf[4][512]; static int idx = 0;
    idx = (idx + 1) % 4;
    va_list ap; va_start(ap, fmt); std::vsnprintf(buf[idx], sizeof(buf[idx]), fmt, ap); va_end(ap);
    return buf[idx];
}
