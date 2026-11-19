// Window app. Run with --headless (see headless.h) to simulate without drawing.
#include <cstdio>
#include <filesystem>
#include <string>
#include "brain_io.h"
#include "headless.h"
#include "raylib.h"
#include "render.h"
#include "slider.h"
#include "world.h"

namespace {

constexpr int kScreenWidth = 1666;
constexpr int kScreenHeight = 768;
constexpr Rectangle kBrainArea{1390.0f, 150.0f, 200.0f, 520.0f};

void PanCamera(Camera2D& camera) {
    const float speed = 4.0f / camera.zoom;
    if (IsKeyDown(KEY_W)) camera.target.y -= speed;
    if (IsKeyDown(KEY_S)) camera.target.y += speed;
    if (IsKeyDown(KEY_A)) camera.target.x -= speed;
    if (IsKeyDown(KEY_D)) camera.target.x += speed;
}

void PrintOrganism(const Organism& o) {
    std::printf("Organism #%llu: generation %d, %zu neurons, %zu synapses, brain uses %.1f KB\n",
                (unsigned long long)o.id, o.generation, o.brain.NeuronCount(), o.brain.SynapseCount(),
                double(o.brain.MemoryBytes()) / 1024.0);
}

// Saves an organism's brain to saved_brains/ (next to where the app was started).
void SaveFocused(const World& world, const Organism& o) {
    const std::filesystem::path dir = "saved_brains";
    std::filesystem::create_directories(dir);
    const std::string name = "step" + std::to_string(world.stats.steps) + "_gen" + std::to_string(o.generation) +
                             "_id" + std::to_string(o.id) + ".brain";
    const std::filesystem::path path = dir / name;
    if (SaveOrganism(o, world.cfg, path.string())) {
        std::printf("Saved organism #%llu to %s\n", (unsigned long long)o.id,
                    std::filesystem::absolute(path).string().c_str());
    } else {
        std::printf("Could not save to %s\n", path.string().c_str());
    }
}

int RunWindow() {
    World world{Config{}};

    InitWindow(kScreenWidth, kScreenHeight, "Nworks");
    SetTargetFPS(60);

    Slider fpsSlider(120, 20, 1000, 60, 6000, 60);
    Slider speedSlider(120, 45, 1000, 1, 100, 1);

    Camera2D camera{};
    camera.offset = Vector2{20.0f, 60.0f};
    camera.zoom = 0.85f;

    uint64_t selectedId = 0;

    while (!WindowShouldClose()) {
        if (fpsSlider.Update()) SetTargetFPS(fpsSlider.Value());
        speedSlider.Update();
        PanCamera(camera);

        for (int i = 0; i < speedSlider.Value(); ++i) world.Step();

        // Pointers into the world stay valid until the next Step().
        const Vector2 mouse = GetMousePosition();
        const bool overSlider = fpsSlider.Contains(mouse) || speedSlider.Contains(mouse);
        const Organism* hovered = overSlider ? nullptr : OrganismAt(world, GetScreenToWorld2D(mouse, camera));
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !overSlider) {
            selectedId = hovered ? hovered->id : 0;
            if (hovered) PrintOrganism(*hovered);
        }
        const Organism* selected = selectedId ? world.Find(selectedId) : nullptr;
        if (!selected) selectedId = 0;  // it died
        const Organism* leader = world.Leader();
        const Organism* focus = selected ? selected : leader;
        if (focus && IsKeyPressed(KEY_P)) SaveFocused(world, *focus);

        BeginDrawing();
        ClearBackground(GRAY);

        BeginMode2D(camera);
        DrawWorld(world, (!selected && leader) ? leader->id : 0, selectedId);
        EndMode2D();

        if (focus) {
            DrawBrain(focus->brain, kBrainArea);
            DrawOrganismStats(*focus, 1400, 16);
        }
        DrawPopulationStats(world, 20, 730);
        DrawWorldStats(world, 1400, 694);
        if (hovered) DrawTooltip(*hovered, mouse);

        fpsSlider.Draw(TextFormat("Target FPS: %d", fpsSlider.Value()));
        speedSlider.Draw(TextFormat("Steps per frame: %d", speedSlider.Value()));
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (HasFlag(argc, argv, "--headless")) return RunHeadless(argc, argv);
    return RunWindow();
}
