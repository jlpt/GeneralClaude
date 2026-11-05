#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include "slider.cpp"
#include "raylib.h"
#include "neural_network.h"
#include "organism.h"
#include "food_sensor.h"
#include "movement_module.h"
#include "slice_module.h"
#include "food.h"
#include "organism_sensor.h"
#include "slice.h"
#include "slice_sensor.h"
#include "health_sensor.h"
#include "player.h"
#include "bad_food.h"
#include "badfood_sensor.h"
#include "position_sensor.h"
#include "entity.h"
#include "air_sensor.h"


void createOrganism() {

}
std::string formatWithCommas(int value) {
    std::string num = std::to_string(value);
    int insertPosition = num.length() - 3;
    while (insertPosition > 0) {
        num.insert(insertPosition, ",");
        insertPosition -= 3;
    }
    return num;
}

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    int numOrganisms = 15;
    int spawnNeurons = 60;
    int spawnSynapses = 600;
    int inputNeurons = 13;
    int outputNeurons = 4;
    int worldsize = 1;
    std::uniform_int_distribution<> distrib_x(40, (700* worldsize));
    std::uniform_int_distribution<> distrib_y(80, 702* worldsize);
    std::uniform_int_distribution<> distrib_x2(40, (300 * worldsize));
    std::uniform_int_distribution<> distrib_y2(80, 702 * worldsize);
    Slider fpsSlider(120, 20, 1000, 60, 6000, 60);
    std::vector<Food> food;
    std::vector<BadFood> badfood;
    std::vector<Slice> slices;
   
    // Global vector for pending offspring if you later want to use reproduction.
    std::vector<std::unique_ptr<Organism>> pendingOffspring;

    InitWindow(1666, 768, "Neurons");
    SetTargetFPS(60);

    
    Organism* selectedOrganism = nullptr;
    std::vector<std::unique_ptr<Organism>> organisms;
   
    // Create some initial food items.
    for (int i = 0; i < 100; i++) {
        int rand_x = distrib_x(gen);
        int rand_y = distrib_y(gen);
        food.push_back(Food(rand_x, rand_y, 5));
    }

    for (int i = 0; i < 10; i++) {
        int rand_x = distrib_x(gen);
        int rand_y = distrib_y(gen);
        badfood.push_back(BadFood(rand_x, rand_y, 8));
    }

    // Create organisms as unique_ptr so their addresses remain constant.
   /* int rand_x = distrib_x(gen);
    int rand_y = distrib_y(gen);
    auto newOrganism = std::make_unique<Organism>(rand_x, rand_y, 12);

    // Initialize its neural network and add modules.

    newOrganism->InitNetwork(spawnNeurons, spawnSynapses, inputNeurons, outputNeurons);
    newOrganism->AddModule(std::make_unique<ModuleSensor>(newOrganism->network.get(), newOrganism.get(), &food));
    newOrganism->AddModule(std::make_unique<MovementModule>(newOrganism->network.get(), newOrganism.get()));
    newOrganism->AddModule(std::make_unique <SliceSensor>(newOrganism->network.get(), newOrganism.get(), &slices));
    newOrganism->AddModule(std::make_unique<OrganismSensor>(newOrganism->network.get(), newOrganism.get(), &organisms));
    newOrganism->AddModule(std::make_unique<HealthSensor>(newOrganism->network.get(), newOrganism.get()));

    newOrganism->AddModule(std::make_unique<SliceModule>(newOrganism->network.get(), newOrganism.get(), &slices));




    organisms.push_back(std::move(newOrganism));
  */
    Camera2D camera;
    Player player = Player(300, 300, 12, camera);
  
    while (!WindowShouldClose()) {
        // Spawn new food if needed.s
        static int frameCounter = 0;
        frameCounter++;

        camera.target = Vector2{ (float)player.x-(1666/2)-player.size/2,(float)player.y-(768/2)+player.size/2 };
        camera.offset = Vector2{ 0,0 };
        camera.zoom = .85f;
        camera.rotation = 0.0f;
        
      //  std::cout << slices.size() << '\n';
        fpsSlider.Update();
        for (auto it = slices.begin(); it != slices.end(); ) {
            it->Update();
            if (it->isDead) {
                it = slices.erase(it);
            }

            else {
                it->Draw();
                ++it;
            }
        }

       
            if (organisms.size() < numOrganisms) {
                int rand_x = distrib_x(gen);
                int rand_y = distrib_y(gen);
                auto newOrganism = std::make_unique<Organism>(rand_x, rand_y, 12);

                // Initialize its neural network and add modules.
                 
                newOrganism->InitNetwork(spawnNeurons, spawnSynapses, inputNeurons, outputNeurons);
               
               newOrganism->AddModule(std::make_unique<FoodSensor>(newOrganism->network.get(), newOrganism.get(), &food));
                
                newOrganism->AddModule(std::make_unique<BadFoodSensor>(newOrganism->network.get(), newOrganism.get(), &badfood));
                newOrganism->AddModule(std::make_unique<MovementModule>(newOrganism->network.get(), newOrganism.get()));

          //     newOrganism->AddModule(std::make_unique<PositionSensor>(newOrganism->network.get(), newOrganism.get()));
               // newOrganism->AddModule(std::make_unique<OrganismSensor>(newOrganism->network.get(), newOrganism.get(), &organisms));
             
              //  newOrganism->AddModule(std::make_unique <SliceSensor> (newOrganism->network.get(), newOrganism.get(), &slices));
              
                newOrganism->AddModule(std::make_unique<HealthSensor>(newOrganism->network.get(), newOrganism.get()));
                newOrganism->AddModule(std::make_unique<AirSensor>(newOrganism->network.get(), newOrganism.get()));

                 //   newOrganism->AddModule(std::make_unique<SliceModule>(newOrganism->network.get(), newOrganism.get(), &slices));
                
                


                organisms.push_back(std::move(newOrganism));
            }
        
       
        int maxGenTemp = organisms[0].get()->generation;
            if (maxGenTemp > 200) {
                maxGenTemp = 200;
            }
            int maxGenTempFood = organisms[0].get()->generation;
            if (maxGenTempFood > 800) {
                maxGenTempFood = 800;
            }

            if ((static_cast<float>(rand()) / RAND_MAX) < 0.015f && food.size() < 100) {
                int rand_x = distrib_x(gen) - (maxGenTempFood /2);
                if (rand_x < 0) {
                    rand_x = 0;
                }
                int rand_y = distrib_y(gen);
                food.push_back(Food(rand_x, rand_y, 5));

            }
        if ((static_cast<float>(rand()) / RAND_MAX) < 0.015f && badfood.size() < 50) {

            int rand_x = distrib_x2(gen)+ (maxGenTemp *5);
            int rand_y = distrib_y2(gen);
            badfood.push_back(BadFood(rand_x, rand_y, 8));

        }
       /* if ((static_cast<float>(rand()) / RAND_MAX) < 0.015f && badfood.size() < organisms[0].get()->generation / 100) {
            int rand_x = distrib_x(gen);
            int rand_y = distrib_y(gen);
            if (badfood.size() < 5) {
               badfood.push_back(BadFood(rand_x, rand_y, 8));
            }
          */

     //   }

        BeginDrawing();
        
        BeginMode2D(camera);
        DrawRectangle(0, 0, 1306, 768, BLACK);
        DrawRectangle(4, 4, 1298, 760, SKYBLUE);
        // Update each organism.
        std::vector<std::unique_ptr<Organism>> newOffspring;

        for (auto& o_ptr : organisms) {
            o_ptr->Update();
            for (const Slice& slice : slices) {
                if (slice.IsActive() && slice.CollidesWith(o_ptr->x, o_ptr->y, o_ptr->size)) {
                    o_ptr->alive = false;
                    break;
                }
            }

            // Reproduction: if an organism has eaten enough food, reproduce.
            if (o_ptr->foodEaten >= 3) {
                auto offspring = o_ptr->Reproduce();
                if (offspring) {
                    newOffspring.push_back(std::move(offspring));
                    //std::cout << "Organism reproduced! (pending offspring count: "
                     //   << newOffspring.size() << ")\n";
                    o_ptr->foodEaten = 0;
                }
            }
        }

        // After iterating, add new offspring to the main vector.
        for (auto& child : newOffspring) {
            organisms.push_back(std::move(child));
        }
        // Remove dead organisms.
        organisms.erase(
            std::remove_if(organisms.begin(), organisms.end(), [](const std::unique_ptr<Organism>& o) {
                return !o->alive;
                }),
            organisms.end()
        );
        if (selectedOrganism != nullptr) {
            bool stillValid = false;
            for (auto& o_ptr : organisms) {
                if (o_ptr.get() == selectedOrganism) {
                    stillValid = true;
                    break;
                }
            }
            if (!stillValid) {
                selectedOrganism = nullptr;
            }
        }

        // Check for mouse click and select an organism.
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            selectedOrganism = nullptr;
            for (auto& o_ptr : organisms) {
                Organism* o = o_ptr.get();
                if (CheckCollisionPointCircle(GetScreenToWorld2D(GetMousePosition(), camera),
                    Vector2{ static_cast<float>(o->x), static_cast<float>(o->y) }, o->size)) {


                    selectedOrganism = o;
                    o->network->GetSize();  // Display network size in console.
                    std::cout << "Inputs used: " << selectedOrganism->inputsUsed << '\n';
                    std::cout << "Outputs used: " << selectedOrganism->outputsUsed << '\n';
                    break;
                }
            }
        }

        ClearBackground(GRAY);
        
        // Optionally draw a rectangle panel (commented out below).
        // DrawRectangle(1320, 20, 280, 688, GRAY);

        // If an organism is selected, highlight it and draw its network.
       

      
        // Draw each organism.
       
       
        int maxGen = -1;

        Organism* o = organisms[0].get();
        for (auto& o_ptr : organisms) {
            
            if (o_ptr->generation > maxGen) {
                maxGen = o_ptr->generation;
                o = o_ptr.get();
                
            }
        }
      
        /* if (maxGen >= 500) {
            int spawnInterval;
            if (maxGen >= 6000) {
             //   spawnInterval = 7;
            }
            else if (maxGen >= 4000) {
               // spawnInterval = 15;
            }
            else if (maxGen >= 2000) {
                spawnInterval = 30;
            }
            else if (maxGen >= 1000) {
                spawnInterval = 60;
            }
            else {
                spawnInterval = 120;
            }

            if (frameCounter % spawnInterval == 0) {
                // Define random spawn boundaries
                int x1 = distrib_x(gen);
                int y1 = distrib_y(gen);
                int x2 = distrib_x(gen);
                int y2 = distrib_y(gen);
               // slices.push_back(Slice(x1, y1, x2, y2));
            }
        }*/
               
            
        
        // Draw additional info for the organism under the mouse.
        if (selectedOrganism != nullptr) {
            DrawCircle(selectedOrganism->x, selectedOrganism->y, selectedOrganism->size + 4, BLACK);

        }
       
        for (auto& o_ptr : organisms) {
            o_ptr->Draw();
        }
        if (o->network) {
            if (selectedOrganism == nullptr) {

                DrawCircle(o->x, o->y, o->size + 32, Color{o->color.r,o->color.g,o->color.b, 192});
                o->Draw();
                //o->DrawNetwork();
            }
        }
       
        // Draw food.
        for (Food& f : food) {
            f.Draw();
        }

        for (BadFood& f : badfood) {
            f.Draw();
        }
      
        player.Draw();
        player.Update();

        for (auto& o_ptr : organisms) {
            Organism* o = o_ptr.get();
            
            if (o->network->neuron_count % 2 == 0) {
             //   DrawText(TextFormat("%0d", o->network->neuron_count), o_ptr->x, o_ptr->y - 16, 20, DARKBLUE);
            }
            else {
             //   DrawText(TextFormat("%0d", o->network->neuron_count), o_ptr->x, o_ptr->y - 16, 20, WHITE);
            }
           

            if (CheckCollisionPointCircle(GetMousePosition(), Vector2{ static_cast<float>(o->x) - camera.target.x, static_cast<float>(o->y) - camera.target.y }, o->size)) {
                if (o->network) {
                    DrawText(TextFormat("Gen: %s", formatWithCommas(o->generation).c_str()),
                        GetMousePosition().x + 16 + camera.target.x,
                        GetMousePosition().y + 80 + camera.target.y, 20, WHITE);

                    DrawText(TextFormat("Health: %.0f", o->health),
                        GetMousePosition().x + 16 + camera.target.x,
                        GetMousePosition().y + 64 + camera.target.y, 20, WHITE);

                    DrawText(TextFormat("Neurons: %s", formatWithCommas(o->network->neuron_count).c_str()),
                        GetMousePosition().x + 16 + camera.target.x,
                        GetMousePosition().y + 48 + camera.target.y, 20, WHITE);

                    DrawText(TextFormat("Synapses: %s", formatWithCommas(o->network->synapse_count).c_str()),
                        GetMousePosition().x + 16 + camera.target.x,
                        GetMousePosition().y + 32 + camera.target.y, 20, WHITE);
                    const char* neuron_rank =
                        (o->network->neuron_count < 10) ? "Spark" :
                        (o->network->neuron_count < 25) ? "Cell" :
                        (o->network->neuron_count < 50) ? "Node" :
                        (o->network->neuron_count < 100) ? "Cluster" :
                        (o->network->neuron_count < 250) ? "Core" :
                        (o->network->neuron_count < 500) ? "Lobe" :
                        (o->network->neuron_count < 1000) ? "Complex" :
                        (o->network->neuron_count < 2500) ? "System" :
                        (o->network->neuron_count < 5000) ? "Cerebrum" :
                        (o->network->neuron_count < 10000) ? "Cortex" :
                        (o->network->neuron_count < 25000) ? "Overmind" :
                        (o->network->neuron_count < 50000) ? "Omnipresence" :
                        (o->network->neuron_count < 100000) ? "Transcendence" :
                        "Singularity";

                    const char* connection_rank =
                        (o->network->synapse_count < 100) ? "Sparse" :
                        (o->network->synapse_count < 500) ? "Linked" :
                        (o->network->synapse_count < 1000) ? "Chained" :
                        (o->network->synapse_count < 2500) ? "Integrated" :
                        (o->network->synapse_count < 5000) ? "Webbed" :
                        (o->network->synapse_count < 10000) ? "Latticed" :
                        (o->network->synapse_count < 50000) ? "Matrixed" :
                        (o->network->synapse_count < 100000) ? "Neurogrid" :
                        (o->network->synapse_count < 250000) ? "Hyperlinked" :
                        (o->network->synapse_count < 1000000) ? "Quantum Matrix" :
                        (o->network->synapse_count < 5000000) ? "Metalinked" :
                        (o->network->synapse_count < 10000000) ? "Entangled" :
                        "Infinite";


                    Color color =
                        (o->network->synapse_count < 250) ? WHITE :           // Sparse
                        (o->network->synapse_count < 500) ? BLUE :               // Linked
                        (o->network->synapse_count < 1000) ? GREEN :              // Chained
                        (o->network->synapse_count < 2500) ? DARKGREEN :          // Integrated
                        (o->network->synapse_count < 5000) ? DARKBLUE :           // Webbed
                        (o->network->synapse_count < 10000) ? PURPLE :            // Latticed
                        (o->network->synapse_count < 25000) ? GOLD :              // Matrixed
                        (o->network->synapse_count < 50000) ? ORANGE :            // Neurogrid
                        (o->network->synapse_count < 100000) ? MAROON :           // Hyperlinked
                        (o->network->synapse_count < 250000) ? LIME :             // Quantum Matrix
                        (o->network->synapse_count < 500000) ? SKYBLUE :          // Metalinked
                        (o->network->synapse_count < 1000000) ? VIOLET :          // Entangled
                        (o->network->synapse_count < 2500000) ? MAGENTA :         // New level
                        (o->network->synapse_count < 5000000) ? YELLOW :          // New level
                        (o->network->synapse_count < 10000000) ? RED :            // New level
                        BLACK;

                    DrawText(TextFormat("%s-%s", connection_rank, neuron_rank), GetMousePosition().x + 16 + camera.target.x, GetMousePosition().y + 16 + camera.target.y, 20, color);
                    break;
                }
                else {
                    std::cout << "No network found!";
                }
            }
        }
        
        EndMode2D();
        fpsSlider.Draw();
        std::string fpsText = "FPS: " + std::to_string(GetFPS());
        DrawText(fpsText.c_str(), 10, 10, 20, WHITE);
       
        if (selectedOrganism != nullptr) {
      
            selectedOrganism->DrawNetwork();
        }
        if (o->network) {
            if (selectedOrganism == nullptr) {
                
               // DrawCircle(o->x, o->y, o->size + 4, WHITE);
               
                  o->DrawNetwork();
                  
            }
            const char* neuron_rank =
                (o->network->neuron_count < 10) ? "Spark" :
                (o->network->neuron_count < 25) ? "Cell" :
                (o->network->neuron_count < 50) ? "Node" :
                (o->network->neuron_count < 100) ? "Core" :
                (o->network->neuron_count < 250) ? "Cluster" :
                (o->network->neuron_count < 500) ? "Lobe" :
                (o->network->neuron_count < 1000) ? "Conglomerate" :
                (o->network->neuron_count < 2500) ? "System" :
                (o->network->neuron_count < 5000) ? "Cerebrum" :
                (o->network->neuron_count < 10000) ? "Cortex" :
                (o->network->neuron_count < 25000) ? "Overmind" :
                (o->network->neuron_count < 50000) ? "Omnipresence" :
                (o->network->neuron_count < 100000) ? "Transcendence" :
                "Singularity";

            const char* connection_rank =
                (o->network->synapse_count < 250) ? "Sparse" :
                (o->network->synapse_count < 500) ? "Linked" :
                (o->network->synapse_count < 1000) ? "Chained" :
                (o->network->synapse_count < 2500) ? "Integrated" :
                (o->network->synapse_count < 5000) ? "Webbed" :
                (o->network->synapse_count < 10000) ? "Latticed" :
                (o->network->synapse_count < 50000) ? "Connected" :
                (o->network->synapse_count < 100000) ? "Meshed" :
                (o->network->synapse_count < 250000) ? "Hyperlinked" :
                (o->network->synapse_count < 1000000) ? "Quantic" :
                (o->network->synapse_count < 5000000) ? "Metalinked" :
                (o->network->synapse_count < 10000000) ? "Entangled" :
                "Infinite";


            Color color =
                (o->network->synapse_count < 250) ? WHITE :           // Sparse
                (o->network->synapse_count < 500) ? BLUE :               // Linked
                (o->network->synapse_count < 1000) ? GREEN :              // Chained
                (o->network->synapse_count < 2500) ? DARKGREEN :          // Integrated
                (o->network->synapse_count < 5000) ? DARKBLUE :           // Webbed
                (o->network->synapse_count < 10000) ? PURPLE :            // Latticed
                (o->network->synapse_count < 25000) ? GOLD :              // Matrixed
                (o->network->synapse_count < 50000) ? ORANGE :            // Neurogrid
                (o->network->synapse_count < 100000) ? MAROON :           // Hyperlinked
                (o->network->synapse_count < 250000) ? LIME :             // Quantum Matrix
                (o->network->synapse_count < 500000) ? SKYBLUE :          // Metalinked
                (o->network->synapse_count < 1000000) ? VIOLET :          // Entangled
                (o->network->synapse_count < 2500000) ? MAGENTA :         // New level
                (o->network->synapse_count < 5000000) ? YELLOW :          // New level
                (o->network->synapse_count < 10000000) ? RED :            // New level
                BLACK;                                            // Infinite
            DrawText(TextFormat("Mutation Rate: %.2fx", o->mutationRate), 1400, 80, 20, WHITE);
            DrawText(TextFormat("Gen: %s", formatWithCommas(o->generation).c_str()), 1400, 64, 20, WHITE);
            DrawText(TextFormat("Neurons: %s", formatWithCommas(o->network->neuron_count).c_str()), 1400, 48, 20, WHITE);
            DrawText(TextFormat("Synapses: %s", formatWithCommas(o->network->synapse_count).c_str()), 1400, 32, 20, WHITE);
            DrawText(TextFormat("Health: %s", formatWithCommas(o->health).c_str()), 1400, 96, 20, WHITE);
            DrawText(TextFormat("Air: %s", formatWithCommas(o->air).c_str()), 1400, 112, 20, WHITE);



            DrawText(TextFormat("%s-%s", connection_rank, neuron_rank), 1400, 16, 20, color);
            int totalNeurons = 0;
            int totalSynapses = 0;
            for (auto& o_ptr : organisms) {
                if (o_ptr->network) {
                    totalNeurons += o_ptr->network->neuron_count;
                    totalSynapses += o_ptr->network->synapse_count;
                }
            }
            neuron_rank =
                (totalNeurons < 10) ? "Spark" :
                (totalNeurons < 25) ? "Cell" :
                (totalNeurons < 50) ? "Node" :
                (totalNeurons < 100) ? "Core" :
                (totalNeurons < 250) ? "Cluster" :
                (totalNeurons < 500) ? "Lobe" :
                (totalNeurons < 1000) ? "Conglomerate" :
                (totalNeurons < 2500) ? "System" :
                (totalNeurons < 5000) ? "Cerebrum" :
                (totalNeurons < 10000) ? "Cortex" :
                (totalNeurons < 25000) ? "Overmind" :
                (totalNeurons < 50000) ? "Omnipresence" :
                (totalNeurons < 100000) ? "Transcendence" :
                "Singularity";

            connection_rank =
                (totalSynapses < 250) ? "Sparse" :
                (totalSynapses < 500) ? "Linked" :
                (totalSynapses < 1000) ? "Chained" :
                (totalSynapses < 2500) ? "Integrated" :
                (totalSynapses < 5000) ? "Webbed" :
                (totalSynapses < 10000) ? "Latticed" :
                (totalSynapses < 50000) ? "Connected" :
                (totalSynapses < 100000) ? "Meshed" :
                (totalSynapses < 250000) ? "Hyperlinked" :
                (totalSynapses < 1000000) ? "Quantic" :
                (totalSynapses < 5000000) ? "Metalinked" :
                (totalSynapses < 10000000) ? "Entangled" :
                "Infinite";


            color =
                (totalSynapses < 250) ? WHITE :           // Sparse
                (totalSynapses < 500) ? BLUE :               // Linked
                (totalSynapses < 1000) ? GREEN :              // Chained
                (totalSynapses < 2500) ? DARKGREEN :          // Integrated
                (totalSynapses < 5000) ? DARKBLUE :           // Webbed
                (totalSynapses < 10000) ? PURPLE :            // Latticed
                (totalSynapses < 25000) ? GOLD :              // Matrixed
                (totalSynapses < 50000) ? ORANGE :            // Neurogrid
                (totalSynapses < 100000) ? MAROON :           // Hyperlinked
                (totalSynapses < 250000) ? LIME :             // Quantum Matrix
                (totalSynapses < 500000) ? SKYBLUE :          // Metalinked
                (totalSynapses < 1000000) ? VIOLET :          // Entangled
                (totalSynapses < 2500000) ? MAGENTA :         // New level
                (totalSynapses < 5000000) ? YELLOW :          // New level
                (totalSynapses < 10000000) ? RED :            // New level
                BLACK;
            
            DrawText(TextFormat("Total Organisms: %s", formatWithCommas((int)organisms.size()).c_str()), 20, 730, 20, WHITE);
            DrawText(TextFormat("Total Neurons: %s", formatWithCommas(totalNeurons).c_str()), 260, 730, 20, WHITE);
            DrawText(TextFormat("Total Synapses: %s", formatWithCommas(totalSynapses).c_str()), 520, 730, 20, WHITE);

            DrawText(TextFormat("%s-%s", connection_rank, neuron_rank),780, 730, 20, color);
        }
        else {
            std::cout << "No network found!";
        }
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
