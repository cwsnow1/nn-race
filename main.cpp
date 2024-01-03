#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <thread>
#include <vector>
#include <sstream>
#include <string>

#include "raylib/include/raylib.h"

#include "nn.h"
#include "matrix.h"

constexpr bool controlSpeed = true;
constexpr bool steering = false;
constexpr bool drawRays = true;

constexpr size_t cNumCars = 50;
constexpr size_t cNumLayers = 5;
constexpr size_t cNumGroups = 1;
constexpr size_t cMaxNumGroups = 2;
constexpr float cSightDistance = 100.0f;
constexpr float cEpsilon = 0.3f;
constexpr size_t cMaxSteps = static_cast<size_t>(cSightDistance / cEpsilon);
constexpr int cScreenWidth = 1920;
constexpr int cScreenHeight = 1080;

constexpr size_t cSteeringInputLayer = steering ? 1 : 0;
constexpr float  cWMax = 3.0f;
constexpr float cWMin = -cWMax;

constexpr size_t cSpeedInputLayer = controlSpeed ? 1 : 0;
constexpr float cVMin = 10.0f;
constexpr float cVMax = 60.0f;


constexpr size_t cNumRays = 15;
constexpr size_t cNumTracks = 7;
constexpr size_t cInputLayerSize = cNumRays + cSteeringInputLayer + cSpeedInputLayer;
constexpr size_t cOutputLayerSize = 1 + cSpeedInputLayer;

struct car_t {
    Vector2 pos;
    std::vector<Vector2> rays;
    float v;
    float dir;
    float w;
    float time;
    bool crashed;
    bool half_lap;
    bool finished;
};

struct car_info_t {
    float theta;
    float time;
    size_t index;
    bool finished;
};

struct drive_thread_ctx_t {
    nn_t* nn;
    car_t* car;
};

struct gen_thread_ctx_t {
    nn_t** nns;
    car_t* cars;
};

static Texture2D track;
static Color* track_colors;
static Image track_image;
static float time_elapsed = 0.0f;
static float dt;
static volatile bool next = false;
static volatile bool quit = false;

void calculate_distances(float* distances, car_t& car) {
    float rayDirection = -(PI / 2);
    for (size_t ray = 0; ray < cNumRays; ++ray) {
        float rayAngle = car.dir + rayDirection;
        float x = car.pos.x;
        float y = car.pos.y;
        float dx = cosf(rayAngle) * cEpsilon;
        float dy = sinf(rayAngle) * cEpsilon;
        size_t steps = 0;
        for (; steps < cMaxSteps; ++steps) {
            x += dx;
            y += dy;
            Color pixel_color = track_colors[(size_t)y * track_image.width + (size_t)x];
            if (pixel_color.r < 10) {
                break;
            }
        }
        car.rays[ray].x = x;
        car.rays[ray].y = y;
        distances[ray] = (steps * cEpsilon) / cSightDistance;
        rayDirection += (PI / static_cast<float>(cNumRays - 1));
    }
}

Vector2 screen_to_cartesian(Vector2 pos) {
    pos.x = (2.0f * pos.x / cScreenWidth) - 1.0f;
    pos.y = (-2.0f * (pos.y / cScreenHeight)) + 1.0f;
    return pos;
}

void init_cars(car_t* cars, size_t num_cars) {
    for (size_t i = 0; i < num_cars; ++i) {
        cars[i].rays = std::vector<Vector2>(cNumRays);
        cars[i].pos.x = 1550;
        cars[i].pos.y = 550;
        cars[i].v = 50;
        cars[i].w = 0;
        cars[i].dir = -PI / 2;
        cars[i].crashed = false;
        cars[i].half_lap = false;
        cars[i].finished = false;
    }
}

int cmp_info(const void* a, const void* b) {
    car_info_t* info_a = (car_info_t*)a;
    car_info_t* info_b = (car_info_t*)b;
    if (info_a->finished && info_b->finished) {
        if (info_a->time < info_b->time) {
            return -1;
        }
        return 1;
    }
    if (info_a->finished) {
        return -1;
    }
    if (info_b->finished) {
        return 1;
    }
    if (info_a->theta < info_b->theta) {
        return 1;
    }
    return -1;
}

void grade_car_performance(car_t* cars, size_t num_cars, size_t* sorted_indexes) {
    car_info_t* infos = (car_info_t*)malloc(sizeof(car_info_t) * num_cars);
    for (size_t i = 0; i < num_cars; ++i) {
        Vector2 pos = screen_to_cartesian(cars[i].pos);
        float theta = atan2f(pos.y, pos.x);
        infos[i].index = i;
        infos[i].theta = theta;
        if (cars[i].half_lap && theta < 0) {
            theta += (2 * PI);
        }
        infos[i].finished = cars[i].finished;
        infos[i].time = cars[i].time;
    }
    qsort(infos, num_cars, sizeof(car_info_t), cmp_info);
    for (size_t i = 0; i < num_cars; ++i) {
        sorted_indexes[i] = infos[i].index;
    }
    free(infos);
}

void* drive_car(void* ctx) {
    drive_thread_ctx_t* thread_ctx = (drive_thread_ctx_t*)ctx;
    nn_t* nn = thread_ctx->nn;
    car_t* car = thread_ctx->car;

    while (!(car->crashed || car->finished)) {
        calculate_distances(nn->layers[0].a->data, *car);
        if constexpr (controlSpeed) {
            nn->layers[0].a->data[nn->layers[0].a->rows - 1] = car->v / cVMax;
        }
        if constexpr (steering) {
            nn->layers[0].a->data[nn->layers[0].a->rows - 1 - cSpeedInputLayer] = car->w;// W_MAX;
        }
        nn_forward(nn);
        if constexpr (steering) {
            car->w += (nn->layers[nn->num_layers - 1].a->data[0]) * dt;
            car->w = std::min(car->w, cWMax);
            car->w = std::max(car->w, cWMin);
        } else {
            car->w = (nn->layers[nn->num_layers - 1].a->data[0]);
        }
        if constexpr (controlSpeed) {
            car->v += (nn->layers[nn->num_layers - 1].a->data[1]) * dt;
            car->v = std::min(car->v, cVMax);
            car->v = std::max(car->v, cVMin);
        }
        car->pos.x += car->v * cosf(car->dir) * dt;
        car->pos.y += car->v * sinf(car->dir) * dt;
        Color pixel_color = track_colors[(size_t)car->pos.y * track_image.width + (size_t)car->pos.x];
        if (pixel_color.r < 10) {
            car->crashed = true;
        }
        if ((int)car->pos.x < cScreenWidth / 2) {
            car->half_lap = true;
        }
        if ((car->half_lap) &&
            ((int)car->pos.x > cScreenWidth / 2) &&
            ((int)car->pos.y < cScreenHeight / 2)) {
            car->time = time_elapsed;
            car->finished = true;
        }
        car->dir += car->w * dt;
    }
    return NULL;
}

void* gen_thread(void* c) {
    gen_thread_ctx_t* ctx = (gen_thread_ctx_t*)c;
    nn_t** nns = ctx->nns;
    car_t* cars = ctx->cars;
    drive_thread_ctx_t ctxs[cNumGroups][cNumCars];
    std::thread threads[cNumGroups][cNumCars];
    size_t iterations = 0;
    for (;;) {
        while (next);
        for (size_t group = 0; group < cNumGroups; ++group) {
            size_t group_offset = cNumCars * group;
            init_cars(cars + group_offset, cNumCars);
            time_elapsed = 0.0f;
            for (size_t i = 0; i < cNumCars; ++i) {
                ctxs[group][i].nn = nns[group_offset + i];
                ctxs[group][i].car = &cars[group_offset + i];
                threads[group][i] = std::thread(drive_car, (void*)&ctxs[group][i]);
            }
        }
        for (size_t group = 0; group < cNumGroups; ++group) {
            for (size_t i = 0; i < cNumCars; ++i) {
                threads[group][i].join();
            }
        }
        size_t sorted_indexes[cNumGroups][cNumCars];
        nn_t* new_nns[cNumGroups][cNumCars];
        for (size_t group = 0; group < cNumGroups; ++group) {
            size_t group_offset = cNumCars * group;
            grade_car_performance(cars + group_offset, cNumCars, sorted_indexes[group]);
            for (size_t i = 0; i < cNumCars; ++i) {
                size_t half = cNumCars / 2;
                new_nns[group][i] = nn_average(nns[group_offset + sorted_indexes[group][i / half]], nns[group_offset + sorted_indexes[group][i % half]]);
            }
            for (size_t i = 0; i < cNumCars; ++i) {
                nn_delete(nns[group_offset + i]);
                nns[group_offset + i] = new_nns[group][i];
            }
        }
        next = true;
        if (iterations++ > SIZE_MAX) {
            quit = true;
            break;
        }
    }
    return NULL;
}


int main(void) {
    srand(clock());

    Color group_colors[cMaxNumGroups] = { RED, GREEN };
    nn_t* nns[cNumGroups][cNumCars];
    size_t layer_sizes[cMaxNumGroups][cNumLayers] = {
        {cInputLayerSize, 128, 256, 256, cOutputLayerSize},
        {cInputLayerSize, 64, 256, 128, cOutputLayerSize},
    };


    InitWindow(cScreenWidth, cScreenHeight, "race");
    const int targetFPS = 144;
    SetTargetFPS(targetFPS);

    Image track_images[cNumTracks];
    Color* track_colors_arr[cNumTracks];
    Texture2D tracks[cNumTracks];
    for (size_t i = 0; i < cNumTracks; ++i) {
        std::ostringstream out;
        out << "track" << i+1 << ".png";
        track_images[i] = LoadImage(out.str().c_str());
        track_colors_arr[i] = LoadImageColors(track_images[i]);
        tracks[i] = LoadTextureFromImage(track_images[i]);
    }
    int track_num = 0;
    track_image = track_images[track_num];
    track = tracks[track_num];
    track_colors = track_colors_arr[track_num];
    car_t cars[cNumGroups][cNumCars];
    for (size_t group = 0; group < cNumGroups; ++group) {
        for (size_t i = 0; i < cNumCars; ++i) {
            nns[group][i] = nn_init(cNumLayers, layer_sizes[group]);
        }
    }

    gen_thread_ctx_t ctx;
    ctx.cars = (car_t*)cars;
    ctx.nns = (nn_t**)nns;
    //ctx.locks = locks;
    std::thread driver_thread = std::thread(gen_thread, (void*)&ctx);
    dt = 1.0f / targetFPS;
    while (!WindowShouldClose()) {
        if (next) {
            track_num = rand() % cNumTracks;
            track_image = track_images[track_num];
            track = tracks[track_num];
            track_colors = track_colors_arr[track_num];
            next = false;
        }
        if (quit) {
            break;
        }

        BeginDrawing();
        bool kill = IsKeyPressed(KEY_SPACE) || time_elapsed > 20.0f;
        if (IsTextureReady(track)) {
            DrawTexture(track, 0, 0, WHITE);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Color color = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? WHITE : BLACK;
            Vector2 mouse_pos = GetMousePosition();
            int x = (int)mouse_pos.x;
            int y = (int)mouse_pos.y;
            ImageDrawCircle(&track_image, x, y, 10, color);
            UpdateTexture(track, track_image.data);
            UnloadImageColors(track_colors);
            track_colors = LoadImageColors(track_image);
        }
        for (size_t group = 0; group < cNumGroups; ++group) {
            for (size_t i = 0; i < cNumCars; ++i) {
                if (kill) cars[group][i].crashed = true;
                Rectangle car_shape;
                car_shape.x = cars[group][i].pos.x - 5;
                car_shape.y = cars[group][i].pos.y - 5;
                car_shape.width = 10;
                car_shape.height = 10;
                if constexpr (drawRays) {
                    for (auto ray : cars[group][i].rays) {
                        if (ray.x == 0) continue;
                        DrawLine(static_cast<int>(cars[group][i].pos.x), static_cast<int>(cars[group][i].pos.y), static_cast<int>(ray.x), static_cast<int>(ray.y), GREEN);
                    }
                }
                DrawRectangleRec(car_shape, group_colors[group]);
            }
        }
        time_elapsed += dt;
        std::ostringstream fpsCounter;
        fpsCounter << 1/dt << " FPS";
        DrawText(fpsCounter.str().c_str(), 0, 900, 24, RED);
        EndDrawing();
    }

    driver_thread.join();
    //CloseWindow();

    return 0;
}
