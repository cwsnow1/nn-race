#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#include <raylib.h>

#include "nn.h"
#include "matrix.h"

#define CONTROL_SPEED   (1)
#define STEERING        (1)

#define NUM_CARS                (20)
#define SIGHT_DISTANCE          (100.0f)
#define EPSILON                 (0.2f)
#define MAX_STEPS               ((size_t) (SIGHT_DISTANCE / EPSILON))
#define SCREEN_WIDTH            (1920)
#define SCREEN_HEIGHT           (1080)
#if (STEERING == 1)
#define STEERING_INPUT_LAYER    (1)
#define W_MAX                   (3.0f)
#define W_MIN                   (-W_MAX)
#else
#define STEERING_INPUT_LAYER    (0)
#endif
#if (CONTROL_SPEED == 1)
#define SPEED_INPUT_LAYER       (1)
#define V_MIN                   (10.0)
#define V_MAX                   (60.0f)
#else
#define SPEED_INPUT_LAYER       (0)
#endif

#define NUM_RAYS                (19)
#define NUM_TRACKS              (6)
#define INPUT_LAYER_SIZE        (NUM_RAYS + STEERING_INPUT_LAYER + SPEED_INPUT_LAYER)
#define OUTPUT_LAYER_SIZE       (1 + SPEED_INPUT_LAYER)

typedef struct {
    Vector2 pos;
    float v;
    float dir;
    float w;
    float time;
    bool crashed;
    bool half_lap;
    bool finished;
} car_t;

typedef struct {
    float theta;
    float time;
    size_t index;
    bool finished;
} car_info_t;

typedef struct {
    nn_t *nn;
    car_t *car;
    pthread_mutex_t *lock;
} drive_thread_ctx_t;

typedef struct {
    nn_t** nns;
    car_t *cars;
    pthread_mutex_t *locks;
} gen_thread_ctx_t;

static Texture2D track;
static Color *track_colors;
static Image track_image;
static float time_elapsed = 0.0f;
static float dt;
static volatile bool next = false;

void calculate_distances(float *distances, car_t car) {
    float rayDirection = -(PI/2);
    for (size_t ray = 0; ray < NUM_RAYS; ++ray) {
        float rayAngle = car.dir + rayDirection;
        float x = car.pos.x;
        float y = car.pos.y;
        float dx = cosf(rayAngle) * EPSILON;
        float dy = sinf(rayAngle) * EPSILON;
        size_t steps = 0;
        for (; steps < MAX_STEPS; ++steps) {
            x += dx;
            y += dy;
            Color pixel_color = track_colors[(size_t) y * track_image.width + (size_t) x];
            if (pixel_color.r < 10) {
                break;
            }
        }
        distances[ray] = (steps * EPSILON) / SIGHT_DISTANCE;
        rayDirection += (PI/(float)(NUM_RAYS-1));
    }
}

Vector2 screen_to_cartesian(Vector2 pos) {
    pos.x = (2.0f * pos.x / SCREEN_WIDTH) - 1.0f;
    pos.y = (-2.0f * (pos.y / SCREEN_HEIGHT)) + 1.0f;
    return pos;
}

void init_cars(car_t *cars, size_t num_cars) {
    for (size_t i = 0; i < num_cars; ++i) {
        cars[i].pos.x = 1550;
        cars[i].pos.y = 550;
        cars[i].v = 50;
        cars[i].w = 0;
        cars[i].dir = -PI/2;
        cars[i].crashed = false;
        cars[i].half_lap = false;
        cars[i].finished = false;
    }
}

int cmp_info(const void *a, const void *b) {
    car_info_t *info_a = (car_info_t *)a;
    car_info_t *info_b = (car_info_t *)b;
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

void grade_car_performance(car_t *cars, size_t num_cars, size_t *sorted_indexes) {
    car_info_t *infos = (car_info_t *) malloc(sizeof(car_info_t) * num_cars);
    for (size_t i = 0; i < num_cars; ++i) {
        Vector2 pos = screen_to_cartesian(cars[i].pos);
        float theta = atan2f(pos.y, pos.x);
        infos[i].index = i;
        infos[i].theta = theta;
        if (cars[i].half_lap && theta < 0) {
            theta += (2*PI);
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

void *drive_car(void *ctx) {
    drive_thread_ctx_t *thread_ctx = (drive_thread_ctx_t*) ctx;
    nn_t *nn = thread_ctx->nn;
    car_t *car = thread_ctx->car;
    //pthread_mutex_t *lock = thread_ctx->lock;

    while (!(car->crashed || car->finished)) {
        calculate_distances(nn->layers[0].a->data, *car);
#if (CONTROL_SPEED == 1)
        nn->layers[0].a->data[nn->layers[0].a->rows - 1] = car->v / V_MAX;
#endif
#if (STEERING == 1)
        nn->layers[0].a->data[nn->layers[0].a->rows - 1 - SPEED_INPUT_LAYER] = car->w; /// W_MAX;
#endif
        nn_forward(nn);
#if (STEERING == 1)
        car->w += (nn->layers[nn->num_layers - 1].a->data[0]) * dt;
        if (car->w > W_MAX) {
            car->w = W_MAX;
        } else if (car->w < W_MIN) {
            car->w = W_MIN;
        }
#else
        car->w = (nn->layers[nn->num_layers - 1].a->data[0]);
#endif
#if (CONTROL_SPEED == 1)
        car->v += (nn->layers[nn->num_layers - 1].a->data[1]) * dt;
        if (car->v < V_MIN) {
            car->v = V_MIN;
        } else if (car->v > V_MAX) {
            car->v = V_MAX;
        }
#endif
        //pthread_mutex_lock(lock);
        car->pos.x += car->v * cosf(car->dir) * dt;
        car->pos.y += car->v * sinf(car->dir) * dt;
        //pthread_mutex_unlock(lock);
        Color pixel_color = track_colors[(size_t) car->pos.y * track_image.width + (size_t)car->pos.x];
        if (pixel_color.r < 10) {
            car->crashed = true;
        }
        if ((int)car->pos.x < SCREEN_WIDTH / 2) {
            car->half_lap = true;
        }
        if ((car->half_lap) &&
            ((int)car->pos.x > SCREEN_WIDTH / 2) &&
            ((int)car->pos.y < SCREEN_HEIGHT / 2)) {
            car->time = time_elapsed;
            car->finished = true;
        }
        car->dir += car->w * dt;
    }
    pthread_exit(NULL);
    return NULL;
}

void *gen_thread(void* c) {
    gen_thread_ctx_t *ctx = (gen_thread_ctx_t *)c;
    nn_t **nns = ctx->nns;
    car_t *cars = ctx->cars;
    pthread_mutex_t *locks = ctx->locks;
    drive_thread_ctx_t ctxs[NUM_CARS];
    pthread_t threads[NUM_CARS];
    for (;;) {
        while (next);
        init_cars(cars, NUM_CARS);
        time_elapsed = 0.0f;
        for (size_t i = 0; i < NUM_CARS; ++i) {
            ctxs[i].nn = nns[i];
            ctxs[i].car = &cars[i];
            ctxs[i].lock = &locks[i];
            pthread_create(&threads[i], NULL, drive_car, (void*) &ctxs[i]);
        }
        for (size_t i = 0; i < NUM_CARS; ++i) {
            pthread_join(threads[i], NULL);
        }
        size_t sorted_indexes[NUM_CARS];
        nn_t *new_nns[NUM_CARS];
        grade_car_performance(cars, NUM_CARS, sorted_indexes);
        for (size_t i = 0; i < NUM_CARS; ++i) {
            size_t half = NUM_CARS / 2;
            new_nns[i] = nn_average(nns[sorted_indexes[i/half]], nns[sorted_indexes[i%half]]);
        }
        for (size_t i = 0; i < NUM_CARS; ++i) {
            nn_delete(nns[i]);
            nns[i] = new_nns[i];
        }
        pthread_mutex_lock(&locks[0]);
        next = true;
        pthread_mutex_unlock(&locks[0]);
    }
}


int main(void) {
    srand(clock());

    nn_t *nns[NUM_CARS];
    size_t layer_sizes[5] = {INPUT_LAYER_SIZE, 512, 512, 256, OUTPUT_LAYER_SIZE};


    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "race");
    const int targetFPS = 144;
    SetTargetFPS(targetFPS);

    Image track_images[NUM_TRACKS];
    Color *track_colors_arr[NUM_TRACKS];
    Texture2D tracks[NUM_TRACKS];
    for (size_t i = 0; i < NUM_TRACKS; ++i) {
        char buffer[32];
        sprintf(buffer, "track%zu.png", i + 1);
        track_images[i] = LoadImage(buffer);
        track_colors_arr[i] = LoadImageColors(track_images[i]);
        tracks[i] = LoadTextureFromImage(track_images[i]);
    }
    int track_num = 0;
    track_image = track_images[track_num];
    track = tracks[track_num];
    track_colors = track_colors_arr[track_num];
    car_t cars[NUM_CARS];
    char buffer[256];
    pthread_mutex_t locks[NUM_CARS];
    for (size_t i = 0; i < NUM_CARS; ++i) {
        nns[i] = nn_init(4, layer_sizes);
        pthread_mutex_init(&locks[i], NULL);
    }
    pthread_t driver_thread;
    gen_thread_ctx_t ctx;
    ctx.cars = cars;
    ctx.nns = nns;
    ctx.locks = locks;
    pthread_create(&driver_thread, NULL, gen_thread, (void*) &ctx);
    dt = 1.0f / targetFPS;
    while(!WindowShouldClose()) {
        pthread_mutex_lock(&locks[0]);
        if (next) {
            track_num = rand() % NUM_TRACKS;
            track_image = track_images[track_num];
            track = tracks[track_num];
            track_colors = track_colors_arr[track_num];
            next = false;
        }
        pthread_mutex_unlock(&locks[0]);

        BeginDrawing();
        if (IsTextureReady(track)) {
            DrawTexture(track, 0, 0, WHITE);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Color color = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? WHITE : BLACK;
            Vector2 mouse_pos = GetMousePosition();
            int x = (int) mouse_pos.x;
            int y = (int) mouse_pos.y;
            ImageDrawCircle(&track_image, x, y, 10, color);
            UpdateTexture(track, track_image.data);
            UnloadImageColors(track_colors);
            track_colors = LoadImageColors(track_image);
        }
        bool kill = IsKeyPressed(KEY_SPACE);
        for (size_t i = 0; i < NUM_CARS; ++i) {
            if (kill) cars[i].crashed = true;
            Rectangle car_shape;
            car_shape.height = 10;
            car_shape.width = 10;
            car_shape.x = cars[i].pos.x-5;
            car_shape.y = cars[i].pos.y-5;
            DrawRectangleRec(car_shape, RED);
        }
        time_elapsed += dt;
        sprintf(buffer, "%.1f FPS", 1/dt);
        DrawText(buffer, 0, 900, 24, RED);
        EndDrawing();
    }

    return 0;
}
