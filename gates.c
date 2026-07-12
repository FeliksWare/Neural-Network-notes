#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

float or_dataset[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float and_dataset[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

float nor_dataset[][3] = {
    {0, 0, 1},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 0},
};

float nand_dataset[][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

float xor_dataset[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

#define dataset or_dataset

float randf(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / max);
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float cost(float w1, float w2, float b) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = dataset[i][2];

        float a = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = a - y;
        result += d*d;
    }

    result /= (float)n;

    return result;
}

float gradient_cost_w1(float w1, float w2, float b) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = dataset[i][2];

        float a = sigmoidf(x1 * w1 + x2 * w2 + b);
        result += 2.0f * (a - y) * a * (1 - a) * x1;
    }

    result /= (float)n;

    return result;
}

float gradient_cost_w2(float w1, float w2, float b) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = dataset[i][2];

        float a = sigmoidf(x1 * w1 + x2 * w2 + b);
        result += 2.0f * (a - y) * a * (1 - a) * x2;
    }

    result /= (float)n;

    return result;
}

float gradient_cost_b(float w1, float w2, float b) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = dataset[i][2];

        float a = sigmoidf(x1 * w1 + x2 * w2 + b);
        result += 2.0f * (a - y) * a * (1 - a);
    }

    result /= (float)n;

    return result;
}

int main(void) {
    srand(time(0));

    float rate = 1e-1;

    float w1 = randf(0, 1);
    float w2 = randf(0, 1);
    float b = randf(0, 1);

    printf("inital cost: %f\n", cost(w1, w2, b));
    printf("inital w1: %f\n", w1);
    printf("inital w2: %f\n", w1);
    printf("inital b: %f\n", b);

    for (size_t i = 0; i < 100*1000; i++) {
        float gw1 = gradient_cost_w1(w1, w2, b);
        float gw2 = gradient_cost_w2(w1, w2, b);
        float gb = gradient_cost_b(w1, w2, b);

        w1 -= gw1 * rate;
        w2 -= gw2 * rate;
        b -= gb * rate;
        
        //printf("cost: %f\n", cost(w1, w2, b));
    }

    printf("\n");
    printf("final cost: %f\n", cost(w1, w2, b));
    printf("final w1: %f\n", w1);
    printf("final w2: %f\n", w1);
    printf("final b: %f\n", b);
    printf("\n");

    for (size_t i = 0; i < ARRAY_LENGTH(dataset); i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = dataset[i][2];

        float a = sigmoidf(x1 * w1 + x2 * w2 + b);
        printf("%f %f | %f | %f\n", x1, x2, y, a);
    }

    return 0;
}
