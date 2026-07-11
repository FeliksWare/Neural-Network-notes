#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

float dataset[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
};

float randf(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / max);
}

float cost(float w) {
    float result = 0;

    for (size_t i = 0; i < ARRAY_LENGTH(dataset); i++) {
        float x = dataset[i][0];
        float y = dataset[i][1];

        float d = x*w - y;
        result += d*d;
    }

    return result;
}

float gradient_cost_w(float w) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x = dataset[i][0];
        float y = dataset[i][1];

        result += 2*(x*w - y)*x;
    }

    result /= (float)n;

    return result;
}

int main(void) {
    srand(time(0));

    float rate = 1e-1;

    float w = randf(-10, 10);

    printf("initial cost: %f\n", cost(w));
    printf("initial w: %f\n", w);
    printf("\n");

    for (size_t i = 0; i < 40; i++) {
        w -= rate * gradient_cost_w(w);
        printf("cost: %f\n", cost(w));
    }

    printf("\n");
    printf("final cost: %f\n", cost(w));
    printf("final w: %f\n", w);

    return 0;
}
