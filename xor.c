#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct {
    float w1_11;
    float w1_21;
    float w1_12;
    float w1_22;

    float b1_1;
    float b1_2;

    float w2_11;
    float w2_12;

    float b2_1;
} Model;

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

float xnor_dataset[][3] = {
    {0, 0, 1},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

#define dataset xor_dataset

float randf(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / max);
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void debug_model(Model model) {
    printf("w1_11: %.12f\n", model.w1_11);
    printf("w1_21: %.12f\n", model.w1_21);
    printf("w1_12: %.12f\n", model.w1_12);
    printf("w1_22: %.12f\n", model.w1_22);

    printf("b1_1: %.12f\n", model.b1_1);
    printf("b1_2: %.12f\n", model.b1_2);

    printf("w2_11: %.12f\n", model.w2_11);
    printf("w2_12: %.12f\n", model.w2_12);

    printf("b2_1: %.12f\n", model.b2_1);
}

float cost(Model model) {
    float result = 0;

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float z1 = dataset[i][2];

        float a1_1 = sigmoidf(x1*model.w1_11 + x2*model.w1_12 + model.b1_1);
        float a1_2 = sigmoidf(x1*model.w1_21 + x2*model.w1_22 + model.b1_2);
        float a2_1 = sigmoidf(a1_1*model.w2_11 + a1_2*model.w2_12 + model.b2_1);

        float d = a2_1 - z1;
        result += d*d;
    }

    result /= (float)n;

    return result;
}

Model finite_difference(Model model, float epsilon) {
    Model gradient = {0};
    float c = cost(model);
    float temp;

    temp = model.w1_11;
    model.w1_11 += epsilon;
    gradient.w1_11 = (cost(model) - c) / epsilon;
    model.w1_11 = temp;

    temp = model.w1_21;
    model.w1_21 += epsilon;
    gradient.w1_21 = (cost(model) - c) / epsilon;
    model.w1_21 = temp;

    temp = model.w1_12;
    model.w1_12 += epsilon;
    gradient.w1_12 = (cost(model) - c) / epsilon;
    model.w1_12 = temp;

    temp = model.w1_22;
    model.w1_22 += epsilon;
    gradient.w1_22 = (cost(model) - c) / epsilon;
    model.w1_22 = temp;

    temp = model.b1_1;
    model.b1_1 += epsilon;
    gradient.b1_1 = (cost(model) - c) / epsilon;
    model.b1_1 = temp;

    temp = model.b1_2;
    model.b1_2 += epsilon;
    gradient.b1_2 = (cost(model) - c) / epsilon;
    model.b1_2 = temp;

    temp = model.w2_11;
    model.w2_11 += epsilon;
    gradient.w2_11 = (cost(model) - c) / epsilon;
    model.w2_11 = temp;

    temp = model.w2_12;
    model.w2_12 += epsilon;
    gradient.w2_12 = (cost(model) - c) / epsilon;
    model.w2_12 = temp;

    temp = model.b2_1;
    model.b2_1 += epsilon;
    gradient.b2_1 = (cost(model) - c) / epsilon;
    model.b2_1 = temp;

    return gradient;
}

Model gradient_cost(Model model) {
    Model gradient = {0};

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float z1 = dataset[i][2];

        float a1_1 = sigmoidf(x1*model.w1_11 + x2*model.w1_12 + model.b1_1);
        float a1_2 = sigmoidf(x1*model.w1_21 + x2*model.w1_22 + model.b1_2);
        float a2_1 = sigmoidf(a1_1*model.w2_11 + a1_2*model.w2_12 + model.b2_1);

        gradient.w2_11 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_1;
        gradient.w2_12 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_2;

        gradient.b2_1 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1);

        gradient.w1_11 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_1 * (1.0f - a1_1) * x1 * model.w2_11;
        gradient.w1_12 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_1 * (1.0f - a1_1) * x2 * model.w2_11;
        gradient.w1_21 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_2 * (1.0f - a1_2) * x1 * model.w2_12;
        gradient.w1_22 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_2 * (1.0f - a1_2) * x2 * model.w2_12;

        gradient.b1_1 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_1 * (1.0f - a1_1) * model.w2_11;
        gradient.b1_2 += 2.0f * (a2_1 - z1) * a2_1 * (1.0f - a2_1) * a1_2 * (1.0f - a1_2) * model.w2_12;
    }

    gradient.w2_11 /= (float)n;
    gradient.w2_12 /= (float)n;

    gradient.b2_1 /= (float)n;

    gradient.w1_11 /= (float)n;
    gradient.w1_21 /= (float)n;
    gradient.w1_12 /= (float)n;
    gradient.w1_22 /= (float)n;

    gradient.b1_1 /= (float)n;
    gradient.b1_2 /= (float)n;

    return gradient;
}

Model backpropagation(Model model) {
    Model gradient = {0};

    size_t n = ARRAY_LENGTH(dataset);
    for (size_t i = 0; i < n; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float z1 = dataset[i][2];

        float a1_1 = sigmoidf(x1*model.w1_11 + x2*model.w1_12 + model.b1_1);
        float a1_2 = sigmoidf(x1*model.w1_21 + x2*model.w1_22 + model.b1_2);
        float a2_1 = sigmoidf(a1_1*model.w2_11 + a1_2*model.w2_12 + model.b2_1);

        float C__a2_1 = 2.0f * (a2_1 - z1);
        float C__a1_1 = C__a2_1 * a2_1 * (1 - a2_1) * model.w2_11;
        float C__a1_2 = C__a2_1 * a2_1 * (1 - a2_1) * model.w2_12;

        gradient.w2_11 += C__a2_1 * a2_1 * (1 - a2_1) * a1_1;
        gradient.w2_12 += C__a2_1 * a2_1 * (1 - a2_1) * a1_2;

        gradient.b2_1 += C__a2_1 * a2_1 * (1 - a2_1);

        gradient.w1_11 += C__a1_1 * a1_1 * (1 - a1_1) * x1;
        gradient.w1_12 += C__a1_1 * a1_1 * (1 - a1_1) * x2;
        gradient.w1_21 += C__a1_2 * a1_2 * (1 - a1_2) * x1;
        gradient.w1_22 += C__a1_2 * a1_2 * (1 - a1_2) * x2;

        gradient.b1_1 += C__a1_1 * a1_1 * (1 - a1_1);
        gradient.b1_2 += C__a1_2 * a1_2 * (1 - a1_2);
    }

    gradient.w2_11 /= (float)n;
    gradient.w2_12 /= (float)n;

    gradient.b2_1 /= (float)n;

    gradient.w1_11 /= (float)n;
    gradient.w1_21 /= (float)n;
    gradient.w1_12 /= (float)n;
    gradient.w1_22 /= (float)n;

    gradient.b1_1 /= (float)n;
    gradient.b1_2 /= (float)n;

    return gradient;
}

Model gradient_descent(Model model, Model gradient, float rate) {
    model.w1_11 -= gradient.w1_11 * rate;
    model.w1_21 -= gradient.w1_21 * rate;
    model.w1_12 -= gradient.w1_12 * rate;
    model.w1_22 -= gradient.w1_22 * rate;

    model.b1_1 -= gradient.b1_1 * rate;
    model.b1_2 -= gradient.b1_2 * rate;

    model.w2_11 -= gradient.w2_11 * rate;
    model.w2_12 -= gradient.w2_12 * rate;

    model.b2_1 -= gradient.b2_1 * rate;

    return model;
}

int main(void) {
    srand(time(0));

    float rate = 1e-1;

    Model model = {0};

    model.w1_11 = randf(0, 1);
    model.w1_21 = randf(0, 1);
    model.w1_12 = randf(0, 1);
    model.w1_22 = randf(0, 1);

    model.b1_1 = randf(0, 1);
    model.b1_2 = randf(0, 1);

    model.w2_11 = randf(0, 1);
    model.w2_12 = randf(0, 1);

    model.b2_1 = randf(0, 1);

    printf("inital cost: %f\n", cost(model));

    for (size_t i = 0; i < 100 * 1000; i++) {
        //Model gradient = finite_difference(model, 1e-1);
        //Model gradient = gradient_cost(model);
        Model gradient = backpropagation(model);
        model = gradient_descent(model, gradient, rate);

        //printf("cost: %f\n", cost(model));
    }

    printf("final cost: %f\n", cost(model));

    for (size_t i = 0; i < ARRAY_LENGTH(dataset); i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float z1 = dataset[i][2];

        float a1_1 = sigmoidf(x1*model.w1_11 + x2*model.w1_12 + model.b1_1);
        float a1_2 = sigmoidf(x1*model.w1_21 + x2*model.w1_22 + model.b1_2);
        float a2_1 = sigmoidf(a1_1*model.w2_11 + a1_2*model.w2_12 + model.b2_1);

        printf("%f %f | %f | %f\n", x1, x2, z1, a2_1);
    }

    return 0;
}
