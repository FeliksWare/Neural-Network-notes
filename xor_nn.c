#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define ARRAY_LENGTH(array) (sizeof(array) / sizeof((array)[0]))

typedef struct {
    size_t rows;
    size_t columns;
    float *elements;
} Matrix;

#define MATRIX_AT(matrix, row, column) ((matrix).elements[(matrix).columns * (row) + (column)])

typedef struct {
    size_t layer_count;
    size_t *layer_sizes;

    Matrix *weights;
    Matrix *biases;
    Matrix *activations;
} NN;

#define NN_INPUT(nn) ((nn).activations[0])
#define NN_OUTPUT(nn) ((nn).activations[(nn).layer_count - 1])

float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + exp(-x));
}

Matrix new_matrix(size_t rows, size_t columns) {
    Matrix m;
    m.rows = rows;
    m.columns = columns;
    m.elements = malloc(sizeof(*m.elements) * rows * columns);
    assert(m.elements != NULL);

    return m;
}

void free_matrix(Matrix m) {
    free(m.elements);
}

void print_matrix(Matrix matrix) {
    for (size_t i = 0; i < matrix.rows; i++) {
        for (size_t j = 0; j < matrix.columns; j++) {
            printf("%f ", MATRIX_AT(matrix, i, j));
        }
        printf("\n");
    }
}

void matrix_random(Matrix m) {
    for (size_t i = 0; i < m.rows * m.columns; i++) {
        m.elements[i] = randf();
    }
}

void matrix_fill(Matrix m, float x) {
    for (size_t i = 0; i < m.rows * m.columns; i++) {
        m.elements[i] = x;
    }
}

void matrix_dot(Matrix dst, Matrix a, Matrix b) {
    assert(a.columns == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.columns == b.columns);

    assert(dst.elements != a.elements);
    assert(dst.elements != b.elements);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.columns; j++) {
            MATRIX_AT(dst, i, j) = 0;

            for (size_t k = 0; k < a.columns; k++) {
                MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
        }
    }
}

void matrix_add(Matrix dst, Matrix m) {
    assert(dst.rows == m.rows);
    assert(dst.columns == m.columns);

    for (size_t i = 0; i < dst.rows * dst.columns; i++) {
        dst.elements[i] += m.elements[i];
    }
}

void matrix_copy(Matrix dst, Matrix m) {
    assert(dst.rows == m.rows);
    assert(dst.columns == m.columns);

    for (size_t i = 0; i < dst.rows * dst.columns; i++) {
        dst.elements[i] = m.elements[i];
    }
}

Matrix row_slice(Matrix m, size_t row, size_t column, size_t length) {
    assert(row < m.rows);
    assert(column + length <= m.columns);

    Matrix slice;
    slice.rows = 1;
    slice.columns = length;
    slice.elements = &MATRIX_AT(m, row, column);

    return slice;
}

void matrix_map(Matrix m, float (*f)(float)) {
    for (size_t i = 0; i < m.rows * m.columns; i++) {
        m.elements[i] = f(m.elements[i]);
    }
}

NN new_nn(const size_t *layer_sizes, size_t layer_count) {
    assert(layer_count != 0);

    NN nn;

    nn.layer_count = layer_count;
    nn.layer_sizes = malloc(sizeof(*nn.layer_sizes) * layer_count);
    assert(nn.layer_sizes != NULL);

    nn.weights = malloc(sizeof(*nn.weights) * (layer_count - 1));
    assert(nn.weights != NULL);
    nn.biases = malloc(sizeof(*nn.biases) * (layer_count - 1));
    assert(nn.biases);
    nn.activations = malloc(sizeof(*nn.activations) * layer_count);
    assert(nn.activations);

    nn.layer_sizes[0] = layer_sizes[0];
    nn.activations[0] = new_matrix(1, layer_sizes[0]);

    for (size_t i = 1; i < layer_count; i++) {
        nn.layer_sizes[i] = layer_sizes[i];

        nn.weights[i - 1] = new_matrix(layer_sizes[i - 1], layer_sizes[i]);
        nn.biases[i - 1] = new_matrix(1, layer_sizes[i]);
        nn.activations[i] = new_matrix(1, layer_sizes[i]);
    }

    return nn;
}

void free_nn(NN nn) {

    free_matrix(nn.activations[0]);
    for (size_t l = 1; l < nn.layer_count; l++) {
        free_matrix(nn.weights[l - 1]);
        free_matrix(nn.biases[l - 1]);
        free_matrix(nn.activations[l]);
    }

    free(nn.layer_sizes);
    free(nn.weights);
    free(nn.biases);
    free(nn.activations);
}

void print_nn(NN nn) {
    printf("weights:\n");
    for (size_t l = 0; l < nn.layer_count - 1; l++) {
        print_matrix(nn.weights[l]);
        printf("\n");
    }

    printf("biases:\n");
    for (size_t l = 0; l < nn.layer_count - 1; l++) {
        print_matrix(nn.biases[l]);
        printf("\n");
    }

    printf("activations:\n");
    for (size_t l = 0; l < nn.layer_count; l++) {
        print_matrix(nn.activations[l]);
        printf("\n");
    }
}

void nn_random(NN nn) {
    for (size_t i = 0; i < nn.layer_count - 1; i++) {
        matrix_random(nn.weights[i]);
        matrix_random(nn.biases[i]);
    }
}

void nn_fill(NN nn, float x) {
    matrix_fill(nn.activations[0], x);
    for (size_t i = 1; i < nn.layer_count; i++) {
        matrix_fill(nn.weights[i - 1], x);
        matrix_fill(nn.biases[i - 1], x);
        matrix_fill(nn.activations[i], x);
    }
}

void nn_feedforward(NN nn) {
    for (size_t i = 0; i < nn.layer_count - 1; i++) {
        matrix_dot(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
        matrix_add(nn.activations[i + 1], nn.biases[i]);
        matrix_map(nn.activations[i + 1], sigmoidf);
    }
}

float nn_cost(NN nn, Matrix dataset) {
    assert(NN_INPUT(nn).columns + NN_OUTPUT(nn).columns == dataset.columns);

    size_t n = dataset.rows;
    float cost = 0;

    for (size_t i = 0; i < n; i++) {
        Matrix input = row_slice(dataset, i, 0, NN_INPUT(nn).columns);
        Matrix output = row_slice(dataset, i, NN_INPUT(nn).columns, NN_OUTPUT(nn).columns);

        matrix_copy(NN_INPUT(nn), input);
        nn_feedforward(nn);

        for (size_t j = 0; j < output.columns; j++) {
            float d = MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(output, 0, j);
            cost += d*d;
        }
    }

    cost /= (float)n;

    return cost;
}

void nn_backpropagate(NN nn, NN gradient, Matrix dataset) {
    assert(NN_INPUT(nn).columns + NN_OUTPUT(nn).columns == dataset.columns);

    assert(nn.layer_count == gradient.layer_count);
    for (size_t l = 0; l < nn.layer_count; l++) {
        assert(nn.layer_sizes[l] == gradient.layer_sizes[l]);
    }

    nn_fill(gradient, 0);

    size_t n = dataset.rows;

    for (size_t i = 0; i < n; i++) {
        Matrix input = row_slice(dataset, i, 0, NN_INPUT(nn).columns);
        Matrix output = row_slice(dataset, i, NN_INPUT(nn).columns, NN_OUTPUT(nn).columns);

        matrix_copy(NN_INPUT(nn), input);
        nn_feedforward(nn);

        for (size_t l = 0; l < gradient.layer_count; l++) {
            matrix_fill(gradient.activations[l], 0);
        }

        for (size_t j = 0; j < NN_OUTPUT(nn).columns; j++) {
            MATRIX_AT(NN_OUTPUT(gradient), 0, j) = 2.0f * (MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(output, 0, j));
        }

        for (size_t l = nn.layer_count - 1; l > 0; l--) {
            for (size_t j = 0; j < nn.activations[l].columns; j++) {

                float aj = MATRIX_AT(nn.activations[l], 0, j);
                float g_aj = MATRIX_AT(gradient.activations[l], 0, j);

                MATRIX_AT(gradient.biases[l - 1], 0, j) += g_aj * aj * (1.0f - aj);

                for (size_t k = 0; k < nn.activations[l - 1].columns; k++) {
                    float ak = MATRIX_AT(nn.activations[l - 1], 0, k);
                    float wkj = MATRIX_AT(nn.weights[l - 1], k, j);

                    MATRIX_AT(gradient.activations[l - 1], 0, k) += g_aj * aj * (1.0f - aj) * wkj;
                    MATRIX_AT(gradient.weights[l - 1], k, j) += g_aj * aj * (1.0f - aj) * ak;
                }
            }
        }
    }

    for (size_t l = 0; l < gradient.layer_count - 1; l++) {
        for (size_t i = 0; i < gradient.weights[l].rows * gradient.weights[l].columns; i++) {
            gradient.weights[l].elements[i] /= (float)n;
        }

        for (size_t i = 0; i < gradient.biases[l].columns; i++) {
            gradient.biases[l].elements[i] /= (float)n;
        }
    }
}


void nn_gradient_descent(NN nn, NN gradient, float rate) {
    assert(nn.layer_count == gradient.layer_count);
    for (size_t l = 0; l < nn.layer_count; l++) {
        assert(nn.layer_sizes[l] == gradient.layer_sizes[l]);
    }

    for (size_t l = 0; l < nn.layer_count - 1; l++) {
        for (size_t i = 0; i < nn.weights[l].rows * nn.weights[l].columns; i++) {
            nn.weights[l].elements[i] -= gradient.weights[l].elements[i] * rate;
        }

        for (size_t i = 0; i < nn.biases[l].columns; i++) {
            nn.biases[l].elements[i] -= gradient.biases[l].elements[i] * rate;
        }
    }
}

int main(void) {
    srand(time(0));

    float rate = 1e-1;

    Matrix xor_dataset = new_matrix(4, 3);

    for (int i = 0; i < 4; i++) {
        MATRIX_AT(xor_dataset, i, 0) = i / 2;
        MATRIX_AT(xor_dataset, i, 1) = i % 2;
        MATRIX_AT(xor_dataset, i, 2) = (i / 2) ^ (i % 2);
    }

    static const size_t arch[] = {2, 2, 1};
    NN nn = new_nn(arch, ARRAY_LENGTH(arch));
    NN gradient = new_nn(arch, ARRAY_LENGTH(arch));

    nn_random(nn);

    printf("Initial cost: %f\n", nn_cost(nn, xor_dataset));

    for (size_t i = 0; i < 100 * 1000; i++) {
        nn_backpropagate(nn, gradient, xor_dataset);
        nn_gradient_descent(nn, gradient, rate);
        //printf("cost: %f\n", nn_cost(nn, xor_dataset));
    }

    printf("Final cost: %f\n", nn_cost(nn, xor_dataset));

    for (size_t i = 0; i < xor_dataset.rows; i++) {
        Matrix input = row_slice(xor_dataset, i, 0, NN_INPUT(nn).columns);

        matrix_copy(NN_INPUT(nn), input);
        nn_feedforward(nn);

        printf("%f %f | %f | %f\n", 
            MATRIX_AT(xor_dataset, i, 0),
            MATRIX_AT(xor_dataset, i, 1),
            MATRIX_AT(xor_dataset, i, 2),
            MATRIX_AT(NN_OUTPUT(nn), 0, 0)
        );
    }

    free_nn(nn);
    free_nn(gradient);
    free_matrix(xor_dataset);

    return 0;
}
