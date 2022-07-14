#include "ap.h"
#include "ap_fixed.h"

#define BITS 16

#define APTYPE ap_fixed<BITS, 1>

void float_to_ap(float* a, APTYPE* o, unsigned len) {
    for (int i=0; i<len; i++) {
        o[i] = a[i];
    }
}

void ap_to_float(APTYPE* a, float* o, unsigned len) {
    for (int i=0; i<len; i++) {
        o[i] = a[i].to_float();
    }
}

/**
 * @brief Multiply a matrix with another one's transpose (a*bT)
 *
 * @param a left matrix
 * @param ax rows of a
 * @param ay columns of a
 * @param b right matrix
 * @param bx rows of b
 * @param by columns of b
 * @param res buffer for result. Should be at least ax*bx in size
 * @return int 0 on success
 */
int matmul_trans(APTYPE* a, unsigned ax, unsigned ay, APTYPE* b, unsigned bx, unsigned by, APTYPE* res) {
    if (ay != by) {
        return -1;
    }
    // output size
    unsigned rx = ax;
    unsigned ry = bx;

    // clear buffer
    memset(res, 0, rx*ry*sizeof(APTYPE));

    unsigned i, j, k, ri, ai, bi;

    for (i = 0; i < rx; i++) {
        for( j = 0; j < ry; j++) {
            ri = i*ry + j;
            for (k=0; k < ay; k++) {
                ai = i*ay + k;
                bi = j*by + k;
                res[ri] += a[ai] * b[bi];   // res[i][j] += a[i][k] * b[j][k]
            }
        }
    }
    return 0;
}

/**
 * @brief Normalize each row of a matrix using the Frobenius norm - sqrt(sum()^2)
 *
 * @param a matrix
 * @param ax number of rows
 * @param ay number of columns
 * @return int N/A
 */
int rownorm(float* a, unsigned ax, unsigned ay) {
    unsigned i,j;
    for (i = 0; i < ax; i++) {
        float sum=0;
        for (j = 0; j < ay; j++) {
            float v = a[i*ay + j];
            sum += v*v;
        }
        float norm = sqrt(sum);

        for (j = 0; j < ay; j++) {
            a[i*ay + j] /= norm;
        }
    }

    return 0;
}

/*
 * Normalize, then 1-a*bT
 */
int ap_nn_cosine_dist(float* a, unsigned ax, unsigned ay,
                      float* b, unsigned bx, unsigned by,
                      float* res, unsigned is_normalized) {
    if (!is_normalized) {
        rownorm(a, ax, ay);
        rownorm(b, bx, by);
    }
    // output size
    unsigned rx = ax;
    unsigned ry = bx;


    APTYPE* c = new APTYPE[ax*ay];
    APTYPE* d = new APTYPE[bx*by];
    APTYPE* apmul = new APTYPE[rx*ry];

    float_to_ap(a, c, ax*ay);
    float_to_ap(b, d, bx*by);

    int err = matmul_trans(c, ax, ay, d, bx, by, apmul);

    float* mul = (float*)malloc(rx*ry*sizeof(float));
    ap_to_float(apmul, mul, rx*ry);

    delete[] c;
    delete[] d;
    delete[] apmul;


    if (err) {
        return err;
    }

    unsigned i, j , ri;
    for (i = 0; i < rx; i++) {
        for (j = 0; j < ry; j++) {
            ri = i*ry + j;
            mul[ri] = 1 - mul[ri];
        }
    }

    for (i=0; i<ry; i++) {
        float min = mul[i];

        for (j=1; j<rx; j++) {
            ri = j*ry + i;
            if (mul[ri] < min) {
                min = mul[ri];
            }
        }

        res[i] = min;
    }

    free(mul);

    return 0;
}

