#ifndef _CMX_H
#define _CMX_H

int rownorm(float* a, unsigned ax, unsigned ay);

int matmul(float* a, unsigned ax, unsigned ay,
           float* b, unsigned bx, unsigned by, float* res);

int matmul_trans(float* a, unsigned ax, unsigned ay,
                 float* b, unsigned bx, unsigned by, float* res);

int nn_cosine_dist(float* a, unsigned ax, unsigned ay,
                   float* b, unsigned bx, unsigned by, float* res, unsigned is_normalized);

int nn_cosine_dist_ap(float* a, unsigned ax, unsigned ay,
                      float* b, unsigned bx, unsigned by, float* res, unsigned is_normalized);

#endif