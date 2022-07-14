#ifndef _AP_H
#define _AP_H

#ifdef __cplusplus
extern "C" {
#endif

float mul(float a, float b);
int ap_nn_cosine_dist(float* a, unsigned ax, unsigned ay,
                   float* b, unsigned bx, unsigned by,
                   float* res, unsigned is_normalized);

#ifdef __cplusplus
}
#endif

#endif