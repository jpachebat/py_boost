import math
import numba
import cupy as cp
import numpy as np


loss_kernel2 = cp.ElementwiseKernel(
    """
    float32 grad, float32 hess, 
    float32 total_grad, float32 total_hess, 
    float32 nan_grad, float32 nan_hess, 
    uint64 nodes_count, 
    float32 lambda_l2, float32 min_data_in_left
    """,
    'raw float32 res',

    """
    float rG;
    float rH;
    float G = grad;
    float H = hess;
    float C = hess / total_hess * nodes_count;
    if (fmin(nodes_count - C, C) > min_data_in_left) {
        rG = total_grad - G;
        rH = total_hess - H;
        res[2 * i] = G * G / (H + lambda_l2) + rG * rG / (rH + lambda_l2);
    }
    if ((nan_hess > 0) and (hess < total_hess)) {
        C -= nan_hess / total_hess * nodes_count;
        if (fmin(nodes_count - C, C) > min_data_in_left) {
            G -= nan_grad;
            H -= nan_hess;
            rG = total_grad - G;
            rH = total_hess - H;
            res[2 * i + 1] = G * G / (H + lambda_l2) + rG * rG / (rH + lambda_l2);
        }
    }
    """,

    'loss_kernel')




loss_kernel3 = cp.ElementwiseKernel(
    """
    float32 grad, float32 hess, float32 grad3,
    float32 total_grad, float32 total_hess, float32 total_grad3,
    float32 nan_grad, float32 nan_hess, float32 nan_grad3,

    uint64 nodes_count, 
    float32 lambda_l2, float32 min_data_in_left


    """,
    'raw float32 res',

    """

    float rG;
    float rH;
    float rG3;
    float G = grad;
    float H = hess;
    float G3 = grad3;

    float lOmega;
    float rOmega;

    float lL;
    float rL;

    float C = hess / total_hess * nodes_count;


    if (fmin(nodes_count - C, C) > min_data_in_left) {
        rG = total_grad - G;
        rH = total_hess - H;
        rG3 = total_grad3 - G3;

        lOmega = - 2 * G * (H + lambda_l2) / (2 * (H + lambda_l2) * (H + lambda_l2) - G * G3);
        rOmega = - 2 * rG * (rH + lambda_l2) / (2 * (rH + lambda_l2) * (rH + lambda_l2) - rG * rG3);

        lL = lOmega * (G + lOmega * ((H + lambda_l2) / 2 + lOmega * (G3 / 6)));
        rL = rOmega * (rG + rOmega * ((rH + lambda_l2) / 2 + rOmega * (rG3 / 6)));

        res[2 * i] = - (lL + rL);
    }

    if ((nan_hess > 0) and (hess < total_hess)) {

        C -= nan_hess / total_hess * nodes_count;

        if (fmin(nodes_count - C, C) > min_data_in_left) {

            G -= nan_grad;
            H -= nan_hess;
            G3 -= nan_grad3;

            rG = total_grad - G;
            rH = total_hess - H;
            rG3 = total_grad3 - G3;
                
            lOmega = - 2 * G * (H + lambda_l2) / (2 * (H + lambda_l2) * (H + lambda_l2) - G * G3);
            rOmega = - 2 * rG * (rH + lambda_l2) / (2 * (rH + lambda_l2) * (rH + lambda_l2) - rG * rG3);

            lL = lOmega * (G + lOmega * ((H + lambda_l2) / 2 + lOmega * (G3 / 6)));
            rL = rOmega * (rG + rOmega * ((rH + lambda_l2) / 2 + rOmega * (rG3 / 6)));

            res[2 * i + 1] = - (lL + rL);
    

        }
    }

    """,

    'loss_kernel')



loss_kernel4 = cp.ElementwiseKernel(
    """
    float32 grad, float32 hess, float32 grad3, float32 grad4,
    float32 total_grad, float32 total_hess, float32 total_grad3, float32 total_grad4,
    float32 nan_grad, float32 nan_hess, float32 nan_grad3, float32 nan_grad4,

    uint64 nodes_count, 
    float32 lambda_l2_2, float32 lambda_l2_3, float32 lambda_l2_4, float32 min_data_in_left


    """,
    'raw float32 res',

    """

    float rG;
    float rH;
    float rG3;
    float rG4;
    float G = grad;
    float H = hess;
    float G3 = grad3;
    float G4 = grad4;

    float lOmega;
    float rOmega;

    float lL;
    float rL;

    float C = hess / total_hess * nodes_count;


    if (fmin(nodes_count - C, C) > min_data_in_left) {
        rG = total_grad - G;
        rH = total_hess - H;
        rG3 = total_grad3 - G3;
        rG4 = total_grad4 - G4;

        lOmega = - G * ((H + lambda_l2_2) * (H + lambda_l2_2) - G * (G3 + lambda_l2_3) / 2) / ((H + lambda_l2_2) * (H + lambda_l2_2) * (H + lambda_l2_2) - G * (H + lambda_l2_2) * (G3 + lambda_l2_3) + (G4 + lambda_l2_4) * G * G / 6);
        rOmega = - rG * ((rH + lambda_l2_2) * (rH + lambda_l2_2) - rG * (rG3 + lambda_l2_3) / 2) / ((rH + lambda_l2_2) * (rH + lambda_l2_2) * (rH + lambda_l2_2) - rG * rH * (rG3 + lambda_l2_3) + (rG4 + lambda_l2_4) * rG * rG / 6);

        lL = lOmega * (G + lOmega * ((H + lambda_l2_2) / 2 + lOmega * ((G3 + lambda_l2_3) / 6 + lOmega * (G4 + lambda_l2_4) / 24)));
        rL = rOmega * (rG + rOmega * ((rH + lambda_l2_2) / 2 + rOmega * ((rG3 + lambda_l2_3) / 6 + rOmega * (rG4 + lambda_l2_4) / 24)));

        res[2 * i] = - (lL + rL);
    }

    if ((nan_hess > 0) and (hess < total_hess)) {

        C -= nan_hess / total_hess * nodes_count;

        if (fmin(nodes_count - C, C) > min_data_in_left) {

            G -= nan_grad;
            H -= nan_hess;
            G3 -= nan_grad3;
            G4 -= nan_grad4;

            rG = total_grad - G;
            rH = total_hess - H;
            rG3 = total_grad3 - G3;
            rG4 = total_grad3 - G4;
                
            lOmega = - G * ((H + lambda_l2_2) * (H + lambda_l2_2) - G * (G3 + lambda_l2_3) / 2) / ((H + lambda_l2_2) * (H + lambda_l2_2) * (H + lambda_l2_2) - G * (H + lambda_l2_2) * (G3 + lambda_l2_3) + (G4 + lambda_l2_4) * G * G / 6);
            rOmega = - rG * ((rH + lambda_l2_2) * (rH + lambda_l2_2) - rG * (rG3 + lambda_l2_3) / 2) / ((rH + lambda_l2_2) * (rH + lambda_l2_2) * (rH + lambda_l2_2) - rG * rH * (rG3 + lambda_l2_3) + (rG4 + lambda_l2_4) * rG * rG / 6);

            lL = lOmega * (G + lOmega * ((H + lambda_l2_2) / 2 + lOmega * ((G3 + lambda_l2_3) / 6 + lOmega * (G4 + lambda_l2_4) / 24)));
            rL = rOmega * (rG + rOmega * ((rH + lambda_l2_2) / 2 + rOmega * ((rG3 + lambda_l2_3) / 6 + rOmega * (rG4 + lambda_l2_4) / 24)));

            res[2 * i + 1] = - (lL + rL);

        }
    }

    """,

    'loss_kernel')



loss_kernel3_minus = cp.ElementwiseKernel(
    """
    float32 grad, float32 hess, float32 grad3,
    float32 total_grad, float32 total_hess, float32 total_grad3, 
    float32 nan_grad, float32 nan_hess, float32 nan_grad3,
    uint64 nodes_count, 
    float32 lambda_l2, float32 min_data_in_left
    """,
    'raw float32 res',

    """
    float rG;
    float rH;
    float rG3;
    float G = grad;
    float H = hess;
    float G3 = grad3;

    float lOmega;
    float rOmega;

    float lL;
    float rL;

    float C = hess / total_hess * nodes_count;

    if (fmin(nodes_count - C, C) > min_data_in_left) {
        rG = total_grad - G;
        rH = total_hess - H;
        rG3 = total_grad3 - G3;

        lOmega = - H / G3 * (1 - sqrt(1 - 2 * G * G3 / (H + lambda_l2) / (H + lambda_l2)));
        rOmega = - rH / rG3 * (1 - sqrt(1 - 2 * rG * rG3 / (rH + lambda_l2) / (rH + lambda_l2)));

        lL = lOmega * (G + lOmega * ((H + lambda_l2) / 2 + lOmega * (G3 / 6)));
        rL = rOmega * (rG + rOmega * ((rH + lambda_l2) / 2 + rOmega * (rG3 / 6)));

        res[2 * i] = - (lL + rL);

    }
    if ((nan_hess > 0) and (hess < total_hess)) {
        C -= nan_hess / total_hess * nodes_count;
        if (fmin(nodes_count - C, C) > min_data_in_left) {
            G -= nan_grad;
            H -= nan_hess;
            G3 -= nan_grad3;

            rG = total_grad - G;
            rH = total_hess - H;
            rG3 = total_grad3 - G3;

            lOmega = - H / G3 * (1 - sqrt(1 - 2 * G * G3 / (H + lambda_l2) / (H + lambda_l2)));
            rOmega = - rH / rG3 * (1 - sqrt(1 - 2 * rG * rG3 / (rH + lambda_l2) / (rH + lambda_l2)));


            lL = lOmega * (G + lOmega * ((H + lambda_l2) / 2 + lOmega * (G3 / 6)));
            rL = rOmega * (rG + rOmega * ((rH + lambda_l2) / 2 + rOmega * (rG3 / 6)));

            res[2 * i + 1] = - (lL + rL);
        }
    }
    """,

    'loss_kernel')

