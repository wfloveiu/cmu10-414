#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void mat_mul(const float *X, const float *Y, float *Z, size_t m, size_t k, size_t n)
{
    for(size_t i=0; i<m; i++)
        for(size_t j=0; j<n; j++)
        {
            Z[i*n+j] = 0;
            for(size_t t=0; t<k; t++)
                Z[i*n+j] += X[i*k+t] * Y[t*n+j];
        }
        
}
void softmax(float *Z, size_t batch, size_t k)
{
    float sum[batch];
    for(size_t i=0; i<batch; i++)
    {
        sum[i] = 0;
        for(size_t j=0; j<k; j++)
        {
            Z[i*k+j] = exp(Z[i*k+j]);
            sum[i] += Z[i*k+j];
        }
    }
    for(size_t i=0; i<batch; i++)
        for(size_t j=0; j<k; j++)
            Z[i*k+j] = Z[i*k+j]/sum[i];
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t b=0; b<m/batch; b++)
    {
        float *z = new float[batch*k];
        const float *x = &X[b*batch*n];
        mat_mul(x, theta, z, batch, n, k);
        softmax(z, batch, k);
        for(size_t i=0; i<batch; i++)
            z[i*k+y[b*batch+i]] -= 1;
        
        float *grad = new float[n*k];
        float *x_T = new float[n*batch];
        for(size_t i=0; i<n; i++)
            for(size_t j=0; j<batch; j++)
                x_T[i*batch+j] = x[j*n+i];
        mat_mul(x_T, z, grad, n, batch, k);
        for(size_t i=0; i<n*k; i++)
            theta[i] -= lr/batch*grad[i];

        delete[] z;
        delete[] grad;
        delete[] x_T;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
