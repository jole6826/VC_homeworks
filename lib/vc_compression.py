import numpy as np
import unittest
import scipy.signal as sig
import scipy.fftpack as fft
import cv2
import warnings

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)

Q_C = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.float64)

### Subsampling, Transforms and Compression
def chromaSubsampling(src, N):
    """Downsamples a channel (matrix RxC) by N and returns a matrix that has dimensions
    (R/N) x (C/N)."""
    return src[::N, ::N]

def DCTcompression(src, N=8, k=4, component='luma'):
    """Applies DCT on NxN blocks of src image (RxC) and returns
    kxk coefficients of the transform. """
    if N < k:
        warnings.simplefilter('error',UserWarning)
        warnings.warn('N should be greater than k, i.e. blocksize has to be larger than compression factor.')

    [R, C] = src.shape
    T = fft.dct(np.eye(N), norm='ortho')
    out = np.zeros((int(R*k/N), int(C*k/N)))

    for r in range(0,int(R/N)):
        for c in range(0,int(C/N)):
            dctCoeffs = applyTransform(src[r*N:N*(r+1), c*N:N*(c+1)], T)
            if component == 'chroma':
                quantized = np.round(dctCoeffs / Q_C[0:N, 0:N])
                # quantized = np.round(dctCoeffs)
            else:
                quantized = np.round(dctCoeffs / Q[0:N, 0:N])
                # quantized = np.round(dctCoeffs)
            out[r*k:k*(r+1), c*k:k*(c+1)] = quantized[0:k,0:k]
    return out

def inverseDCTcompression(src, N=8, k=4, component='luma'):
    """Applies "inverse" compression. Generates full matrix with zero entries."""
    if N < k:
        warnings.simplefilter('error',UserWarning)
        warnings.warn('N should be greater than k, i.e. blocksize has to be larger than compression factor.')

    [R, C] = src.shape
    T = fft.dct(np.eye(N), norm='ortho')
    Tinv = np.linalg.inv(T)
    out = np.zeros((int(R/k*N), int(C/k*N)))

    for r in range(0,int(R/k)):
        for c in range(0,int(C/k)):
            dctCoeffs = np.zeros((N,N))
            dctCoeffs[0:k, 0:k] = src[r*k:(r+1)*k, c*k:(c+1)*k]
            if component == 'chroma':
                out[r * N:r * N + N, c * N:(c + 1) * N] = applyTransform(dctCoeffs * Q_C, Tinv)
                # out[r * N:r * N + N, c * N:(c + 1) * N] = applyTransform(dctCoeffs, Tinv)
            else:
                out[r*N:r*N+N, c*N:(c+1)*N] = applyTransform(dctCoeffs * Q, Tinv)
                # out[r * N:r * N + N, c * N:(c + 1) * N] = applyTransform(dctCoeffs, Tinv)
    return out

def applyTransform(src, T):
    """Applies a matrix block transform (NxN) on a block of size NxN
    and returns coefficients. """
    return np.dot(np.dot(np.transpose(T),src),T)

class TestVC_Lib(unittest.TestCase):
    def setUp(self):
        self.block1 = np.eye(8)
        self.T = np.ones((8,8))

    def test_applyTransform(self):
        self.assertEqual(applyTransform(self.block1,self.T).shape,
                                        self.block1.shape)

    def test_DCTcompression(self):
        self.assertEqual(DCTcompression(self.block1, N=8, k=4).shape, (4,4))
        self.assertEqual(DCTcompression(self.block1, N=8, k=8).shape, (8,8))
        self.assertEqual(DCTcompression(self.block1, N=8, k=6).shape, (6,6))

    def test_invDCTcompression(self):
        self.assertEqual(inverseDCTcompression(self.block1, N=8, k=4).shape, (16,16))
        self.assertEqual(inverseDCTcompression(self.block1, N=8, k=2).shape, (32,32))
        self.assertEqual(inverseDCTcompression(self.block1, N=8, k=8).shape, (8,8))


if __name__ == '__main__':
    unittest.main()
