import numpy as np
import unittest
import itertools
import scipy.signal as sig
import scipy.fftpack as fft
import cv2
import warnings

def printUsage(rows, cols, sampOnOff, filtOnOff, filtType):
    ytext=np.zeros((rows,cols),dtype=float)
    cv2.putText(ytext,"Down- and upsampling and LP filtering Demo", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0))
    cv2.putText(ytext,"Toggle LP filter in 2D-FFT on/off ('f'): " + filtOnOff, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
    cv2.putText(ytext, "Switch between pyramid and rect filter ('k'): " + filtType, (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
    cv2.putText(ytext,"Toggle sampling on/off ('s'): " + sampOnOff, (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
    cv2.putText(ytext,"Quit ('q')", (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
    return ytext


def predictionMode(i):
    frametypes = ['I', 'P']
    # frametypes = ['I']
    return frametypes[i % len(frametypes)]


### Color Transforms
def bgr2yCbCr(bgr):
    """Applies color transform from BGR image to YCbCr color space.
    It returns float values in (u)int8 range """

    r = bgr[:,:,2]
    g = bgr[:,:,1]
    b = bgr[:,:,0]

    y = 0.299*r + 0.587*g + 0.114*b
    cb = -0.16864*r - 0.33107*r + 0.49970*b
    cr = 0.499813*r - 0.418531*g - 0.081282*b

    return y, cb, cr

def yCbCr2bgr(y, cb, cr):
    """Applies color transform from YCbCr image to BGR color space.
    Returns float values in uint8 range. """

    r = 1.0*y + 1.4025*cr
    g = 1.0*y - 0.34434*cb - 0.7144*cr
    b = 1.0*y + 1.7731*cb

    bgr = np.ndarray((r.shape[0],r.shape[1],3),dtype=r.dtype)
    bgr[:,:,0] = b
    bgr[:,:,1] = g
    bgr[:,:,2] = r

    return bgr

### Filters and filtering
def rectFilt(N=4):
    """Creates NxN filter kernel with 1/(N**2) at each value. """
    return np.ones((int(N),int(N))) / (N**2)

def pyramidFilt(N=5):
    """Creates NxN (if N odd) or N-1xN-1 (if N even) filter kernel with pyramidal values using convolution
    of two rect filter kernels.
    This way it is always a symetrical filter. """
    rect = rectFilt(round((N)/2))
    return sig.convolve2d(rect,rect)


def motionEstimation(curr, prev, N=8):
    rows, cols = curr.shape
    mv = np.zeros((rows, cols, 2))

    for r in range(7*N, rows-7*N, N):
        for c in range(7*N, cols-7*N, N):
            block = curr[r:r+N, c:c+N]
            mv[r, c, :] = threeStepSearch(r, c, block, prev, d=4, N=N)
    return mv


def threeStepSearch(r, c, block, prev, d=4, N=8):

    # first iteration
    blocks = [prev[r+rd*N: r+rd*N + N, c+cd*N: c+cd*N + N]
              for (rd, cd) in itertools.product(np.linspace(-d, d, d-1, dtype=int), np.linspace(-d, d, 3, dtype=int))]

    indices = [(r+rd*N, c+cd*N) for (rd, cd) in itertools.product(np.linspace(-d, d, d-1, dtype=int), np.linspace(-d, d, d-1, dtype=int))]

    SADs = [np.sum(np.abs(block - pBlock)) for pBlock in blocks]
    minIx = SADs.index(min(SADs))

    r1, c1 = indices[minIx]

    # second iteration

    blocks = [prev[int(r1 + rd * N/2): int(r1 + rd * N/2 + N), int(c1 + cd * N/2): int(c1 + cd * N/2 + N)] for (rd, cd) in
              itertools.product(np.linspace(-d, d, d - 1, dtype=int), np.linspace(-d, d, 3, dtype=int))]

    indices = [(int(r1 + rd * N/2), int(c1 + cd * N/2)) for (rd, cd) in
               itertools.product(np.linspace(-d, d, d - 1, dtype=int), np.linspace(-d, d, 3, dtype=int))]

    SADs = [np.sum(np.abs(block - pBlock)) for pBlock in blocks]
    minIx = SADs.index(min(SADs))
    r2, c2 = indices[minIx]

    # third iteration


    blocks = [prev[int(r2 + rd * N / 4): int(r2 + rd * N / 4 + N), int(c2 + cd * N / 4): int(c2 + cd * N / 4 + N)] for
              (rd, cd) in
              itertools.product(np.linspace(-d, d, d - 1, dtype=int), np.linspace(-d, d, 3, dtype=int))]

    indices = [(int(r2 + rd * N / 4), int(c2 + cd * N / 4)) for (rd, cd) in
               itertools.product(np.linspace(-d, d, d - 1, dtype=int), np.linspace(-d, d, 3, dtype=int))]

    SADs = [np.sum(np.abs(block - pBlock)) for pBlock in blocks]
    minIx = SADs.index(min(SADs))
    r3, c3 = indices[minIx]

    return [r3-r, c3-c]


def predictFrame(prev, mvs, N):
    [rows, cols] = prev.shape
    pred = np.zeros((rows, cols))
    for r in range(0, rows, N):
        for c in range(0, cols, N):
            rIx = int(r + mvs[r, c, 0])
            cIx = int(c + mvs[r, c, 1])
            pred[r:r+N, c:c+N] = prev[rIx:rIx+N, cIx:cIx+N]

    return pred


def dpcm(src):
    (row, col) = src.shape
    out = np.zeros(src.shape)
    for r in range(row):
        for c in range(col):
            if r == 0 and c == 0:
                out[r, c] = src[r, c] - 128
            elif c == 0:
                out[r, c] = src[r, c] - src[r-1, c]
            else:
                out[r, c] = src[r, c] - src[r, c-1]
    return out

def idpcm(src):
    (row, col) = src.shape
    out = np.zeros(src.shape)
    for r in range(row):
        for c in range(col):
            if r == 0 and c == 0:
                out[r, c] = src[r, c] + 128
            elif c == 0:
                out[r, c] = src[r, c] + src[r - 1, c]
            else:
                out[r, c] = src[r, c] + src[r, c - 1]
    return out

class TestVC_Lib(unittest.TestCase):
    def setUp(self):
        self.block1 = np.eye(8)



if __name__ == '__main__':
    unittest.main()