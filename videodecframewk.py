import numpy as np
import scipy.signal as sig
import cv2
import sys
import _pickle as pickle
import lib.vc_lib as vc
import lib.vc_compression as comp

#Program to open a video input file 'videorecord.txt' (python txt format using pickle) and display it on the screen.
#This is a framework for a simple video decoder to build.
#Gerald Schuller, April 2015

filename = 'data/videorecord_dct_test.txt'
N = 2
BLK_SIZE = 8
QUAL_FACT = 2

with open(filename, 'rb') as f:

    while True:
        # load next frame from file f and "de-pickle" it, convert from a string back to matrix or tensor:
        compYdct = pickle.load(f)
        compCrdct = pickle.load(f)
        compCbdct = pickle.load(f)

        # inverse DCT compression step
        y = comp.inverseDCTcompression(compYdct, N=BLK_SIZE, k=QUAL_FACT) + 128
        cr = comp.inverseDCTcompression(compCrdct, N=BLK_SIZE, k=QUAL_FACT)
        cb = comp.inverseDCTcompression(compCbdct, N=BLK_SIZE, k=QUAL_FACT)

        # upsampling chroma components
        cb_us = np.zeros(y.shape, dtype='float64')
        cr_us = np.zeros(y.shape, dtype='float64')
        cb_us[::N, ::N] = cb
        cr_us[::N, ::N] = cr

        # lowpass
        pyrFilt = N*N * vc.pyramidFild(2*N)
        cr = np.array(sig.convolve2d(cr_us,pyrFilt,mode='same'), dtype=np.float64)
        cb = np.array(sig.convolve2d(cb_us,pyrFilt,mode='same'), dtype=np.float64)

        # put channels back together
        ycrcb = np.ndarray((y.shape[0],y.shape[1],3),dtype='uint8')

        ycrcb[:,:,0] = y.astype('uint8')
        ycrcb[:,:,1] = np.clip(cr+128.0, 0, 255).astype('uint8')
        ycrcb[:,:,2] = np.clip(cb+128.0, 0, 255).astype('uint8')

        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        cv2.imshow('Reconstructed RGB component', bgr)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
