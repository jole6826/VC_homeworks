import numpy as np
import cv2
import _pickle as pickle
import lib.vc_lib as vc
import lib.vc_compression as comp
import scipy.signal as sig

#Program to capture video from a camera and store it in an recording file, in Python txt format, using cPickle
#This is a framework for a simple video encoder to build.
#It writes into file 'videorecord.txt'
#Gerald Schuller, April 2015

writeToFile = True
filename = 'data/videorecord_dct_test.txt'

with open(filename, 'wb') as f:
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    [rows, cols, c] = frame.shape
    print(frame.shape)

    N = 2           # subsampling of chroma components
    BLK_SIZE = 8    # only possible value because of quantization matrix Q
    QUAL_FACT = 2   # value between 1 and N (larger -> better quality)

    chr_subsampling = True
    lp_filt = True
    pyrFilt = vc.pyramidFild(2*N)
    rectFilt = vc.rectFilt(2*N)
    filterType = True   # True -> pyramid filter
                        # False -> rectangular filter

    for n in range(50):
    # while True:
        ret, frame = cap.read()

        if ret:

            frame_cp = frame.copy()
            ycrcb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[:,:,0]*1.0
            cr = ycrcb[:,:,1]-128.0
            cb = ycrcb[:,:,2]-128.0

            if filterType:
                lp = pyrFilt
                fType = 'Pyramid'
            else:
                lp = rectFilt
                fType = 'Rect'

            if lp_filt:
                cr = sig.convolve2d(cr.copy(), lp, mode='same')
                cb = sig.convolve2d(cb.copy(), lp, mode='same')
                fStatus = 'on'
            else:
                fStatus = 'off'

            if chr_subsampling:
                cr = comp.chromaSubsampling(cr, N)
                cb = comp.chromaSubsampling(cb, N)
                sStatus = 'on'
            else:
                sStatus = 'off'

            ytext = vc.printUsage(rows, cols, sStatus, fStatus, fType)
            cv2.imshow('Y component', (y+ytext)/255.0)
            cv2.imshow('Cr component', np.abs((cr+128)/255.0))
            cv2.imshow('Cb component', np.abs((cb+128)/255.0))

            # Compression via DCT
            quantiY = comp.DCTcompression(y-128, BLK_SIZE, k=QUAL_FACT)
            quantiCr = comp.DCTcompression(cr, BLK_SIZE, k=QUAL_FACT)
            quantiCb = comp.DCTcompression(cb, BLK_SIZE, k=QUAL_FACT)

            if writeToFile:
                pickle.dump(np.array(quantiY, dtype='int8'), f)
                pickle.dump(np.array(quantiCr, dtype='int8'), f)
                pickle.dump(np.array(quantiCb, dtype='int8'), f)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                lp_filt = not lp_filt
            if key == ord('k'):
                filterType = not filterType
            if key == ord('s'):
                chr_subsampling = not chr_subsampling
            if key == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
