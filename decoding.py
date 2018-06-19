import numpy as np
import scipy.signal as sig
import cv2
import sys
import _pickle as pickle
import lib.vc_lib as vc
import lib.vc_compression as comp


def load_pickle_iter(filename, mode):
    while 1:
        try:
            if mode == 'I':
                yield (pickle.load(filename), pickle.load(filename), pickle.load(filename), pickle.load(filename))
            elif mode == 'P':
                yield (pickle.load(filename), pickle.load(filename), pickle.load(filename), pickle.load(filename))
        except:
            break


def decode_I_Frame(yComp, crComp, cbComp, N, BLK_SIZE, QUAL_FACT):
    # inverse DCT compression step
    y = comp.inverseDCTcompression(yComp, N=BLK_SIZE, k=QUAL_FACT) + 128
    cr = comp.inverseDCTcompression(crComp, N=BLK_SIZE, k=QUAL_FACT)
    cb = comp.inverseDCTcompression(cbComp, N=BLK_SIZE, k=QUAL_FACT)

    # y = vc.idpcm(y)
    # cr = vc.idpcm(cr)
    # cb = vc.idpcm(cb)

    # upsampling chroma components
    cb_us = np.zeros(y.shape, dtype='float64')
    cr_us = np.zeros(y.shape, dtype='float64')
    cb_us[::N, ::N] = cb
    cr_us[::N, ::N] = cr

    # lowpass
    pyrFilt = N * N * vc.pyramidFilt(2 * N)
    cr = np.array(sig.convolve2d(cr_us, pyrFilt, mode='same'), dtype=np.float64)
    cb = np.array(sig.convolve2d(cb_us, pyrFilt, mode='same'), dtype=np.float64)

    # put channels back together
    ycrcb = np.ndarray((y.shape[0], y.shape[1], 3), dtype='uint8')

    ycrcb[:, :, 0] = y.astype('uint8')
    ycrcb[:, :, 1] = np.clip(cr + 128.0, 0, 255).astype('uint8')
    ycrcb[:, :, 2] = np.clip(cb + 128.0, 0, 255).astype('uint8')

    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return bgr


def decode_P_Frame(yPredError, crPredError, cbPredError, mvs, prev, N, BLK_SIZE, QUAL_FACT):
    ycrcbPrev = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    yPrev = ycrcbPrev[:, :, 0] * 1.0
    crPrev = ycrcbPrev[:, :, 1] - 128.0
    cbPrev = ycrcbPrev[:, :, 2] - 128.0

    # predict current frame from mvs and prev frame
    yPred = vc.predictFrame(yPrev, N=N, mvs=mvs)
    crPred = vc.predictFrame(crPrev, N=N, mvs=mvs)
    cbPred = vc.predictFrame(cbPrev, N=N, mvs=mvs)

    yPredError = comp.inverseDCTcompression(yPredError, BLK_SIZE, QUAL_FACT)
    crPredError = comp.inverseDCTcompression(crPredError, BLK_SIZE, QUAL_FACT)
    cbPredError = comp.inverseDCTcompression(cbPredError, BLK_SIZE, QUAL_FACT)

    # upsampling chroma components
    cbPredError_us = np.zeros(yPrev.shape, dtype='float64')
    crPredError_us = np.zeros(yPrev.shape, dtype='float64')
    cbPredError_us[::N, ::N] = cbPredError
    crPredError_us[::N, ::N] = crPredError

    # lowpass
    pyrFilt = N * N * vc.pyramidFilt(2 * N)
    crPredError_us = np.array(sig.convolve2d(crPredError_us, pyrFilt, mode='same'), dtype=np.float64)
    cbPredError_us = np.array(sig.convolve2d(cbPredError_us, pyrFilt, mode='same'), dtype=np.float64)

    # add pred error and prediction together
    y = yPred + yPredError
    cr = crPred + crPredError_us
    cb = cbPred + cbPredError_us

    # put channels back together
    ycrcb = np.ndarray((y.shape[0], y.shape[1], 3), dtype='uint8')

    ycrcb[:, :, 0] = y.astype('uint8')
    ycrcb[:, :, 1] = np.clip(cr + 128.0, 0, 255).astype('uint8')
    ycrcb[:, :, 2] = np.clip(cb + 128.0, 0, 255).astype('uint8')

    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return bgr


def decode(filename, N=2, BLK_SIZE=8, QUAL_FACT=4):
    cnt = 0
    mode = vc.predictionMode(cnt)

    for (compYdct, compCrdct, compCbdct, mvs) in load_pickle_iter(open(filename, 'rb'), mode):
        mode = vc.predictionMode(cnt)
        cnt += 1
        if mode == 'I':
            bgr = decode_I_Frame(compYdct, compCrdct, compCbdct, N, BLK_SIZE, QUAL_FACT)
        elif mode == 'P':
            bgr = decode_P_Frame(compYdct, compCrdct, compCbdct, mvs, bgr, N, BLK_SIZE, QUAL_FACT)

        cv2.imshow('Reconstructed RGB signal (quality factor: {})'.format(QUAL_FACT), bgr)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
