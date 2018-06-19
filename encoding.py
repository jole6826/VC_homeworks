import numpy as np
import cv2
import _pickle as pickle
import lib.vc_lib as vc
import lib.vc_compression as comp
import scipy.signal as sig
import decoding as dec


def encode_I_Frame(frame, N=2, BLK_SIZE=8, QUAL_FACT=2, lp_filt=True, filterType=True, chr_subsampling=True):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0] * 1.0
    cr = ycrcb[:, :, 1] - 128.0
    cb = ycrcb[:, :, 2] - 128.0

    lp = vc.pyramidFilt(2 * N)
    cr = sig.convolve2d(cr.copy(), lp, mode='same')
    cb = sig.convolve2d(cb.copy(), lp, mode='same')

    cr = comp.chromaSubsampling(cr, N)
    cb = comp.chromaSubsampling(cb, N)

    # Compression via DCT
    quantiY = comp.DCTcompression(y - 128, BLK_SIZE, k=QUAL_FACT)
    quantiCr = comp.DCTcompression(cr, BLK_SIZE, k=QUAL_FACT)
    quantiCb = comp.DCTcompression(cb, BLK_SIZE, k=QUAL_FACT)

    return quantiY, quantiCr, quantiCb


def encode_P_Frame(frame, prev, N=2, BLK_SIZE=8, QUAL_FACT=2, mode='I', lp_filt=True, filterType=True, chr_subsampling=True):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0] * 1.0
    cr = ycrcb[:, :, 1] - 128.0
    cb = ycrcb[:, :, 2] - 128.0

    ycrcbPrev = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    yPrev = ycrcbPrev[:, :, 0] * 1.0
    crPrev = ycrcbPrev[:, :, 1] - 128.0
    cbPrev = ycrcbPrev[:, :, 2] - 128.0

    mvs = vc.motionEstimation(y, yPrev)
    yPred = vc.predictFrame(yPrev, mvs, 8)
    crPred = vc.predictFrame(crPrev, mvs, 8)
    cbPred = vc.predictFrame(cbPrev, mvs, 8)
    yPredError = y - yPred
    crPredError = cr - crPred
    cbPredError = cb - cbPred

    lp = vc.pyramidFilt(2 * N)
    crPredError = sig.convolve2d(crPredError.copy(), lp, mode='same')
    cbPredError = sig.convolve2d(cbPredError.copy(), lp, mode='same')

    crPredError = comp.chromaSubsampling(crPredError, N)
    cbPredError = comp.chromaSubsampling(cbPredError, N)

    quantiY = comp.DCTcompression(yPredError, BLK_SIZE, k=QUAL_FACT)
    quantiCr = comp.DCTcompression(crPredError, BLK_SIZE, k=QUAL_FACT)
    quantiCb = comp.DCTcompression(cbPredError, BLK_SIZE, k=QUAL_FACT)

    return quantiY, quantiCr, quantiCb, mvs


def encode(filename, duration, writeToFile=False, N=2, BLK_SIZE=8, QUAL_FACT=2):

    with open(filename, 'wb') as f:
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        [rows, cols, c] = frame.shape

        chr_subsampling = True
        lp_filt = True
        filterType = True   # True -> pyramid filter / False -> rectangular filter
        cnt = 0
        bgrPrev = np.zeros((rows, cols, c))

        for n in range(duration):
            ret, frame = cap.read()
            mode = vc.predictionMode(cnt)
            cnt += 1

            if ret:
                frame_cp = frame.copy()
                frame_cp = frame_cp[0:400,100:500,:]
                cv2.imshow('Original RGB signal', frame_cp)

                if mode == 'I':
                    quantiY, quantiCr, quantiCb = encode_I_Frame(frame_cp, N, BLK_SIZE, QUAL_FACT, lp_filt, filterType, chr_subsampling)
                    bgrPrev = dec.decode_I_Frame(quantiY, quantiCr, quantiCb, N, BLK_SIZE, QUAL_FACT)

                    if writeToFile:
                        pickle.dump(np.array(quantiY, dtype='int8'), f)
                        pickle.dump(np.array(quantiCr, dtype='int8'), f)
                        pickle.dump(np.array(quantiCb, dtype='int8'), f)
                        pickle.dump(np.array([0], dtype='int8'), f)

                elif mode == 'P':
                    yPredError, crPredError, cbPredError, mvs = encode_P_Frame(frame_cp, bgrPrev, N, BLK_SIZE, QUAL_FACT)

                    if writeToFile:
                        pickle.dump(np.array(yPredError, dtype='int8'), f)
                        pickle.dump(np.array(crPredError, dtype='int8'), f)
                        pickle.dump(np.array(cbPredError, dtype='int8'), f)
                        pickle.dump(np.array(mvs, dtype='int8'), f)


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
