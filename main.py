from encoding import encode
from decoding import decode

DS_FACT = 2
BLK_SIZE = 8

for q in [6]:
    filename = 'data/videorecord_dct_q' + str(q) + '.txt'
    print('Encoding file: ' + filename)
    encode(filename, 25, writeToFile=True, N=DS_FACT, BLK_SIZE=BLK_SIZE, QUAL_FACT=q)
    print('Decoding file: ' + filename)
    decode(filename, N=DS_FACT, BLK_SIZE=BLK_SIZE, QUAL_FACT=q)
