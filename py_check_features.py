import os
import sys
import h5py

import numpy as np
import climate
logging = climate.get_logger(__name__)
climate.enable_default_logging()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <fea_dir>'.format(sys.argv[0])
        sys.exit()

    fea_dir = sys.argv[1]

    features = None
    num_imgs = 0
    fns_dict = {}

    total_fea = 0
    total_fns = 0

    for root, dirs, fns in os.walk(fea_dir, followlinks = True):
        for fn in fns:
            full_fn = os.path.join(root, fn)
            hdf_f = h5py.File(full_fn,'r')
            fea = hdf_f['fea'][:]
            fns = hdf_f['fns'][:]
            total_fea += fea.shape[0]
            total_fns += fns.shape[0]
            print fea.shape[0]
            assert fea.shape[0] == fns.shape[0], "Should not happen, we have re-runed the feature extraction."
            hdf_f.close()

    logging.info('total fea = %d, fns = %d', total_fea, total_fns)
