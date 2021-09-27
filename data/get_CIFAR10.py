import pickle
import numpy as np
import os
import sys
import argparse
import urllib.request
import tarfile
from skimage import io

output_dir='.'

# Unpickle Dataset
def unpickle(file):
    f = open(file, 'rb')
    dict_dataset = pickle.load(f, encoding='latin1')
    f.close()
    return dict_dataset


if __name__ == '__main__':
    
    # Download
    os.makedirs(f'{output_dir}', exist_ok=True)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    fname = url.split('/')[-1]
    fpath = os.path.join(output_dir, fname)
    
    def _progress(cnt, chunk, total):
        now = cnt * chunk
        if now > total: now = total
        sys.stdout.write(f'\rdownloading {fname} {now} / {total} ({now/total:.1%})')
        sys.stdout.flush()
    urllib.request.urlretrieve(url, fpath, _progress)
    print('')
    
    # Unzip
    tarfile.open(fpath, 'r:gz').extractall(output_dir)
    
    os.rename(os.path.join(output_dir, 'cifar-10-batches-bin'), os.path.join(output_dir, 'cifar'))
    

