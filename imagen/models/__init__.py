# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import os
import bz2
import urllib
import hashlib
import progressbar


DATA_CHUNK = 8192
DATA_FOLDER = 'data'
DATA_URL = 'https://github.com/alexjc/neural-imagen/releases/download/0.0'


def download_to_file(model, hexdigest):
    filename = f'data/{model}.model'
    if os.path.exists(filename):
        return filename

    response = urllib.request.urlopen(f'{DATA_URL}/{model}.model.bz2')
    
    widgets = [progressbar.Percentage(), progressbar.Bar(marker='■', fill='·'), progressbar.DataSize(), ' ', progressbar.ETA()]
    bunzip, output, hasher = bz2.BZ2Decompressor(), open(f'{DATA_FOLDER}/{model}.model', 'wb'), hashlib.new('md5')

    with progressbar.ProgressBar(max_value=response.length, widgets=widgets) as bar:
        for i in range((response.length // DATA_CHUNK)+1):
            chunk = response.read(DATA_CHUNK)
            data = bunzip.decompress(chunk)

            bar.update(i * DATA_CHUNK)
            hasher.update(data)
            output.write(data)

    assert hasher.hexdigest() == hexdigest, 'WARNING: Data has unexpected MD5 checksum.'
    return filename
