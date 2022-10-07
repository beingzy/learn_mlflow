# utf-8
import os
import tarfile
from six.moves import urllib

import pandas as pd


DOWNLOAD_ROOTS = ('https://raw.githubusercontent.com/' +
                  'ageron/handson-ml/master/')
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOTS + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(
        housing_url=HOUSING_URL,
        housing_path=HOUSING_PATH):
    """
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as housing_tgz:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(housing_tgz, path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
