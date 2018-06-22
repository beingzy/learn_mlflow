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
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
