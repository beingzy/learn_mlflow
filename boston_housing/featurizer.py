from data_loader import load_housing_data

from sklearn.preprocessing import (
    LabelBinarizer,
    StandardScaler,
    Imputer,
)
from sklearn.pipeline import Pipeline, FeatureUnion


boston_df = load_housing_data()

num_feat_list = []
cat_feat_list = []


num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")), 
    ('attribs_adder', )
    ])