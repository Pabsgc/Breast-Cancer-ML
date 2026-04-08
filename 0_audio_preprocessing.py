import opensmile
import pandas as pd

# Initialize the OpenSMILE feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

files = ["audio1.wav", "audio2.wav", "audio3.wav"]

# Dataset containing the extracted features
X = []
for f in files:
    feat = smile.process_file(f)
    X.append(feat)

X = pd.concat(X)
print(X.shape)