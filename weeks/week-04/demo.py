from mini_ml.linalg import matmul
import numpy as np

m1=[[1,2,3],[4,5,6]]
m2=[[7,8],[9,10],[11,12]]
assert matmul(m1,m2)==[[58,64],[139,154]]
assert np.allclose(matmul(m1,m2),np.asarray(m1) @ np.asarray(m2))

# BIOLOGICAL DATA
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
data_dir=Path(__file__).resolve().parents[1]/"data"/"week-04"
training=pd.read_csv(data_dir/"train.csv")
bio_x=training[["mean_radius","mean_texture"]].values.tolist()
bio_y=training.malignant.tolist()
weights_column=[[0.4],[0.1]]
weighted_sums=matmul(bio_x[:3],weights_column)
scores=[[value-7.0 for value in row] for row in weighted_sums]
assert np.allclose(weighted_sums,np.asarray(bio_x[:3]) @ np.asarray(weights_column))
print({'weighted_sums':weighted_sums,'linear_scores':scores})
