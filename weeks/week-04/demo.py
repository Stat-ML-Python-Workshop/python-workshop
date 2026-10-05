from mini_ml.linalg import dot,matvec
from mini_ml.probability import stable_softmax
import numpy as np
logits=[dot([2,1],[0.8,-0.2]),dot([2,1],[-0.3,0.7])]
print(logits,stable_softmax(logits))
shift=np.asarray(logits)-np.max(logits);reference=np.exp(shift)/np.exp(shift).sum()
assert np.allclose(stable_softmax(logits),reference)

# BIOLOGICAL DATA
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
data_dir=Path(__file__).resolve().parents[1]/"data"/"week-04"
training=pd.read_csv(data_dir/"train.csv")
bio_x=training[["mean_radius","mean_texture"]].values.tolist()
bio_y=training.malignant.tolist()
from mini_ml.probability import stable_softmax
weights=[0.1,0.2]
weighted_scores=[dot(row,weights) for row in bio_x[:5]]
print({'scores':weighted_scores,'probabilities':stable_softmax(weighted_scores)}) # softmax 只正規化相對分數，不會讓人為分數成為校準過的疾病風險。
