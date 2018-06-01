# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
path="/home/rsanchez/app/scipion/pyworkflow/em/packages/xmipp3/backupDeepLearning"
x=pd.read_table(os.path.join(path,"scores.tab"), names=["score","categ"])
x["score"][x["categ"]<1].hist(bins=20)
plt.show()
print("Done")