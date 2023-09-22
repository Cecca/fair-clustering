import pandas as pd
import numpy as np
import json

with open("emoji.json") as fp:
    emoji = json.load(fp)

variants = [0x1F3FB,0x1F3FC,0x1F3FD]#,0x1F3FE,0x1F3FF]

bases = [
    "👶",
    "🧒",
    "👧",
    "🧑","👱","👨","🧔","🧔‍♀️"
]
# for base in bases:
#     print(base)


np.random.seed(1234)
n_male = 100
n_female = 100
n_other = 20

m_weight = np.random.normal(70, 10, size=n_male)
m_height = np.random.normal(170, 20, size=n_male)

f_weight = np.random.normal(50, 10, size=n_female)
f_height = np.random.normal(165, 10, size=n_female)

o_weight = np.random.normal(60, 10, size=n_other)
o_height = np.random.normal(168, 10, size=n_other)

ethnicities=np.array(["white", "black", "asian", "other"])
ethnicity = ethnicities[np.random.choice(len(ethnicities), size = n_male+n_female+n_other)]

df = pd.concat([
    pd.DataFrame({"weight": m_weight, "height": m_height, "gender": "male", "label": bases[0]}),
    pd.DataFrame({"weight": f_weight, "height": f_height, "gender": "female", "label": bases[1]}),
    pd.DataFrame({"weight": o_weight, "height": o_height, "gender": "other", "label": bases[2]}),
])
df["ethnicity"] = ethnicity

df.to_csv("example.csv", index=False)
print(df)


