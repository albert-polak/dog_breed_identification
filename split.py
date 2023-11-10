import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv("./dog-breed-identification/labels.csv")
label_encoder = LabelEncoder()
print(labels['breed'])
labels['label'] = label_encoder.fit_transform(labels['breed'])

train, val = train_test_split(labels, train_size=0.8, random_state=42)

test, val = train_test_split(val, train_size=0.5, random_state=42)

pd.DataFrame(train).to_csv("./train.csv")
pd.DataFrame(test).to_csv("./test.csv")
pd.DataFrame(val).to_csv("./val.csv")