import json
import pandas as pd

data_len = 384
data = pd.read_csv('../configs/datasheet.csv').head(data_len)
data_json = {}
for i in range(data_len):
    data_row = data.loc[i]
    path = data_row['Path']
    key = path.split('/')[1]
    path = "{data_root}/" + path
    idx = str(int(float(data_row['Label'])))

    data_json[key] = {
        "wav": path,
        "spk_id": idx
    }

with open('../configs/datasheet.json', 'w') as f:
    json.dump(data_json, f)
