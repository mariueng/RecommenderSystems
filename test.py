import project_example as pe

import os
import json


DATA_FOLDER = 'active1000'
files = os.listdir(DATA_FOLDER)
# print(files)

ARBITRARY_INDEX = 5
filepath = os.path.join(DATA_FOLDER, files[ARBITRARY_INDEX])

# one way to load all events into memory
events = []
for line in open(filepath):
    events.append(json.loads(line.strip()))

# print(json.dumps(events[ARBITRARY_INDEX], indent=4))

df=pe.load_data("active1000")
print(df)
pe.load_dataset(df)
