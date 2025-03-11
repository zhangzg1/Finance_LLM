fd1 = open("train.json")
fw = open("new_train.json", "w")
fw2 = open("new_test.json", "w")

import json

data = json.load(fd1)
import random
random.seed(123)

random.shuffle(data)
train_size = int(len(data) * 0.9)
train = data[:train_size]
test = data[train_size:]
print(len(test), len(train))

new_train = json.dumps(train, ensure_ascii=False, indent=4)
new_test = json.dumps(test, ensure_ascii=False, indent=4)
fw.write(new_train)
fw2.write(new_test)
