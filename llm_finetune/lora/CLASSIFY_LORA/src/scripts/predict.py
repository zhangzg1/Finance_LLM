import os
import numpy as np
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams

tpl = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop=["<|im_end|>", "<|endoftext|>"])

llm = LLM(
    model="merge_qwen1.5-7b-intent-v2",
    gpu_memory_utilization=0.7,
	max_model_len=2048,
	dtype="half",
    trust_remote_code=True
)

fd = open("data/test.json", "r")
data = json.load(fd)
print("size: ", len(data))

labels = []
preds = []
for idx, line in enumerate(data):
    query = line["instruction"]
    target = line["output"]
    labels.append(target)
    t1 = time.time()
    prompts = [query]
    for idx, p in enumerate(prompts):
        prompts[idx] = tpl.format(p)

    outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()
    for output in outputs:
        text = output.outputs[0].text
        preds.append(text)
        print(f"processed {idx}", target, text)

labels = np.array(labels)
preds = np.array(preds)
label_maps = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
preds = [label_maps[k] for k in preds]
labels = [label_maps[k] for k in labels]

acc = accuracy_score(labels, preds)

print("=" * 100)
print("预测结果：")
print(f"准确率：{acc}")
print()
print("=" * 100)

