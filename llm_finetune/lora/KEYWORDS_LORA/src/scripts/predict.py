import os
import numpy as np
import json
import time
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams

tpl = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop=["<|im_end|>", "<|endoftext|>"])

llm = LLM(
    model="merge_qwen1.5-7b-keywords-v3",
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

avg_acc = []
for p, l in zip(preds, labels):
    p = p.split(",")
    l = l.split(",")
    acc = len(set(p).intersection(set(l))) / (len(set(p + l)) + 1e-6)
    avg_acc.append(acc)

avg_acc = np.mean(avg_acc)

print("=" * 100)
print("预测结果：")
print(f"准确率：{avg_acc}")
print()
print("=" * 100)

