import os
import numpy as np
import json
import time
import numpy as np
import jieba
from rouge_chinese import Rouge

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams

tpl = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

sampling_params = SamplingParams(temperature=0, max_tokens=4096, stop=["<|im_end|>", "<|endoftext|>"])

llm = LLM(
    model="merge_qwen1.5-7b-no2sql-v2",
    gpu_memory_utilization=0.7,
	max_model_len=4096,
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
        print(text)


preds = np.array(preds)
labels = np.array(labels)
np.save("preds.npy", preds)
np.save("labels.npy", labels)

rouge1 = []
rouge2 = []
rougel = []
rouge = Rouge()
for lab, pre in zip(labels, preds):
    lab = " ".join(jieba.cut(lab))
    pre = " ".join(jieba.cut(pre))
    score = rouge.get_scores(lab, pre)
    rouge1.append(score[0]['rouge-1']["f"])
    rouge2.append(score[0]['rouge-2']["f"])
    rougel.append(score[0]['rouge-l']["f"])

rouge1 = np.mean(rouge1)
rouge2 = np.mean(rouge2)
rougel = np.mean(rougel)
print("=" * 100)
print("预测结果：")
print(f"Rouge-1：{rouge1}")
print(f"Rouge-2：{rouge2}")
print(f"Rouge-L：{rougel}")
print()
print("=" * 100)

