# 基于 LLM Lora 微调的金融知识问答系统，主要结合了 PDF解析、LLM微调、vllm 推理优化框架等技术

# 1、介绍

本项目属于大模型微调任务，主要回答用户的金融相关的问题。首先对 1k 份上市公司年度报告文件进行解析，然后构建高质量的 SFT 数据微调 Qwen2-7B 模型生成多个子模型，最后利用 vllm 框架实现单底座 + 多 adapter 的部署模式进行推理，从而节省 GPU 的资源。该项目主要结合了 PDF 解析、LLM 微调、vllm 推理优化框架等技术。

## 2、下载源码与环境安装（Linux）

```
# 下载源码
git clone https://github.com/zhangzg1/Finance_LLM.git
cd Finance_LLM

# 创建虚拟环境
conda create -n finance_llm python=3.9
conda activate finance_llm

# 安装其他依赖包
pip install -r requirements.txt
```

## 3、代码结构

```text
.
├── config
├── generate_util                       # 问题处理辅助函数
    └── company_table.py 
    └── recall_report_names.py    
    └── recall_report_text.py     
    └── type1.py
    └── type2.py
├── llm_finetune
    └── lora                            # lora微调
    └── ptuning                         # ptuning微调
    └── chatglm_ptuning.py              # ptuning测试
    └── qwen_lora.py                    # lora测试
├── models                              # 基座模型       
    └── chatglm3-6b
    └── Qwen2-7B-Instruct     
    └── text2vec-base-chinese
├── pdf_data
    └── all_pdf                         # pdf数据集
    └── check                           # 检测文件 
    └── test                            # 测试数据
├── pdf_process                         # pdf文件处理
    └── financial_state.py
    └── pdf_parse.py
    └── pdf_util.py
├── utils                               # 辅助函数
    └── file.py            
    └── prompt_util.py           
    └── question_util.py         
    └── re_util.py           
    └── sql_correct_util.py          
├── xpdf                                # pdf查看器
├── check.py                            # 检测目录数据
├── generate_answer_with_classfiy.py    # 问题处理流程
├── vllm_lorax.sh                       # vllm框架推理
├── qwen_run.py                         # qwen模型测试
├── chatglm_run.py                      # chatglm模型测试
├── test_score.py                       # 测试集得分计算
├── requirements.txt                    # 第三方依赖库
├── README.md                           # 说明文档             
```

## 4、代码运行

基于 qwen 模型的 lora 微调，使用 vllm 框架实现单底座 + 多 adapter 的部署模式进行推理

```
# lora微调三个模型
cd llm_finetune/lora/CLASSIFY_LORA/
bash src/scripts/train.sh

cd llm_finetune/lora/KEYWORDS_LORA/
bash src/scripts/train.sh

cd llm_finetune/lora/NL2SQL_LORA/
bash src/scripts/train.sh

# 启动vllm推理服务
bash vllm_lorax.sh

# 使用微调后的qwen模型进行推理测试
python qwen_run.py
```

基于 chatglm 模型的 ptuning 微调，实现推理测试

```
# ptuning微调三个模型
cd llm_finetune/ptuning/CLASSIFY_PTUNING/
bash train.sh

cd llm_finetune/ptuning/KEYWORDS_PTUNING/
bash train.sh

cd llm_finetune/ptuning/NL2SQL_PTUNING/
bash train.sh

# 使用微调后的chatglm模型进行推理测试
python chatglm_run.py
```

## 5、项目概述

### 5.1 基于大模型的金融知识问答

该项目的目标是深度解析上市公司年报的问答系统。面对金融文本中的专业术语与暗含信息，致力于用AI实现专家级别的金融分析。在 AI 领域，虽然已在文本对话取得进展，但真正的金融交互场景仍然是一个巨大挑战。上市公司年报为投资者呈现了公司的经营状况、财务状况和未来规划。专业知识是解读的关键，而项目的目标是通过 AI 技术让这一过程变得更简单、更准确。落地应用可用于股票交易问答助手，智能金融行业分析助手等。问题示例：

```
问题1: 2019年中国工商银行财务费用是多少元？
答案1: 2019年中国工商银行财务费用是12345678.9元。

问题2: 工商银行2019年营业外支出和营业外收入分别是多少元？
答案2: 工商银行2019年营业外支出为12345678.9元，营业外收入为2345678.9元。
```

### 5.2 数据集

这里的训练数据集是 1k 份上市公司年度报告文件，我们需要将这些 pdf 文件进行解析，然后利用解析后的数据对大模型进行微调训练。关于测试数据集，以下测试集问题的一些示例：

```
{"id": 0, "question": "无形资产是指什么？"}
{"id": 1, "question": "2019年负债总额第2高的上市公司是？"}
{"id": 2, "question": "哪家上市公司，2019年净利润第十二高？"}
```

## 6、项目流程

### 6.1 PDF解析与信息抽取

PDF 文件中不仅有文本数据，还有大量表格数据，如资产负债表等三大表，需要通过其他方式提取数据，保存到关系型数据库中。一般，我们将 PDF 文件进行格式转化。然后，基于 TXT、HTML、DOCX 或 OCR 识别的方式，提取其中的表格数据。为了以后查询方便，项目将关系型数据（包括表格和基本信息），按照每家公司一行的方式，做成一张大宽表。这样可以避免跨表查询的复杂度。

解析与抽取步骤如下：

1. **pdf文本抽取**: 采用 xpdf 工具
2. **页面召回**: 根据报表名称设置关键词找到对应的页
3. **表格识别**: 利用 camelot-py 库实现基于图像识别的表格提取
4. **信息过滤**: 非合并报表，调整报表，母公司报表

整个部分的代码在[pdf_process](https://github.com/zhangzg1/Finance_LLM/tree/main/pdf_process)

### 6.2 问题分类

要让大模型给出准确回答，前提是理解用户的问题和意图。针对不同类别的问题，采取不同的方法处理。我们可以通过微调一个分类模型，来进行自动化的问题分类。实际上是构造一个“分类器”，步骤如下：

1. 首先，通过少样本 in-context 学习生成数据集，并进行人工校验。
2. 然后，用该数据集进行微调（LoRA 或 P-Tuning）训练，构造问题分类模型。
3. 最后，把用户问题直接给到模型，可以让模型做选择题只给出类别。

总共有6个问题类别：

A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等。

B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等。

C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等。

D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重, 占比等。

E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果。

F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是 XXX。

微调大模型来完成问题分类的代码在[llm_finetune/lora/CLASSIFY_LORA](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/lora/CLASSIFY_LORA)或者[llm_finetune/ptuning/CLASSIFY_PTUNING](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/ptuning/CLASSIFY_PTUNING)

### 6.3 SQL生成

NL2SQL 微调训练重点任务还是构造训练数据集。这里需要注意的是，要针对实际数据查询场景构造数据集，而非使用 Spider 等通用的数据集。主要有两种方法：

1. 基于 SQL 模板生成数据集，但是泛化能力就比较低。
2. 构建 SQL 问答模板，对字段进行随机填充，然后利用 ChatGPT 等大模型对问题改写，生成数据集，效果相对更好。

无论使用哪一种方法，数据集必须经过人工校验。对于微调来说，数据的质量要求，远大于数据的数量要求。微调大模型来完成 SQL 生成的代码在[llm_finetune/lora/NL2SQL_LORA](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/lora/NL2SQL_LORA)或者[llm_finetune/ptuning/NL2SQL_PTUNING](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/ptuning/NL2SQL_PTUNING)

### 6.4 关键词抽取

很多场景下，我们都需要通过 prompt，给到大模型来提取关键词。比如，通过提取用户问题中的关键词，我们可以理解用户意图，再按照规则来匹配回答问题的方法。但是对于专业领域，如果像 ChatGLM 这样的模型，不能很好地理解和准确提取专业术语，那么我们就可以微调一个关键词模型，用于在用户提问中提取相关专业的关键词。

关键词提取任务涉及从文本中提取关键词或短语，有助于后面检索相关信息，越精准的提取越能够让返回的信息更加准确。微调大模型来完成关键词抽取的代码在[llm_finetune/lora/KEYWORDS_LORA](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/lora/KEYWORDS_LORA)或者[llm_finetune/ptuning/KEYWORDS_PTUNING](https://github.com/zhangzg1/Finance_LLM/tree/main/llm_finetune/ptuning/KEYWORDS_PTUNING)

### 6.5 答案生成

经过对问题的处理后，主要可以分为三类问题：基本信息类、统计计算类、总结推理类。关于统计计算类的问题，主要是匹配比值类关键字，然后通过 python 计算来得到结果，而即便信息类和总结推理类的问题，则主要是通过大模型的推理来得到的。

## 7、vllm框架推理优化

vLLM 是一个基于 Python 的 LLM 推理和服务框架，它的主要优势在于简单易用和性能高效。通过 PagedAttention 技术、连续批处理、CUDA 核心优化以及分布式推理支持，vLLM 能够显著提高 LLM 的推理速度，降低显存占用，更好地满足实际应用需求。vLLM 推理框架使大模型推理速度得到明显提升，推理速度比普通推理有 1 倍的加速。在产品级的部署上，vLLM 既能满足 batch 推理的要求，又能实现高并发下的 continuous batching，在实际产品部署中应用是非常广泛的。

在这个项目中，我们采用 Qwen2-7B-Chat 大模型作为基座模型，分别使用不同数据集微调得到问题分类模型、SQL 生成模型，关键字抽取模型，然后在实现推理的过程中利用 vllm 框架实现单底座 + 多 adapter 的部署模式进行推理，也就是实现了 3 个 lora 模型的 batch 推理，这节省了 GPU 的部署资源。（具体代码[vllm_lorax.sh](https://github.com/zhangzg1/Finance_LLM/blob/main/vllm_lorax.sh)）

