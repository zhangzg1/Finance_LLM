import os

CLASSIFY_PTUNING_PRE_SEQ_LEN = 512
KEYWORDS_PTUNING_PRE_SEQ_LEN = 256
NL2SQL_PTUNING_PRE_SEQ_LEN = 128
NL2SQL_PTUNING_MAX_LENGTH = 2200

# 基础路径
BASE_DIR = "/root/Finance_LLM/"
# 存储pdf中所有数据的路径
DATA_PATH = BASE_DIR + "pdf_data/"
# 存储pdf中文本信息的路径
PDF_TEXT_DIR = 'pdf_docs'
# 存储错误pdf文件的路径
ERROR_PDF_DIR = 'error_pdfs'

# 并发运行的进程数
NUM_PROCESSES = 5
CLASSIFY_CHECKPOINT_PATH = BASE_DIR + "llm_finetune/ptuning/CLASSIFY_PTUNING/output/Fin-Train-chatglm3-6b-pt-512-2e-2/checkpoint-400"
KEYWORDS_CHECKPOINT_PATH = BASE_DIR + "llm_finetune/ptuning/KEYWORDS_PTUNING/output/Fin-Train-chatglm3-6b-pt-256-2e-2/checkpoint-250"
NL2SQL_CHECKPOINT_PATH = BASE_DIR + "llm_finetune/ptuning/NL2SQL_PTUNING/output/Fin-Train-chatglm3-6b-pt-128-2e-2/checkpoint-600"

# XPDF是一个轻量级的PDF查看器
XPDF_PATH = BASE_DIR + 'xpdf/bin64'

LLM_MODEL_DIR = BASE_DIR + 'models/chatglm3-6b'
