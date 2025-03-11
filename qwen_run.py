import os

GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

import torch
from datetime import datetime
from loguru import logger
from config import cfg
from llm_finetune.qwen_lora import QwenLoRA, LoraType
from utils.file import download_data, extract_pdf_filenames
from generate_util.company_table import count_table_keys, build_table
from pdf_process.pdf_parse import extract_pdf_text, extract_pdf_tables
from check import init_check_dir, check_text, check_tables
from generate_answer_with_classify import do_gen_keywords, do_classification, do_sql_generation, generate_answer, \
    make_answer


# 检查项目中各种关键路径和依赖项是否完好
def check_paths():
    if not os.path.exists(cfg.DATA_PATH):
        raise Exception('DATA_PATH not exists: {}'.format(cfg.DATA_PATH))

    if not os.path.exists(cfg.XPDF_PATH):
        raise Exception('XPDF_PATH not exists: {}'.format(cfg.XPDF_PATH))
    else:
        os.chdir(cfg.XPDF_PATH)
        os.system(f"chmod 755 {cfg.XPDF_PATH}/pdftotext")
        os.system(
            f'{cfg.XPDF_PATH}/pdftotext -table -enc UTF-8 {cfg.DATA_PATH}/check/test.pdf {cfg.DATA_PATH}/check/test.txt')
        with open(f'{cfg.DATA_PATH}/check/test.txt', 'r', encoding='utf-8') as f:
            print(f.readlines()[:10])
        print('Test xpdf success!')

    print('Torch cuda available ', torch.cuda.is_available())

    """
    for name in ['basic_info', 'employee_info', 'cbs_info', 'cscf_info', 'cis_info', 'dev_info']:
        table_path = os.path.join(cfg.DATA_PATH, '{}.json'.format(name))
        if not os.path.exists(table_path):
            raise Exception('table {} not exists: {}'.format(name, table_path))

    if not os.path.exists(os.path.join(cfg.DATA_PATH, 'CompanyTable.csv')):
        raise Exception('CompanyTable.csv not exists: {}'.format(os.path.join(cfg.DATA_PATH, 'CompanyTable.csv')))
    """

    print('Check paths success!')


if __name__ == '__main__':
    # 获取所有PDF的文件名，在pfd_data下生成pdf_name.txt
    extract_pdf_filenames("./pdf_data/all_pdf", "./pdf_data/pdf_name.txt")

    # 配置日志信息，在pfd_data下生成 {data}.main.log
    DATE = datetime.now().strftime('%Y%m%d')
    log_path = os.path.join(cfg.DATA_PATH, '{}.main.log'.format(DATE))
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.add(log_path, level='DEBUG')

    # 检查目录数据是否齐全
    check_paths()

    # 1. 下载数据到pdf_data目录, 生成pdf_info.json
    download_data()

    # 2. 解析pdf, 提取相关数据，在pdf_data下生成pdf_docs, basic/cbs/cis/cscf/dev/employee_info.json
    extract_pdf_text()
    extract_pdf_tables()

    # 3. 检查一下数据, 缺失之类的，在pfd_data下生成error_pdfs
    init_check_dir()
    check_text(copy_error_pdf=True)
    check_tables(copy_error_pdf=True)

    # 4. 根据表中的字段生成总表，在pfd_data下生成key_count.json和CompanyTable.csv
    count_table_keys()
    build_table()

    # 5. 对问题进行分类，在pfd_data下生成classify
    model = QwenLoRA(LoraType.Classify)
    do_classification(model)

    # 6. 给问题生成keywords，在pfd_data下生成keywords
    model = QwenLoRA(LoraType.Keywords)
    do_gen_keywords(model)

    # 7. 对于统计类问题生成SQL，在pfd_data下生成sql
    model = QwenLoRA(LoraType.NL2SQL)
    do_sql_generation(model)

    # 8. 生成回答，在pfd_data下生成answers
    model = QwenLoRA(LoraType.Nothing)
    generate_answer(model)

    # 9. 生成预测结果，在pfd_data下生成result_{data}.json
    make_answer()
