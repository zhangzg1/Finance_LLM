import os
from multiprocessing import Pool
from loguru import logger
from config import cfg
from utils.file import load_pdf_info
from pdf_process.pdf_util import PdfExtractor
from pdf_process.financial_state import (extract_basic_info, extract_employee_info, extract_cbs_info, extract_cscf_info,
                                         extract_cis_info, extract_dev_info, merge_info)


def setup_xpdf():
    os.chdir(cfg.XPDF_PATH)
    cmd = 'chmod +x pdftotext'
    os.system(cmd)


# 单词提取 PDF 中的文本内容并保存
def extract_pure_content(idx, key, pdf_path):
    logger.info('Extract text for {}:{}'.format(idx, key))
    save_dir = os.path.join(cfg.DATA_PATH, cfg.PDF_TEXT_DIR)
    key_dir = os.path.join(save_dir, key)
    if not os.path.exists(key_dir):
        os.mkdir(key_dir)
    save_path = os.path.join(key_dir, 'pure_content.txt')
    if os.path.exists(save_path):
        os.remove(save_path)
    PdfExtractor(pdf_path).extract_pure_content_and_save(save_path)


# 使用多进程并行提取多个 PDF 文件的文本内容，默认使用 extract_pure_content 函数
def extract_pdf_text(extract_func=extract_pure_content):
    setup_xpdf()
    save_dir = os.path.join(cfg.DATA_PATH, cfg.PDF_TEXT_DIR)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pdf_info = load_pdf_info()
    # 创建多进程池，用于并行处理多个任务
    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.starmap(extract_func, [(i, k, v['pdf_path']) for i, (k, v) in enumerate(pdf_info.items())])


# 是从多个PDF文件中提取不同类型的表格信息，并使用多进程并行处理
def extract_pdf_tables():
    pdf_info = load_pdf_info()
    pdf_keys = list(pdf_info.keys())

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_basic_info, pdf_keys)
    merge_info('basic_info')

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_employee_info, pdf_keys)
    merge_info('employee_info')

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_cbs_info, pdf_keys)
    merge_info('cbs_info')

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_cscf_info, pdf_keys)
    merge_info('cscf_info')

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_cis_info, pdf_keys)
    merge_info('cis_info')

    with Pool(processes=cfg.NUM_PROCESSES) as pool:
        pool.map(extract_dev_info, pdf_keys)
    merge_info('dev_info')


if __name__ == '__main__':
    extract_pdf_text()
    extract_pdf_tables()
