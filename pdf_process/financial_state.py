import os
import json
import re
import numpy as np
from functools import cmp_to_key
from multiprocessing import Pool
from loguru import logger
from config import cfg
from utils.re_util import clean_row_name
from pdf_process.pdf_util import PdfExtractor
from utils.file import get_raw_pdf_path, get_pdf_table_path, load_pdf_info, load_pdf_pure_text

DEBUG = False


# 根据预定义的关键词和上下文条件，在 PDF 文件中找到符合特定条件的页面。
def find_match_page(pdf_name, max_continuous_lines=100, min_match_number=0, required_line_keywords=[],
                    invalid_line_keywords=[], required_post_keywords=[], invalid_pre_keywords=[],
                    invalid_post_keywords=[]):
    pure_text = load_pdf_pure_text(pdf_name)
    text_lines = []
    for page in pure_text:
        page_lines = page['text'].split('\n')
        text_lines.extend([{'page': page['page'], 'text': t.replace(' ', '')} \
                           for t in page_lines if len(t.replace(' ', '')) > 0])
    matched_line_index = None

    for i in range(len(text_lines)):
        find = False
        for keyword in required_line_keywords:
            if keyword in text_lines[i]['text']:
                if DEBUG:
                    print(keyword, text_lines[i]['page'])
                find = True
                break
        for keyword in invalid_line_keywords:
            if keyword in text_lines[i]['text']:
                if DEBUG:
                    print('Filter as invalid line keyword {}'.format(keyword))
                find = False
        if not find:
            continue
        start = max(0, i - max_continuous_lines)
        pre_text = '\n'.join([t['text'] for t in text_lines[start:i]])
        end = min(len(text_lines), i + max_continuous_lines)
        post_text = '\n'.join([t['text'] for t in text_lines[i:end]])
        num_match = 0
        for keyword in required_post_keywords:
            if keyword in post_text:
                num_match += 1
        if num_match < min_match_number:
            if DEBUG:
                print('Filter as not enough post keywords {}<{}'.format(num_match, min_match_number))
                if text_lines[i]['page'] == 154:
                    print(post_text)
            find = False
        for invalid_keyword in invalid_pre_keywords:
            if invalid_keyword in pre_text:
                print('Filter as invalid pre keyword {}'.format(invalid_keyword))
                find = False
        for invalid_keyword in invalid_post_keywords:
            if invalid_keyword in post_text:
                if DEBUG:
                    print('Filter as invalid post keyword {}'.format(invalid_keyword))
                find = False
        if find:
            matched_line_index = i
            break
    if len(pure_text) > 0:
        max_page_id = np.max([t['page'] for t in pure_text])
    else:
        max_page_id = None
    print(max_page_id)
    if matched_line_index is None:
        return None, max_page_id
    else:
        return text_lines[matched_line_index]['page'], max_page_id


# 从 PDF 文件中提取符合特定关键词条件的表格
def extract_table_for_rows(pdf_name, max_continuous_lines=100, min_match_number=0, required_line_keywords=[],
                           invalid_line_keywords=[], required_post_keywords=[], invalid_pre_keywords=[],
                           invalid_post_keywords=[], prefix_pages=1, post_pages=1):
    match_page_id, max_page_id = find_match_page(pdf_name, max_continuous_lines, min_match_number,
                                                 required_line_keywords, invalid_line_keywords, required_post_keywords,
                                                 invalid_pre_keywords, invalid_post_keywords)
    print(match_page_id)
    tables = []
    if match_page_id is None:
        return tables
    pdf_path = get_raw_pdf_path(pdf_name)
    extractor = PdfExtractor(pdf_path)
    page_start = max(0, match_page_id - prefix_pages)
    page_end = min(match_page_id + post_pages + 1, max_page_id + 1)
    try:
        near_tables = extractor.extract_table_of_pages(range(page_start, page_end))
    except Exception as e:
        print(e, pdf_name, 'parse error')
        near_tables = []
    for table in near_tables:
        table_text = table.df.to_string().replace(' ', '')
        has_match = False
        for name in required_post_keywords:
            if name in table_text:
                has_match = True
                break
        if has_match:
            tables.append(table)
    return tables


# 过滤掉包含无效关键词的表格
def filter_tables(tables, invalid_keywords):
    filtered_tables = []
    for table in tables:
        table_text = table.df.to_string().replace(' ', '')
        has_match = False
        for keyword in invalid_keywords:
            if keyword in table_text:
                has_match = True
                break
        if not has_match:
            if table_text.count('指') < 5:
                filtered_tables.append(table)
    if len(filtered_tables) == 0:
        filtered_tables = tables
    return filtered_tables


# 将提取到的表格数据以文本形式保存到指定文件中
def tables_to_file(tables, file_path):
    text = ''
    for table in tables:
        text += 'page|{}\n'.format(table.page)
        for i, row in table.df.iterrows():
            row = [re.sub('\n', '', t) for t in row]
            text += '|'.join(row) + '\n'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


# 从 PDF 文件中提取公司基本信息表格，并将其保存到指定文件中
def extract_basic_info(key):
    logger.info('Extract basic info for {}'.format(key))
    bi_path = get_pdf_table_path(key)['basic_info']
    row_keywords = ['股票简称', '股票代码', '中文简称', '外文名称', '法定代表人',
                    '注册地址', '邮政编码', '办公地址', '电子信箱']
    bi_tables = extract_table_for_rows(key, int(len(row_keywords) * 1.5),
                                       min_match_number=0.3 * len(row_keywords),
                                       required_line_keywords=['公司简介', '基本情况', '公司信息', '中文简称',
                                                               '电子信箱'],
                                       invalid_line_keywords=[],
                                       required_post_keywords=row_keywords,
                                       invalid_pre_keywords=[],
                                       invalid_post_keywords=[],
                                       prefix_pages=1, post_pages=1)
    invalid_keywords = ['会计师事务所', '董事会秘书', '证券事务代表', '保荐机构',
                        '公司股票简况', '持续督导', '变更前股票']
    filtered_tables = filter_tables(bi_tables, invalid_keywords)
    if DEBUG:
        for table in filtered_tables:
            print(table.page, '*' * 30)
            print(table.df)
    tables_to_file(filtered_tables, bi_path)


# 从 PDF 文件中提取员工信息表格，并将其保存到指定文件中
def extract_employee_info(key):
    logger.info('Extract employee info for {}'.format(key))
    ei_path = get_pdf_table_path(key)['employee_info']
    row_keywords = ['在职员工', '职工人数', '专业构成', '离退休职',
                    '生产人员', '销售人员', '技术人员',
                    '行政人员', '管理人员', '业务人员',
                    '教育程度', '硕士', '本科', '大专', '研究生',
                    '专科']
    ei_tables = extract_table_for_rows(key, int(1.5 * len(row_keywords)),
                                       min_match_number=0.3 * len(row_keywords),
                                       required_line_keywords=['员工情况', '员工的数量', '员工数量', '专业构成',
                                                               '离退休职工人数', '员工教育结构'],
                                       invalid_line_keywords=[],
                                       required_post_keywords=row_keywords,
                                       invalid_pre_keywords=[],
                                       invalid_post_keywords=[],
                                       prefix_pages=1, post_pages=1)
    invalid_keywords = ['其他单位', '董事', '经理', '审议']
    filtered_tables = filter_tables(ei_tables, invalid_keywords)
    if DEBUG:
        for table in filtered_tables:
            print(table.page, '*' * 30)
            print(table.df)
    tables_to_file(filtered_tables, ei_path)


# 对表格组进行排序，优先按页码排序，若页码相同，则按位置排序
def sort_table_groups(table_groups):
    def sort_func(a, b):
        a_page = np.mean([t.page for t in a])
        b_page = np.mean([t.page for t in b])
        if a_page != b_page:
            return a_page - b_page
        else:
            a_top = np.mean([t._bbox[1] for t in a])
            b_top = np.mean([t._bbox[1] for t in b])
            return b_top - a_top

    table_groups = sorted(table_groups, key=cmp_to_key(sort_func))
    return table_groups


# 对表格列表进行排序，优先按页码排序，若页码相同，则按位置排序
def sort_tables(tables):
    def sort_func(a, b):
        if a.page != b.page:
            return a.page - b.page
        else:
            return b._bbox[1] - a._bbox[1]

    tables = sorted(tables, key=cmp_to_key(sort_func))
    return tables


# 从表格列表中移除重叠过多的表格
def remove_overlap_tables(tables, valid_overlap_words=[], maximum_overlap_words=2):
    def get_row_names(table):
        names = table.df.iloc[:, 0]
        names = [clean_row_name(name) for name in names]
        names = [name for name in names if name != '']
        return set(names)

    idx_to_remove = []
    table_row_names = [get_row_names(table) for table in tables]
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            num_overlap_words = 0
            overlap_words = []
            for name_i in table_row_names[i]:
                for name_j in table_row_names[j]:
                    if name_i == name_j:
                        is_valid_overlap = False
                        for word in valid_overlap_words:
                            if word in name_i:
                                is_valid_overlap = True
                        if not is_valid_overlap:
                            num_overlap_words += 1
                            overlap_words.append(name_i)
            if num_overlap_words >= maximum_overlap_words:
                idx_to_remove.append(j)
                if DEBUG:
                    print(tables[j].page, overlap_words)
    filtered_tables = []
    for i in range(len(tables)):
        if i not in idx_to_remove:
            filtered_tables.append(tables[i])
    return filtered_tables


# 移除同一页上的重复表格，仅保留关键字匹配数量最多的表格
def remove_tables_same_page_by_keywords(tables, keywords):
    def get_overlap_count(table, words):
        count = 0
        for word in words:
            if word in table.df.to_string():
                count += 1
        return count

    table_keyword_counts = [get_overlap_count(table, keywords) for table in tables]
    idx_to_remove = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            if tables[i].page == tables[j].page:
                if table_keyword_counts[i] > table_keyword_counts[j]:
                    idx_to_remove.append(j)
                elif table_keyword_counts[i] < table_keyword_counts[j]:
                    idx_to_remove.append(i)
    filtered_tables = []
    for i in range(len(tables)):
        if i not in idx_to_remove:
            filtered_tables.append(tables[i])
    return filtered_tables


# 从表格列表中移除跨页的表格
def remove_tables_over_pages(tables):
    # 删除跨页的
    idx_to_remove = []
    for i in range(len(tables)):
        j = i + 1
        if j >= len(tables):
            continue
        if tables[j].page - tables[i].page > 1:
            idx_to_remove.append(j)
    filtered_tables = []
    for i in range(len(tables)):
        if i not in idx_to_remove:
            filtered_tables.append(tables[i])
    return filtered_tables


# 从 PDF 文件中提取资产负债（cbs）表信息
def extract_cbs_info(key):
    logger.info('Extract cbs info for {}'.format(key))
    cbs_path = get_pdf_table_path(key)['cbs_info']
    row_names = [
        '流动资产', '货币资金', '结算备付金', '拆出资金', '交易性金融资产',
        '应收保费', '应收分保账款', '应收分保合同准备金', '其他应收款',
        '存货', '合同资产', '持有待售资产', '返售金融资产',
        '非流动资产', '债权投资', '其他债权投资', '长期应收款',
        '固定资产', '油气资产', '商誉', '无形资产',
        '递延所得税资产', '其他非流动资产', '资产合计', '资产总计',
        '应付账款', '预收款项', '合同负债', '应付职工薪酬',
        '持有待售负债', '一年内到期的非流动负债', '其他流动负债', '流动负债合计',
        '长期应付职工薪酬', '预计负债', '递延收益', '递延所得税负债',
        '其他权益工具', '优先股', '资本公积', '负债合计', '股本',
        '未分配利润', '股东权益合计',
        '所有者权益', '少数股东权益'
    ]
    cbs_tables = extract_table_for_rows(key, int(1.5 * len(row_names)),
                                        min_match_number=0.3 * len(row_names),
                                        required_line_keywords=['资产负债表'],
                                        invalid_line_keywords=['母公司资产负债表'],
                                        required_post_keywords=row_names,
                                        invalid_pre_keywords=[],
                                        invalid_post_keywords=['调整', '变动比例', '变更'],
                                        prefix_pages=0, post_pages=5)
    if len(cbs_tables) == 0:
        cbs_tables = extract_table_for_rows(key, int(1.5 * len(row_names)),
                                            min_match_number=0.2 * len(row_names),
                                            required_line_keywords=['合并及母公司资产负债表'],
                                            invalid_line_keywords=[],
                                            required_post_keywords=row_names,
                                            invalid_pre_keywords=[],
                                            invalid_post_keywords=['调整', '变更'],
                                            prefix_pages=0, post_pages=5)

    if len(cbs_tables) == 0:
        cbs_tables = extract_table_for_rows(key, int(2 * len(row_names)),
                                            min_match_number=0.2 * len(row_names),
                                            required_line_keywords=['合并资产负债表和资产负债表'],
                                            invalid_line_keywords=[],
                                            required_post_keywords=row_names,
                                            invalid_pre_keywords=[],
                                            invalid_post_keywords=['调整', '变更'],
                                            prefix_pages=0, post_pages=5)

    invalid_keywords = ['非流动资产处置损益']
    filtered_tables = filter_tables(cbs_tables, invalid_keywords)
    filtered_tables = sort_tables(filtered_tables)
    filtered_tables = remove_overlap_tables(filtered_tables,
                                            valid_overlap_words=['优先股', '永续债', '金额单位', '年度报告'])
    filtered_tables = remove_tables_same_page_by_keywords(filtered_tables, row_names)
    filtered_tables = remove_tables_over_pages(filtered_tables)
    filtered_tables = remove_tables_over_pages(filtered_tables)
    if DEBUG:
        for table in filtered_tables:
            print(table.page, '*' * 30)
            print(table.df)
    tables_to_file(filtered_tables, cbs_path)


# 从 PDF 文件中提取现金流量（cscf）表信息
def extract_cscf_info(key):
    logger.info('Extract cscf info for {}'.format(key))
    row_keywords = [
        '经营活动产生的', '收到的现金', '客户存款',
        '其他金融机构', '中央银行', '原保险合同',
        '收取利息', '拆入资金', '买卖证券',
        '税费返还', '经营活动有关',
        '经营活动现金', '支付的现金',
        '客户贷款及垫款', '原保险合同',
        '拆出资金', '手续费及佣金',
        '支付给职工', '支付的各项税费',
        '投资活动现金', '收回投资',
        '处置固定资产', '处置子公司',
        '投资支付', '筹资活动',
        '汇率变动', '现金及现金等价物',
        '现金等价物余额'
    ]
    cscf_tables = extract_table_for_rows(key, 50,
                                         min_match_number=0.3 * len(row_keywords),
                                         required_line_keywords=['合并现金流量表'],
                                         invalid_line_keywords=['母公司现金流量表'],
                                         required_post_keywords=row_keywords,
                                         invalid_pre_keywords=[],
                                         invalid_post_keywords=['变动比例', '调整', '变更'],
                                         prefix_pages=0, post_pages=3)
    if len(cscf_tables) == 0:
        cscf_tables = extract_table_for_rows(key, 50,
                                             min_match_number=0.3 * len(row_keywords),
                                             required_line_keywords=['现金流量表'],
                                             invalid_line_keywords=['母公司现金流量表'],
                                             required_post_keywords=row_keywords,
                                             invalid_pre_keywords=[],
                                             invalid_post_keywords=['变动比例', '调整', '变更'],
                                             prefix_pages=0, post_pages=3)
    cscf_tables = sort_tables(cscf_tables)
    filtered_tables = remove_overlap_tables(cscf_tables, valid_overlap_words=['净额', '人民币'])
    filtered_tables = remove_tables_same_page_by_keywords(filtered_tables, row_keywords)
    filtered_tables = remove_tables_over_pages(filtered_tables)
    if DEBUG:
        for table in filtered_tables:
            print(table.page, '*' * 30)
            print(table.df)
    cscf_path = get_pdf_table_path(key)['cscf_info']
    tables_to_file(filtered_tables, cscf_path)


# 从 PDF 文件中提取利润（cis）表信息
def extract_cis_info(key):
    logger.info('Extract cis info for {}'.format(key))
    row_keywords = [
        '营业总收入', '营业收入', '利息收入', '已赚保费',
        '手续费及佣金收入', '营业总成本', '营业成本', '利息支出',
        '保单红利支出', '分保费用', '营业税金及附加', '销售费用',
        '管理费用', '研发费用', '财务费用', '利息费用',
        '以摊余成本计量', '汇兑收益', '净敞口套期收益', '公允价值变动收益',
        '营业外收入', '营业外支出', '利润总额', '所得税费用',
        '少数股东损益', '其他综合收益的税后净额', '收益的税后净额', '不能重分类进损益',
        '权益法下可转损益', '其他债权投资公允价值', '可供出售金融资产', '金融资产重分类计入',
        '外币财务报表', '归属于少数股东的',
        '归属于少数股东的',
        '基本每股收益', '稀释每股收益']
    cis_tables = extract_table_for_rows(key, int(1.5 * len(row_keywords)),
                                        min_match_number=0.3 * len(row_keywords),
                                        required_line_keywords=['利润表'],
                                        invalid_line_keywords=['母公司利润表'],
                                        required_post_keywords=row_keywords,
                                        invalid_pre_keywords=[],
                                        invalid_post_keywords=['变动比例', '变更', '调整'],
                                        prefix_pages=0, post_pages=3)
    filtered_tables = sort_tables(cis_tables)
    filtered_tables = remove_overlap_tables(filtered_tables, valid_overlap_words=['利息收入', '项目'])
    filtered_tables = remove_tables_same_page_by_keywords(filtered_tables, row_keywords)
    filtered_tables = remove_tables_over_pages(filtered_tables)
    if DEBUG:
        for table in filtered_tables:
            print('*' * 50, table.page)
            print(table.df)
    cis_path = get_pdf_table_path(key)['cis_info']
    tables_to_file(filtered_tables, cis_path)


# 从 PDF 文件中提取研发信息
def extract_dev_info(key):
    logger.info('Extract dev info for {}'.format(key))
    row_keywords = ['研发人员数量', '研发人员的数量']
    dev_tables = extract_table_for_rows(key, int(1.5 * len(row_keywords)),
                                        min_match_number=0.2 * len(row_keywords),
                                        required_line_keywords=['研发人员数量', '研发人员的数量'],
                                        invalid_line_keywords=[],
                                        required_post_keywords=row_keywords,
                                        invalid_pre_keywords=[],
                                        invalid_post_keywords=[],
                                        prefix_pages=0, post_pages=2)
    filtered_tables = sort_tables(dev_tables)
    if DEBUG:
        for table in filtered_tables:
            print('*' * 50, table.page)
            print(table.df)
    dev_path = get_pdf_table_path(key)['dev_info']
    tables_to_file(filtered_tables, dev_path)


# 清理指定类型的表格文件
def clean_info(table_name):
    pdf_info = load_pdf_info()
    for key in list(pdf_info.keys()):
        table_path = get_pdf_table_path(key)[table_name]
        if os.path.exists(table_path):
            logger.info('Remove {}'.format(table_path))
            os.remove(table_path)


# 将所有 PDF 文件中提取的指定类型表格内容合并到一个 JSON 文件中
def merge_info(table_name):
    pdf_info = load_pdf_info()
    for key in list(pdf_info.keys()):
        table_path = get_pdf_table_path(key)[table_name]
        if not os.path.exists(table_path):
            del pdf_info[key]
            continue
        with open(table_path, 'r', encoding='utf-8') as f:
            table_content = f.readlines()
        pdf_info[key][table_name] = table_content
    with open(os.path.join(cfg.DATA_PATH, '{}.json'.format(table_name)), 'w', encoding='utf-8') as f:
        json.dump(pdf_info, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    pdf_info = load_pdf_info()
    pdf_keys = list(pdf_info.keys())

    table_name = 'cis_info'
    with Pool(processes=32) as pool:
        results = pool.map(extract_dev_info, pdf_keys)
    merge_info(table_name)
