import re
import os
from collections import Counter
import json
import sqlite3
import numpy as np
import pandas as pd
from config import cfg
from utils.file import load_pdf_info, load_tables_of_years
from utils.file import load_total_tables


# 统计表格中每个键的出现频率，并将结果保存到一个 JSON 文件中
def count_table_keys():
    pdf_info = load_pdf_info()
    all_tables = load_total_tables()
    all_keys = []

    for pdf_key, pdf_item in list(pdf_info.items()):
        company = pdf_item['company']
        year = pdf_item['year'].replace('年', '')
        table = load_tables_of_years(company, [year], all_tables, pdf_info)
        row_names = list(set([t[2] for t in table]))
        all_keys.extend(row_names)
    all_keys = Counter(all_keys)

    with open(os.path.join(cfg.DATA_PATH, 'key_count.json'), 'w', encoding='utf-8') as f:
        json.dump(all_keys, f, ensure_ascii=False, indent=4)


# 根据键的出现频率筛选出常用的键，并构建一个公司表格，将结果保存为 CSV 文件
def build_table(min_ratio=0.1):
    pdf_info = load_pdf_info()
    all_tables = load_total_tables()

    with open(os.path.join(cfg.DATA_PATH, 'key_count.json'), 'r', encoding='utf-8') as f:
        key_count = json.load(f)
    max_count = max(key_count.values())
    key_count = sorted(key_count.items(), key=lambda x: x[1], reverse=True)
    used_keys = [key for key, count in key_count if count > min_ratio * max_count]
    columns = ['公司全称', '年份'] + used_keys
    df_dict = {}

    for col in columns:
        df_dict[col] = []
    for pdf_key, pdf_item in list(pdf_info.items()):
        company = pdf_item['company']
        year = pdf_item['year'].replace('年', '')
        table = load_tables_of_years(company, [year], all_tables, pdf_info)
        df_dict['公司全称'].append(company)
        df_dict['年份'].append(year)
        for key in used_keys:
            value = 'NULLVALUE'
            for table_name, year, row_name, row_value in table:
                if year != year:
                    continue
                if row_name == key:
                    value = row_value
                    break
            value = value.replace('人', '').replace('元', '').replace(' ', '')
            df_dict[key].append(value)
    pd.DataFrame(df_dict).to_csv(os.path.join(cfg.DATA_PATH, 'CompanyTable.csv'), sep='\t', index=False,
                                 encoding='utf-8')


# 加载公司表格，并根据 PDF 文件信息进行筛选
def load_company_table():
    df_path = os.path.join(cfg.DATA_PATH, 'CompanyTable.csv')
    df = pd.read_csv(df_path, sep='\t', encoding='utf-8')
    df['key'] = df.apply(lambda t: t['公司全称'] + str(t['年份']), axis=1)
    pdf_info = load_pdf_info()
    company_keys = [v['company'] + v['year'].replace('年', '').replace(' ', '') for v in pdf_info.values()]
    df = df[df['key'].isin(company_keys)]
    del df['key']
    return df


# 将输入的字符串尝试转换为数值类型
def col_to_numeric(t):
    try:
        value = float(t)
        if value > 2 ** 63 - 1:
            return np.nan
        elif int(value) == value:
            return int(value)
        else:
            return float(t)
    except:
        return np.nan


# 将一个公司表格加载到内存中的 SQLite 数据库，并返回一个游标用于执行 SQL 查询
def get_sql_search_cursor():
    conn = sqlite3.connect(':memory:')
    df = load_company_table()
    dtypes = {}
    for col in df.columns:
        num_count = 0
        tot_count = 0
        for v in df[col]:
            if v == 'NULLVALUE':
                continue
            tot_count += 1
            try:
                number = float(v)
            except ValueError:
                continue
            num_count += 1
        if tot_count > 0 and num_count / tot_count > 0.5:
            print('Find numeric column {}, number count {}, total count {}'.format(col, num_count, tot_count))
            df[col] = df[col].apply(lambda t: col_to_numeric(t)).replace([np.inf, -np.inf], np.nan)
            dtypes[col] = 'REAL'
        else:
            dtypes[col] = 'TEXT'
    dtypes['年份'] = 'TEXT'
    df.to_sql(name='company_table', con=conn, if_exists='replace', dtype=dtypes)
    cursor = conn.cursor()
    return cursor


# 通过给定的 SQLite 游标执行 SQL 查询，并返回查询结果
def get_search_result(cursor, query):
    result = cursor.execute(query)
    return result


def get_cn_en_key_map(model, keys):
    def get_en_key(cn_key):
        prompt = '''
    你的任务是将中文翻译为英文短语。
    注意：
    1. 你只需要回答英文短语，不要进行解释或者回答其他内容。
    2. 尽可能简短的回答。
    3. 你输出的格式是:XXX对应的英文短语是XXXXX。
    -----------------------
    需要翻译的中文为：{}
    '''.format(cn_key)
        en_key = model(prompt)
        print(en_key)
        en_key = ' '.join(re.findall('[ a-zA-Z]+', en_key)).strip(' ').split(' ')
        en_key = [w[0].upper() + w[1:] for w in en_key if len(w) > 1]
        en_key = '_'.join(en_key)
        return en_key

    en_keys = [get_en_key(key) for key in keys]
    key_map = dict(zip(keys, en_keys))

    with open(os.path.join(cfg.DATA_PATH, 'key_map.json'), 'w', encoding='utf-8') as f:
        json.dump(key_map, f, ensure_ascii=False, indent=4)


def load_cn_en_key_map():
    with open(os.path.join(cfg.DATA_PATH, 'key_map.json'), 'r', encoding='utf-8') as f:
        key_map = json.load(f)
    return key_map


# 对加载的公司表格进行检查和筛选，并将特定列保存为一个新的 CSV 文件
def check_company_table():
    df = load_company_table()
    df['key'] = df.apply(lambda t: t['公司全称'] + str(t['年份']), axis=1)
    with open(os.path.join(cfg.DATA_PATH, 'B-pdf-name.txt'), 'r', encoding='utf-8') as f:
        pdf_names = [t.strip() for t in f.readlines()]
    pdf_info = load_pdf_info()
    B_pdf_keys = []
    for pdf_name, pdf_item in pdf_info.items():
        if pdf_name not in pdf_names:
            continue
        B_pdf_keys.append(pdf_item['company'] + pdf_item['year'].replace('年', ''))
    print(B_pdf_keys[:10])
    cols = ['公司全称', '年份', '其他非流动资产', '利润总额', '负债合计', '营业成本',
            '注册地址', '流动资产合计', '营业收入', '货币资金', '资产总计']
    df.loc[:, cols].to_csv(os.path.join(cfg.DATA_PATH, 'B_CompanyTable.csv'), index=False, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    count_table_keys()
    build_table()
