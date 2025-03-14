import os
import json
import re
import pandas as pd
from loguru import logger
from config import cfg
from utils import re_util


# 遍历所有PDF文件，并存储它们的文件名
def extract_pdf_filenames(folder_path, output_file):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for root, dirs, files in os.walk(folder_path):
                pdf_files = [file for file in files if file.endswith(".pdf")]
                for pdf_file in pdf_files:
                    f.write(pdf_file + "\n")
        print(f"PDF文件名已成功写入到 {output_file} 中。")
        return True
    except FileNotFoundError:
        print(f"错误：文件夹 {folder_path} 不存在，请检查路径是否正确。")


# 从一个文本文件中读取PDF文件名列表，解析每个文件名以提取关键信息
def download_data():
    with open(cfg.DATA_PATH + 'pdf_name.txt', 'r', encoding='utf-8') as f:
        pdf_names = [line.strip('\n') for line in f.readlines()]
        print(pdf_names[:10])
    ds = {}

    for name in pdf_names:
        pdf_path = os.path.join(cfg.DATA_PATH + 'all_pdf', name)
        split = name.split('__')
        ds[name] = {
            'key': name,
            'pdf_path': pdf_path,
            'company': split[1],
            'code': split[2],
            'abbr': split[3],
            'year': split[4]
        }

    with open(os.path.join(cfg.DATA_PATH, 'pdf_info.json'), 'w', encoding='utf-8') as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)


# 读取文件中的每一行，返回一个包含所有解析后的json文件列表
def load_test_questions():
    path = os.path.join(cfg.DATA_PATH, 'test/question.json')
    with open(path, 'r', encoding='utf-8') as f:
        test_questions = [json.loads(line) for line in f.readlines()]
    return test_questions


# 读取pdf_info.json文件中的信息，并存储在一个字典中返回
def load_pdf_info():
    with open(os.path.join(cfg.DATA_PATH, 'pdf_info.json'), 'r', encoding='utf-8') as f:
        pdf_info = json.load(f)
    return pdf_info


# 获取某个PDF文件的路径
def get_raw_pdf_path(key):
    pdf_info = load_pdf_info()
    return pdf_info[key]['pdf_path']


def load_total_tables():
    key_and_paths = [
        ('basic_info', os.path.join(cfg.DATA_PATH, 'basic_info.json')),
        ('employee_info', os.path.join(cfg.DATA_PATH, 'employee_info.json')),
        ('cbs_info', os.path.join(cfg.DATA_PATH, 'cbs_info.json')),
        ('cscf_info', os.path.join(cfg.DATA_PATH, 'cscf_info.json')),
        ('cis_info', os.path.join(cfg.DATA_PATH, 'cis_info.json')),
        ('dev_info', os.path.join(cfg.DATA_PATH, 'dev_info.json')),
    ]
    tables = {}
    for key, path in key_and_paths:
        with open(path, 'r', encoding='utf-8') as f:
            tables[key] = json.load(f)
    return tables


def get_pdf_table_path(key):
    path = {
        'basic_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'basic_info.txt'),
        'employee_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'employee_info.txt'),
        'cbs_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'cbs_info.txt'),
        'cscf_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'cscf_info.txt'),
        'cis_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'cis_info.txt'),
        'dev_info': os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'dev_info.txt'),
    }
    return path


# 遍历 all_tables 字典，加载与 key 相关的表格数据
def load_pdf_tables(key, all_tables):
    tables = {}
    for k, v in all_tables.items():
        if key in v.keys() and k in v[key]:
            lines = v[key][k]
            lines = [re_util.sep_numbers(line) for line in lines]
            tables[k] = lines
        else:
            logger.warning('{} not in {}'.format(key, k))
            tables[k] = []
    return tables


# 清理和解析表格行数据，将其转换为标准化的元组列表
def basic_info_to_tuple(year, table_lines):
    tuples = []
    for line in table_lines:
        if 'page' in line:
            continue
        line = line.strip('\n').split('|')
        line_text = []
        for sp in line:
            if sp == '':
                continue
            sp = sp.replace(' ', '').replace('"', '')
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 1:
            row_name = line_text[0]
            row_name = re.sub('[(（].*[）)]', '', row_name)
            row_name = re.sub('(公司|的)', '', row_name)
            # row_name = '"{}"'.format(row_name)
        if len(line_text) == 1:
            tuples.append(('basic_info', year, row_name, ''))
        elif len(line_text) == 2:
            tuples.append(('basic_info', year, row_name, line_text[1]))
        elif len(line_text) == 3:
            tuples.append(('basic_info', year, row_name, '|'.join(line_text[1:])))
        elif len(line_text) >= 4:
            tuples.append(('basic_info', year, row_name, line_text[1]))
            tuples.append(('basic_info', year, line_text[2], line_text[3]))
    return tuples


# 清理和解析员工信息表格行数据，将其转换为标准化的元组列表
def employee_info_to_tuple(year, table_lines):
    tuples = []
    for line in table_lines:
        if 'page' in line:
            continue
        line = line.strip('\n').split('|')
        line_text = []
        for sp in line:
            if sp == '':
                continue
            sp = re.sub('[ ,]', '', sp)
            sp = re.sub('[(（]人[）)]', '', sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 2:
            try:
                number = float(line_text[1])
                row_name = line_text[0]
                tuples.append(('employee_info', year, row_name, line_text[1] + '人'))
            except:
                continue
    return tuples


# 清理和解析财务信息表格行数据，将其转换为标准化的元组列表。
def fs_info_to_tuple(pdf_key, table_name, year, table_lines):
    unit = get_unit(pdf_key, table_lines)
    tuples = []
    page_id = None
    for line in table_lines:
        if 'page' in line:
            page_id = line.split('page')[1]
            continue
        line = line.strip('\n').split('|')
        line_text = []
        for sp in line:
            if sp == '':
                continue
            sp = re.sub('[ ,]', '', sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) == 1:
            continue
        if len(line_text) >= 2:
            row_name = line_text[0]
            row_name = re.sub('[\d \n\.．]', '', line[0])
            row_name = re.sub('（[一二三四五六七八九十]）', '', row_name)
            row_name = re.sub('\([一二三四五六七八九十]\)', '', row_name)
            row_name = re.sub('[一二三四五六七八九十][、.]', '', row_name)
            row_name = re.sub('其中：', '', row_name)
            row_name = re.sub('[加减]：', '', row_name)
            row_name = re.sub('（.*）', '', row_name)
            row_name = re.sub('\(.*\)', '', row_name)
            if row_name == '':
                continue
            row_values = []
            for value in line_text[1:]:
                if value == '' or value == '-':
                    continue
                if set(value).issubset(set('0123456789.,-')):
                    try:
                        if re_util.is_valid_number(value):
                            row_values.append('{:.2f}元'.format(float(value) * unit))
                    except:
                        logger.error('Invalid value {} {} {}'.format(value, pdf_key, table_name))
                        row_values.append(value + '元')
            # print(line_text)
            # print(row_values, '----')
            if len(row_values) == 1:
                # logger.warning('Invalid line(2 values) {} in {} {}'.format(line_text, table_name, year))
                tuples.append((table_name, year, row_name, row_values[0]))
            elif len(row_values) == 2:
                tuples.append((table_name, year, row_name, row_values[0]))
                tuples.append((table_name, str(int(year) - 1), row_name, row_values[1]))
            elif len(row_values) >= 3:
                tuples.append((table_name, year, row_name, row_values[1]))
                tuples.append((table_name, str(int(year) - 1), row_name, row_values[2]))
    return tuples


# 从给定的 PDF 文件中提取表格数据的单位，并返回对应的数值单位
def get_unit(pdf_key, table):
    unit = 1
    if len(table) == 0:
        return unit
    page_num = table[0].strip().split('|')[1]
    pages = load_pdf_pure_text(pdf_key)
    for idx, page_item in enumerate(pages):
        if str(page_item['page']) == page_num:
            last_page_lines = []
            if idx > 0:
                last_page_lines = pages[idx - 1]['text'].split('\n')[-10:]
            current_page_lines = page_item['text'].split('\n')
            search_string = None
            for line in last_page_lines + current_page_lines:
                re_unit = re.findall('单位\s*[:：；].{0,3}元', line) + \
                          re.findall('人民币.{0,3}元', line)
                if len(re_unit) != 0:
                    search_string = re_unit[0]
                    break
            if search_string is not None:
                if '百万' in search_string:
                    unit = 1000000
                elif '万' in search_string:
                    unit = 10000
                elif '千' in search_string:
                    unit = 1000
                else:
                    pass
            else:
                print('cannot find unit for key {} page {}'.format(pdf_key, page_num))
                print(page_item['text'])
            if unit != 1:
                print(pdf_key)
                print(search_string)
                print(page_item['text'])
            break
    if unit != 1:
        logger.info('{}的单位是{}'.format(pdf_key, unit))
    return unit


# 清理和解析表格数据，提取与“研发人员”相关的行，并将其转换为标准化的元组列表
def dev_info_to_tuple(year, table_lines):
    tuples = []
    for line in table_lines:
        if 'page' in line:
            continue
        if not '研发人员' in line:
            continue
        line = line.strip('\n').split('|')
        line_text = []
        for sp in line:
            if sp == '':
                continue
            sp = re.sub('[ ,]', '', sp)
            sp = re.sub('[(（]人[）)]', '', sp)
            if len(line_text) >= 1 and line_text[-1] == sp:
                continue
            line_text.append(sp)
        if len(line_text) >= 2:
            tuples.append(('dev_info', year, line_text[0], line_text[1] + '人'))
    return tuples


# 根据表格类型选择合适的处理函数，将表格数据转换为标准化的元组列表
def table_to_tuples(pdf_key, year, table_name, table_lines):
    if table_name == 'basic_info':
        return basic_info_to_tuple(year, table_lines)
    elif table_name == 'employee_info':
        return employee_info_to_tuple(year, table_lines)
    elif table_name == 'dev_info':
        return dev_info_to_tuple(year, table_lines)
    else:
        return fs_info_to_tuple(pdf_key, table_name, year, table_lines)


# 通过遍历年份，加载指定公司的表格数据，将数据转换为标准化的元组列表。
def load_tables_of_years(company, years, pdf_tables, pdf_info):
    table = []
    for year in years:
        year = year.replace('年', '')
        pdf_key = None
        for k, v in pdf_info.items():
            if v['company'] == company and year in v['year']:
                pdf_key = k
        if pdf_key is None:
            logger.error('Cannot find pdf key for {} {}'.format(company, year))
            continue
        year_tables = load_pdf_tables(pdf_key, pdf_tables)
        for table_name, table_lines in year_tables.items():
            table.extend(table_to_tuples(pdf_key, year, table_name, table_lines))
    alias = {
        '在职员工的数量合计': '职工总人数',
        '负债合计': '总负债',
        '资产总计': '总资产',
        '流动负债合计': '流动负债',
        '非流动负债合计': '非流动负债',
        '流动资产合计': '流动资产',
        '非流动资产合计': '非流动资产'
    }
    new_table = []
    for row in table:
        table_name, row_year, row_name, row_value = row
        new_table.append((table_name, row_year, row_name, row_value))
        if row_name in alias:
            new_table.append((table_name, row_year, alias[row_name], row_value))
    return new_table


# 将元组列表转换为 Pandas DataFrame，并进行清理和排序，生成一个结构化的数据表
def table_to_dataframe(table_rows):
    df = pd.DataFrame(table_rows, columns=['table_name', 'row_year', 'row_name', 'row_value'])
    df['row_year'] = pd.to_numeric(df['row_year'])
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['row_name', 'row_year'], inplace=True)
    return df


# 计算同一字段在相邻年份的增长率，并将结果添加到原始数据中，扩展表格数据的内容。
def add_growth_rate_in_table(table_rows):
    df = table_to_dataframe(table_rows)
    added_rows = []
    for idx, (index, row) in enumerate(df.iterrows()):
        last_row = df.iloc[idx - 1]
        if last_row['row_name'] == row['row_name'] and last_row['row_year'] == row['row_year'] - 1:
            last_values = re_util.find_numbers(last_row['row_value'])
            current_values = re_util.find_numbers(row['row_value'])
            if len(last_values) > 0 and len(current_values) > 0:
                if last_values[0] != 0:
                    growth_rate = (current_values[0] - last_values[0]) / last_values[0] * 100
                    added_rows.append([row['table_name'], str(row['row_year']), row['row_name'] + '增长率',
                                       '{:.2f}%'.format(growth_rate)])
    merged_rows = table_rows + added_rows
    return merged_rows


# 比较同一字段在不同年份的文本值，并将比较结果添加到原始数据中，扩展表格数据的内容
def add_text_compare_in_table(table_rows):
    df = table_to_dataframe(table_rows)
    added_rows = []
    for idx, (index, row) in enumerate(df.iterrows()):
        if idx == 0:
            continue
        last_row = df.iloc[idx - 1]
        if last_row['row_name'] == row['row_name']:
            last_values = re_util.find_numbers(last_row['row_value'])
            current_values = re_util.find_numbers(row['row_value'])
            if len(last_values) == 0 and len(current_values) == 0:
                if row['row_value'] != last_row['row_value']:
                    row_value = '不相同且不同'
                else:
                    row_value = '相同'
                added_rows.append([row['table_name'], '{}与{}相比'.format(row['row_year'], last_row['row_year']),
                                   row['row_name'], row_value])
    merged_rows = table_rows + added_rows
    return merged_rows


# 将表格数据转换为自然语言描述，生成了一段连贯的文本
def table_to_text(company, question, table_rows, with_year=True):
    text_lines = []
    for row in table_rows:
        table_name, row_year, row_name, row_value = row
        if table_name == 'basic_info':
            row_value = '"{}"'.format(row_value)
        else:
            row_name = '"{}"'.format(row_name)
        if not with_year:
            row_year = ''
        else:
            row_year += '年的'
        if row_value in ['相同', '不相同且不同']:
            # print(row_year, row_name, row_value)
            line = '{}的{}{},'.format(row_year, row_name, row_value)
        else:
            if table_name == 'employee_info':
                line = '{}{}有{},'.format(row_year, row_name, row_value)
            else:
                line = '{}{}是{},'.format(row_year, row_name, row_value)
        if line not in text_lines:
            text_lines.append(line)
    return ''.join(text_lines)


# 遍历页面内容，将文本和表格内容转换为纯文本格式
def load_pdf_pure_text(key):
    text_lines = []
    text_path = os.path.join(cfg.DATA_PATH, 'pdf_docs', key, 'pure_content.txt')
    if not os.path.exists(text_path):
        logger.warning('{} not exists'.format(text_path))
        return text_lines
    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            lines = f.readlines()
            text_lines = [json.loads(line) for line in lines]
            text_lines = sorted(text_lines, key=lambda x: x['page'])
            if len(text_lines) == 0:
                logger.warning('{} is empty'.format(text_path))
        except Exception as e:
            logger.error('Unable to load {}, {}'.format(text_path, e))
    return text_lines


# 加载和解析存储在all_txt文件中的纯文本内容，并过滤出满足条件的行
def load_pdf_pure_text_all_txt(key):
    text_lines = []
    text_path = os.path.join(cfg.DATA_PATH, 'all_txt', '{}.txt'.format(os.path.splitext(key)[0]))
    if not os.path.exists(text_path):
        logger.warning('{} not exists'.format(text_path))
        return text_lines
    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        raw_lines = [json.loads(line) for line in lines]
        for line in raw_lines:
            if 'type' not in line or 'inside' not in line:
                continue
            if len(line['inside'].replace(' ', '')) == 0:
                continue
            if line['type'] in ['页脚', '页眉']:
                continue
            if line['type'] == 'text':
                text_lines.append(line)
            elif line['type'] == 'excel':
                try:
                    row = eval(line['inside'])
                    line['inside'] = '\t'.join(row)
                    text_lines.append(line)
                except:
                    logger.warning('Invalid line {}'.format(line))
            else:
                logger.warning('Invalid line {}'.format(line))
        text_lines = sorted(text_lines, key=lambda x: x['allrow'])
        if len(text_lines) == 0:
            logger.warning('{} is empty'.format(text_path))
    return text_lines


# 加载和解析纯文本内容，并按页码分组
def load_pdf_pages(key):
    all_lines = load_pdf_pure_text_all_txt(key)
    pages = []
    if len(all_lines) == 0:
        return pages
    current_page_id = all_lines[0]['page']
    current_page = []
    for line in all_lines:
        if line['page'] == current_page_id:
            current_page.append(line)
        else:
            pages.append('\n'.join([t['inside'] for t in current_page]))
            current_page_id = line['page']
            current_page = [line]
    pages.append('\n'.join([t['inside'] for t in current_page]))
    return pages
