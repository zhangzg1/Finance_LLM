import re


def anoy_question(question, company, abbr, years):
    question_new = question
    if company in question_new:
        question_new = question_new.replace(company, 'XX公司')
    if abbr in question_new:
        question_new = question_new.replace(abbr, 'XX公司')
    for year in years:
        question_new = question_new.replace(year, 'XXXX')

    return question_new


def get_question_related_tables(model, question, company, abbr, years):
    question_new = anoy_question(question, company, abbr, years)
    prompt_classify_question = '''
请问“{}”是属于下面哪个类别的问题?
A: 基本信息查询, 例如证券信息、股票简称、股票代码、外文名称、法定代表人、注册地址、办公地址、公司网址、电子信箱等
B: 公司员工人数统计, 例如员工人数、员工专业、员工教育程度等
C: 财务相关, 例如金额、费用、资产、收入等
D: 以上都不是

例如:
1. XXXX的费用收入是多少元?
输出: C
2. XX公司法定代表人是谁?
输出: A
3. 请简要介绍分析XX公司的XXX情况。
输出: D
4. XX公司硕士人数是什么?
输出: B

你只需要回答编号, 不要回答其他内容.
'''.format(question_new)
    response_classify = model(prompt_classify_question)
    class_map = {
        'A': 'basic_info',
        'B': 'employee_info',
    }
    classes = re.findall('[A-F]+', response_classify)
    related_tables = [class_map[c] for c in classes if c in class_map]
    return related_tables


def get_prompt(question, company, abbr, years):
    comp = company if company in question else abbr
    if len(years) > 1:
        added_question = ''
        for i, year in enumerate(years):
            added_question += '{}. {}年的是?\n'.format(i + 2, year)
        prompt = '''
{{}}

******************************
请回答下面的问题:
1. {{}}
{}'''.format(added_question)

    else:
        answer_format = '{}年{}的XXXX是XXXX'.format(years[0], comp)
        prompt = """
{{}}

请回答问题: {{}}
注意你的回答应该按照以下要求:
1. 你回答的格式应该是:{}。
2. 你只需要回答问题相关的内容, 不要回答无关内容。
3. 你不需要进行计算。
4. 你的回答只能来源于提供的资料。""""".format(answer_format)

    return prompt
