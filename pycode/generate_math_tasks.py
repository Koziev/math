"""
Код генерации диалоговых сэмплов из шаблонов.
Использование:
1) импортируется как модуль в generate_npqa_arith.py при сборке датасета файнтюна
2) запускается в консоли для проверки валидности шаблона.
"""

import json
import os
import io
import traceback
import collections
import argparse
import glob
import logging
import re
import random

import pymorphy2
import ruword2tags

from generative_template import TemplatePattern


class CombinedInflector:
    def __init__(self):
        self.flexer = ruword2tags.RuFlexer()
        self.flexer.load()
        self.morph = pymorphy2.MorphAnalyzer()

    def find_forms_by_tags(self, lemma, req_tags):
        forms2 = list(self.flexer.find_forms_by_tags(lemma, req_tags))
        if len(forms2) == 0:
            if ('КРАТКИЙ', '0') in req_tags and ('ПАДЕЖ', 'ИМ') in req_tags:
                # склоняем прилагательное
                pz = self.morph.parse(lemma)[0]

                req_tags2 = set()
                if ('ЧИСЛО', 'ЕД') in req_tags:
                    req_tags2.add('sing')
                else:
                    req_tags2.add('plur')

                if ('РОД', 'МУЖ') in req_tags:
                    req_tags2.add('masc')
                elif ('РОД', 'ЖЕН') in req_tags:
                    req_tags2.add('femn')
                elif ('РОД', 'СР') in req_tags:
                    req_tags2.add('neut')

                py = pz.inflect(req_tags2)
                if py:
                    forms2 = [py.word]

        return forms2


def Aa(s):
    return s[0].upper() + s[1:]


def restore_Aa(lemma, form):
    if lemma[0].upper() == lemma[0]:
        return form[0].upper() + form[1:]
    else:
        return form


def acc_sing(noun):
    form = decline_noun(noun, case='Acc', number='Sing')
    return restore_Aa(noun, form)


def gen_sing(noun):
    form = decline_noun(noun, case='Gen', number='Sing')
    return restore_Aa(noun, form)


def gen_plur(noun):
    form = decline_noun(noun, case='Gen', number='Plur')
    return restore_Aa(noun, form)


def dat_sing(noun):
    form = decline_noun(noun, case='Dat', number='Sing')
    return restore_Aa(noun, form)



def decline_noun(noun, case, number):
    tags = []
    if case == 'Nom':
        tags.append(('ПАДЕЖ', 'ВИН'))
    elif case == 'Gen':
        tags.append(('ПАДЕЖ', 'РОД'))
    elif case == 'Ins':
        tags.append(('ПАДЕЖ', 'ТВОР'))
    elif case == 'Acc':
        tags.append(('ПАДЕЖ', 'ВИН'))
    elif case == 'Dat':
        tags.append(('ПАДЕЖ', 'ДАТ'))
    elif case == 'Prep':
        tags.append(('ПАДЕЖ', 'ПРЕДЛ'))
    else:
        raise NotImplementedError()

    if number == 'Sing':
        tags.append(('ЧИСЛО', 'ЕД'))
    elif number == 'Plur':
        tags.append(('ЧИСЛО', 'МН'))
    else:
        raise NotImplementedError()

    fx = list(inflector.find_forms_by_tags(noun.lower(), tags))
    if len(fx) == 0:
        print('Не могу просклонять существительное "{}"'.format(noun))
        exit(0)

    return restore_Aa(noun, fx[0])


def decline_adj(adj, case, number, gender, animacy):
    tags = [('КРАТКИЙ', '0'), ('СТЕПЕНЬ', 'АТРИБ')]
    if case == 'Nom':
        tags.append(('ПАДЕЖ', 'ВИН'))
    elif case == 'Gen':
        tags.append(('ПАДЕЖ', 'РОД'))
    elif case == 'Ins':
        tags.append(('ПАДЕЖ', 'ТВОР'))
    elif case == 'Acc':
        tags.append(('ПАДЕЖ', 'ВИН'))
    elif case == 'Dat':
        tags.append(('ПАДЕЖ', 'ДАТ'))
    elif case == 'Prep':
        tags.append(('ПАДЕЖ', 'ПРЕДЛ'))
    else:
        raise NotImplementedError()

    if number == 'Sing':
        tags.append(('ЧИСЛО', 'ЕД'))
    elif number == 'Plur':
        tags.append(('ЧИСЛО', 'МН'))
    else:
        raise NotImplementedError()

    if number == 'Sing':
        if gender == 'Fem':
            tags.append(('РОД', 'ЖЕН'))
        elif gender == 'Masc':
            tags.append(('РОД', 'МУЖ'))
        elif gender == 'Neut':
            tags.append(('РОД', 'СР'))
        else:
            raise NotImplementedError()

        if case == 'Acc' and gender == 'Masc':
            if animacy == 'Inan':
                tags.append(('ОДУШ', 'НЕОДУШ'))
            else:
                tags.append(('ОДУШ', 'ОДУШ'))
    else:
        if case == 'Acc':
            if animacy == 'Inan':
                tags.append(('ОДУШ', 'НЕОДУШ'))
            else:
                tags.append(('ОДУШ', 'ОДУШ'))

    uadj = adj.lower()
    fx = list(inflector.find_forms_by_tags(uadj, tags))
    if len(fx) == 0 and 'ё' in uadj:
        fx = list(inflector.find_forms_by_tags(uadj.replace('ё', 'е'), tags))

    if len(fx) == 0:
        print('Не могу просклонять прилагательное "{}"'.format(adj))
        exit(0)

    return restore_Aa(adj, fx[0])


def decline_collocation(colloc_str, case, number, gender=None, animacy=None):
    if isinstance(number, str):
        number_adj = number
        number_noun = number
    elif isinstance(number, tuple):
        number_adj = number[0]
        number_noun = number[1]
    else:
        raise NotImplementedError()

    if ' ' not in colloc_str:
        # Склоняем единственное слово.
        word = colloc_str
        tagsets = inflector.morph.parse(word)
        if len(tagsets) > 0:
            tagset = tagsets[0].tag._grammemes_tuple
            if 'NOUN' in tagset:
                return decline_noun(word, case, number_noun)
            elif 'ADJF' in tagset:
                return decline_adj(word, case=case, number=number_adj, gender=gender, animacy=animacy)
        return word
    else:
        # 2+ слов.
        # все прилагательные слева - склоняем
        # первое существительное слева - склоняем
        noun = None
        noun_animacy = None
        noun_gender = None

        for word in colloc_str.split(' '):
            tagsets = inflector.morph.parse(word)
            if len(tagsets) > 0:
                tagset = tagsets[0].tag._grammemes_tuple
                if 'NOUN' in tagset:
                    if 'anim' in tagset:
                        noun_animacy = 'Anim'
                    else:
                        noun_animacy = 'Inan'

                    if 'femn' in tagset:
                        noun_gender = 'Fem'
                    elif 'masc' in tagset:
                        noun_gender = 'Masc'
                    else:
                        noun_gender = 'Neut'

                    break

        forms = []
        for word in colloc_str.split(' '):
            tagsets = inflector.morph.parse(word)
            pos = None
            lemma = word
            if len(tagsets) > 0:
                tagset = tagsets[0].tag._grammemes_tuple
                lemma = tagsets[0].normal_form
                if 'ADJF' in tagset:
                    pos = 'ADJ'
                    if lemma == 'больший':
                        lemma = 'большой'
                elif 'NOUN' in tagset:
                    pos = 'NOUN'

            if pos == 'ADJ':
                if noun is None:
                    forms.append(decline_adj(lemma, case, number_adj, gender=noun_gender, animacy=noun_animacy))
                else:
                    forms.append(word)
            elif pos == 'NOUN':
                if noun is None:
                    forms.append(decline_noun(lemma, case, number_noun))
                    noun = word
                else:
                    forms.append(word)
            else:
                forms.append(word)

        res_str = ' '.join(forms)
        return res_str

    raise NotImplementedError()


def numcor(num0, colloc_text, case):
    if num0 > 20:
        num = num0 % 10
    else:
        num = num0 % 20

    if case == 'Nom':
        if num == 1:
            # 1 зеленый человечек
            # 1 красная ягода
            # 1 синее море
            return colloc_text
        elif num in (2, 3, 4):
            # 2 зеленых человечка
            # 2 красных редиски
            # 3 ярких солнышка
            return decline_collocation(colloc_text, case='Gen', number=('Plur', 'Sing'))
        else:
            # 5 желтых жетонов
            # 6 старых дедушек
            # 7 юных девушек
            return decline_collocation(colloc_text, case='Gen', number='Plur')
    elif case == 'Gen':
        if num == 1:
            # из 1 красной ягоды
            return decline_collocation(colloc_text, case='Gen', number='Sing')
        else:
            # из 2 красных ягод
            return decline_collocation(colloc_text, case='Gen', number='Plur')
    elif case == 'Dat':
        if num == 1:
            # к 1 синему сантехнику
            return decline_collocation(colloc_text, case='Dat', number='Sing')
        else:
            # к 3 синим сантехникам
            return decline_collocation(colloc_text, case='Dat', number='Plur')
    elif case == 'Ins':
        if num == 1:
            # перед 1 синим морем
            return decline_collocation(colloc_text, case='Ins', number='Sing')
        else:
            # перед 2 синими морями
            return decline_collocation(colloc_text, case='Ins', number='Plur')
    elif case == 'Prep':
        if num == 1:
            # на 1 маленькой тележке
            return decline_collocation(colloc_text, case='Prep', number='Sing')
        else:
            # на 2 маленькое тележке
            return decline_collocation(colloc_text, case='Prep', number='Plur')
    elif case == 'Acc':
        if num == 1:
            # на 1 маленькую тележку
            # на 1 старенького попугая
            # на 1 золотое колечко
            return decline_collocation(colloc_text, case='Acc', number='Sing')
        elif num in (2, 3, 4):
            # на 2 маленькие тележки
            # на 2 маленьких гномиков
            # на 2 тяжелых камня
            # на 2 железных кольца

            # ищем первое слева существительное
            noun_animacy = None
            noun_gender = None
            for word in colloc_text.split(' '):
                tagsets = inflector.morph.parse(word)
                if len(tagsets) > 0:
                    tagset = tagsets[0].tag._grammemes_tuple
                    if 'NOUN' in tagset:
                        if 'anim' in tagset:
                            noun_animacy = 'Anim'
                        else:
                            noun_animacy = 'Inan'

                        if 'femn' in tagset:
                            noun_gender = 'Fem'
                        elif 'masc' in tagset:
                            noun_gender = 'Masc'
                        else:
                            noun_gender = 'Neut'

                        break

            if noun_animacy == 'Inan': # and noun_gender in ('Masc', 'Fem'):
                # на 2 влажных камня
                return decline_collocation(colloc_text, case='Gen', number=('Plur', 'Sing'))
            else:
                # на 2 важных джентльменов
                return decline_collocation(colloc_text, case='Gen', number='Plur')
        else:
            # на 5 маленьких тележкек
            # на 6 маленьких гномиков
            # на 7 тяжелых камней
            # на 8 железных колец
            return decline_collocation(colloc_text, case='Gen', number='Plur')
    else:
        raise NotImplementedError()


def generate_samples_from_template(template_filepath, named_patterns):
    generated_samples = []
    all_texts = set()
    with open(template_filepath, 'r') as f:
        template = json.load(f)
        for _ in range(args.num_generations):
            vars = dict()
            for var_name, src in template['variables'].items():
                if var_name.startswith('#'):
                    # Суррогатные комментарии пропускаем
                    continue

                if isinstance(src, str):
                    var_str = TemplatePattern(src, named_patterns).run()

                    # подставим значения ранее заданных переменных в строку var_str
                    var_str = subst_vars(var_str, vars)

                    try:
                        var_str = eval(var_str)
                    except Exception as ex:
                        logging.error('Возникла ошибка при вычислении выражения %s в строке объявления переменной %s шаблона "%s"', var_str, var_name, fp)
                        logging.error(ex)
                        raise ValueError

                    if var_name in vars:
                        logging.error('Variable "%s" already defined in template "%s"', var_name, fp)
                        exit(1)
                    else:
                        vars[var_name] = var_str
                else:
                    logging.error('Could not interpret variable "%s" in template "%s"', var_name, fp)
                    raise NotImplementedError()

            # Проверим, что сгенерированные значения переменных удовлетворяют ограничениям.
            constraints_ok = True
            for constraint0 in template.get('constraints', []):
                # Подставим в эту строку значения переменных
                constraint = subst_vars(constraint0, vars)
                v = eval(constraint)
                if v is False:
                    constraints_ok = False
                    logging.debug('Нарушено ограничение "%s". После подстановки значений переменных имеем "%s"',
                                  constraint0, constraint)
                    break

            if constraints_ok is False:
                continue

            # Идем по элементам диалога и генерируем реплики.
            utterances = []
            iline = 0
            while iline < len(template['dialogue']):
                line = template['dialogue'][iline]
                if isinstance(line, str):
                    if line.startswith('!'):
                        oper = line[1:].strip()
                        if oper.startswith('goto '):
                            label = oper[5:].strip()
                            if label == 'EXIT':
                                # Переход на специальную метку EXIT означает прекращение генерации диалога
                                break
                            else:
                                # Переход на заданную метку. Ищем эту метку среди строк диалога, устанавливаем iline.
                                label_found = False
                                for i, line in enumerate(template['dialogue']):
                                    if isinstance(line, str) and line.startswith('!:'+label):
                                        iline = i+1  # следующая строка после метки
                                        label_found = True
                                        break
                                if not label_found:
                                    logging.error('Не найдена метка "%s" для перехода в операторе %s шаблона "%s"', label, line, fp)
                                    raise ValueError()
                                continue
                        elif oper.startswith('if '):
                            # Выделяем подстроки с проверяемым выражением и меткой перехода по шаблону:
                            # !if x1==0 goto OnZero
                            m = re.search(r'\bif\s+(.+)\s+goto\s+([a-zA-Z0-9_]+)\b', oper)
                            if m is None:
                                logging.error('Невалидный формат оператора if в строке %s шаблона "%s"', line, fp)
                                raise ValueError()
                            expr = m.group(1)
                            label = m.group(2)
                            try:
                                expr_val = subst_vars(expr, vars)
                                if eval(expr_val) is True:
                                    # Переходим на заданную метку label
                                    label_found = False
                                    for i, line in enumerate(template['dialogue']):
                                        if isinstance(line, str) and line.startswith('!:' + label):
                                            iline = i
                                            label_found = True
                                            break

                                    if not label_found:
                                        logging.error('Не найдена метка "%s" для перехода в операторе %s шаблона "%s"', label, line, fp)
                                        raise ValueError()
                                else:
                                    iline += 1

                                continue
                            except Exception as ex:
                                logging.error('Возникла ошибка при вычислении выражения %s в условии %s шаблона "%s"', expr, line, fp)
                                logging.error(ex)
                                raise ValueError()
                        elif oper.startswith(':'):
                            # Строка с объявлением метки. Пропускаем строку.
                            iline += 1
                            continue
                        else:
                            logging.error('Неизвестный оператор в строке %s шаблона "%s"', line, fp)
                            raise ValueError

                    line1 = line
                else:
                    # Задан список вариантов шаблона реплики.
                    line1 = random.choice(line)

                pattern = TemplatePattern(line1, named_patterns)
                utterance = pattern.run()
                # подстановка переменных и раскрытие выражений ...
                for expr0 in re.findall(r'\{(.+?)\}', utterance):
                    expr = expr0
                    expr = subst_vars(expr, vars)
                    try:
                        expr2 = str(eval(expr))
                    except Exception as ex:
                        logging.error('Возникла ошибка при вычислении выражения %s в условии %s шаблона "%s"', expr, line, fp)
                        logging.error(ex)
                        raise ValueError()
                    utterance = utterance.replace('{' + expr0 + '}', expr2)
                utterances.append(utterance)
                iline += 1

            text = '\n'.join(utterances)
            if text not in all_texts:
                all_texts.add(text)
                sample = {'src': fp, 'dialogue': utterances}
                generated_samples.append(sample)

    return generated_samples


def subst_vars(text0, vars):
    text = text0

    # подставим значения ранее заданных переменных в строку var_str
    for var_name, var_value in vars.items():
        if isinstance(var_value, str):
            text = re.sub(r'\b' + var_name + r'\b', '"' + var_value + '"', text)
        elif isinstance(var_value, int):
            text = re.sub(r'\b' + var_name + r'\b', str(var_value), text)
        elif isinstance(var_value, list):
            text = re.sub(r'\b' + var_name + r'\b', str(var_value), text)
        else:
            raise NotImplementedError()

    # В строке после подстановок не должно оставаться никаких паттернов переменных {...}
    if '{' in text:
        logging.error('В строке "%s" остались нераскрытые слоты. Исходная строка "%s"', text, text0)
        exit(1)

    return text


def run_tests():
    assert numcor(2, "яблоко", 'Acc') == "яблока"
    assert numcor(4, "яблоко", 'Acc') == "яблока"

    assert(decline_collocation('кошка', case='Gen', number='Plur') == 'кошек')
    assert(decline_collocation('сердечко', case='Gen', number='Plur') == 'сердечек')
    assert(decline_collocation('кот', case='Dat', number='Sing') == 'коту')
    assert(decline_collocation('синяя тележка', case='Acc', number='Plur') == 'синие тележки')
    assert(decline_collocation('синий кот', case='Acc', number='Plur') == 'синих котов')
    assert(decline_collocation('синяя кошка', case='Acc', number='Plur') == 'синих кошек')
    assert(decline_collocation('синяя ромашка', case='Acc', number='Plur') == 'синие ромашки')

    assert(numcor(1, "зеленый человечек", case='Nom') == "зеленый человечек")
    assert(numcor(1, "крупная ягода", case='Nom') == "крупная ягода")
    assert(numcor(1, "синее море", case='Nom') == "синее море")
    assert(numcor(2, "красная редиска", case='Nom') == "красных редиски")
    assert(numcor(3, "яркое солнышко", case='Nom') == "ярких солнышка")
    assert(numcor(5, "желтый жетон", case='Nom') == "желтых жетонов")
    assert(numcor(6, "старый дедушка", case='Nom') == "старых дедушек")
    assert(numcor(7, "юная девушка", case='Nom') == "юных девушек")

    assert(numcor(1, "красная ягода", case='Gen') == "красной ягоды")
    assert(numcor(2, "красная ягода", case='Gen') == "красных ягод")
    assert(numcor(5, "красная ягода", case='Gen') == "красных ягод")

    assert(numcor(1, "синий сантехник", case='Dat') == "синему сантехнику")
    assert(numcor(2, "синий сантехник", case='Dat') == "синим сантехникам")

    assert(numcor(1, "обширное пастбище", case='Ins') == "обширным пастбищем")
    assert(numcor(2, "синее море", case='Ins') == "синими морями")

    assert(numcor(1, "маленькая тележка", 'Prep') == 'маленькой тележке')
    assert(numcor(2, "маленькая тележка", 'Prep') == 'маленьких тележках')
    assert(numcor(5, "маленькая тележка", 'Prep') == 'маленьких тележках')

    assert(numcor(1, "маленькая тележка", 'Acc') == 'маленькую тележку')
    assert(numcor(1, "старенького попугая", 'Acc') == "старенького попугая")
    assert(numcor(1, "тяжелый камень", 'Acc') == "тяжелый камень")
    assert(numcor(1, "золотое колечко", 'Acc') == "золотое колечко")

    assert numcor(5, 'железная тележка', case='Acc') == 'железных тележек'
    assert numcor(6, 'маленький гномик', case='Acc') == 'маленьких гномиков'
    assert numcor(7, 'тяжелый камень', case='Acc') == 'тяжелых камней'
    assert numcor(8, 'железное кольцо', case='Acc') == 'железных колец'

    assert(numcor(2, "круглый малыш", 'Acc') == 'круглых малышей')
    assert(numcor(2, "влажный камень", 'Acc') == 'влажных камня')
    assert(numcor(2, "важный джентльмен", 'Acc') == 'важных джентльменов')

    assert(numcor(2, "маленькая тележка", 'Acc') == 'маленьких тележки')
    assert(numcor(5, "маленькая тележка", 'Acc') == 'маленьких тележек')


if __name__ == '__main__':
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description='Генерация числовых задач в формате диалогов по шаблонам')
    parser.add_argument('--dir', type=str, default=os.path.join(proj_dir, 'data'), help='Путь к каталогу с шаблонами и файлами ресурсов')
    parser.add_argument('--resource_dir', type=str, default=os.path.join(proj_dir, 'data'), help='Путь каталогу, где находятся файлы ресурсов')
    parser.add_argument('--template', type=str, help='Путь к единственному шаблону, который надо проверить')
    parser.add_argument('--output_dir', type=str, default=os.path.join(proj_dir, 'output'), help='Директория, куда скидывать результаты генерации')
    parser.add_argument('--num_generations', type=int, default=10, help='Количество сэмплов, генерируемых из одного шаблона')

    args = parser.parse_args()

    # Настроим логирование в файл
    log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(asctime)-15s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger('')
    logger.setLevel(log_level)

    logfile_path = os.path.join(args.output_dir, 'math_tasks.log')
    lf = logging.FileHandler(logfile_path, mode='w')
    lf.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)

    #gren = ruword2tags.RuWord2Tags()
    #gren.load()
    inflector = CombinedInflector()

    morph = pymorphy2.MorphAnalyzer()

    # НАЧАЛО ОТЛАДКИ
    if True:
        run_tests()
    # КОНЕЦ ОТЛАДКИ

    named_patterns = dict()  # TODO

    # Загружаем ресурсы в глобальгное пространство переменных
    for fp in glob.glob(os.path.join(args.resource_dir, '**', 'resource_*.json'), recursive=True):
        logging.info('Загружаем глобальные переменные из файла "%s"', fp)
        with open(fp, 'r') as f:
            data = json.load(f)
            for var_name, var_value in data.items():
                decl = '{} = {}'.format(var_name, var_value)
                exec(decl)
    logging.info('Загрузка ресурсов завершена')

    if args.template:
        # обрабатываем один заданный шаблон
        fpx = [args.template]
    else:
        # обрабатываем все шаблоны в заданном каталоге и рекурсивно в его подкаталогах
        fpx = list(glob.glob(os.path.join(args.dir, '**', 'template_*.json'), recursive=True))

    generated_samples = []
    for fp in fpx:
        logging.info('Start processing template file "%s"', fp)
        samples = generate_samples_from_template(fp, named_patterns)
        generated_samples.extend(samples)
        for i, sample in enumerate(samples, start=1):
            logging.info('Sample #%d generated from "%s"', i, fp)
            for j, msg in enumerate(sample['dialogue'], start=1):
                logging.info('[%d] %s', j, msg)
            logging.info('-'*100)

    # Сохраним итоговый список сгенерированных сэмплов
    res_path = os.path.join(args.output_dir, 'math_tasks.json')
    logging.info('Writing %d generated samples to "%s"', len(generated_samples), res_path)
    with open(res_path, 'w') as f:
        json.dump(generated_samples, fp=f, indent=4, ensure_ascii=False)
    logging.info('All done :)')
