import numpy as np 
import pandas as pd
import jieba
from jieba import posseg
from utils.tokenizer import segment


Remove_Words = ['|', '[', ']', '语音', '图片']


def read_stopwords(path):
    '''读取停用词'''
    lines = set()
    with open ('path', 'r', encoding='utf-8') as f:    
        for line in f.readlines():
            line = line.strip()
            lines.add(line)	
    return lines


def remove_words(words_list):
    '''读取无用词'''
    words_list = [word for word in words_list if word not in Remove_Words]
    return words_list

def parse_data(train_path, test_path):
    '''去除填充空值'''
    train_df = pd.read_csv(train_path, 'r', encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = train_df.Report
    
    test_df = pd.read_csv(test_path, 'r', encoding='utf-8')
    test_df.dropna(subset=['Report'], how='any', inplace=True)
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y

def save_data(data_1, data_2, data_3, data_path_1, data_path_2, data_path_3, stop_words_path=''):
    '''清洗、分词保存数据'''
    stopwords = read_stopwords(stop_words_path)
    with open (data_path_1, 'w', encoding='utf-8') as f1:
        count_1 = 0
        for line in data_1:
            seg_list = segment(line.strip(), cut_type='word')
            seg_list = remove_words(seg_list)
            if len(seg_list) > 0:
                seg_line = ' '.join(seg_list)
                f1.write('%s' % seg_line)
                f1.write('\n')
                count_1 += 1
        print('train_x length is {}',count_1)
        
    with open (data_path_2, 'w', encoding='utf-8') as f2:
        count_2 = 0
        for line in data_2:
            seg_list = segment(line.strip(), cut_type='word')
            seg_list = remove_words(seg_list)
            if len(seg_list) > 0:
                seg_line = ' '.join(seg_list)
                f2.write('%s' % seg_line)
                f2.write('\n')
                count_2 += 1
        print('train_y length is {}',count_2)
        
    with open (data_path_3, 'w', encoding='utf-8') as f3:
        count_3 = 0
        for line in data_3:
            seg_list = segment(line.strip(), cut_type='word')
            seg_list = remove_words(seg_list)
            if len(seg_list) > 0:
                seg_line = ' '.join(seg_list)
                f3.write('%s' % seg_line)
                f3.write('\n')
                count_3 += 1
        print('test_x length is {}',count_3)


def preprocess_sentence(sentence):
    '''句子分词'''
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    train_list_src, train_list_trg, test_list_src, _ = parse_data('./datasets/AutoMaster_TrainSet.csv',
                                                                  './datasets/AutoMaster_TestSet.csv')
    save_data(train_list_src,
              train_list_trg,
              test_list_src,
              './datasets/train_set.seg_x.txt',
              './datasets/train_set.seg_y.txt',
              './datasets/test_set.seg_x.txt',
              stop_words_path='./datasets/stop_words.txt')