from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from utils.data_utils import dump_pkl


def read_lines(path, col_sep=None):
    lines = []
    with open (path, 'r', encoding='utf-8') as f:
        for line in f:
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(path1, path2, path3):
    lines = read_lines(path1)
    lines += read_lines(path2)
    lines += read_lines(path3)
    return lines


def save_sentence(lines, path):
    with open (path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n'%line.strip())
    print('save sentence:%s' % path)


def build(path1, path2, path3, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(path1, path2, path3)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')

    # train model
    w2v = Word2Vec(sg=1,
                 sentences = LineSentence(sentence_path),
                 size=256,
                 negative=5,
                 workers=8,
                 iter=40,
                 window=3,
                 min_count=min_count)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('宝马', '车主')
    print('宝马 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('../datasets/train_set.seg_x.txt',
          '../datasets/train_set.seg_y.txt',
          '../datasets/test_set.seg_x.txt',
          out_path='../datasets/word2vec.txt',
          sentence_path='../datasets/sentences.txt',
          )