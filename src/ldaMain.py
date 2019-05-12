# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corpus = []
corpus_name = []
length = 0

def cutting():
    rootdir = '..\\text'
    list = os.listdir(rootdir)
    length = len(list)
    print length
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            with open(path) as f:
                document = f.read()

                document_decode = document.decode('utf-8')
                document_cut = jieba.cut(document_decode)
                result = ' '.join(document_cut)
                result = ' '.join(result.split())
                result = result.encode('utf-8')

                with open('..\cut\\' + list[i], 'w') as f2:
                    f2.write(result)

                corpus.append(result)
                corpus_name.append(list[i])

            f.close()
            f2.close()

def print_top_words(model,feature_names,n_top_words):

    for topic_idx,topic in enumerate(model.components_):
        # with open('../result/topic_word', 'a+') as f:
        #     f.write("Topic #%d:" % topic_idx)
        #     f.write(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) + '\n')

        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    print
    print model.components_

def lda_func():
    stpwrdpath="..\stop_words.txt"
    stpwrd_dic = open(stpwrdpath,'rb')
    stpwrd_content = stpwrd_dic.read()

    # use stopword table convert into list
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()

    cntVector = CountVectorizer(stop_words=stpwrdlst)
    cntTf = cntVector.fit_transform(corpus)
    print cntTf
    print cntVector.get_feature_names()
    print cntVector.vocabulary_

    # print length/3
    lda = LatentDirichletAllocation(n_topics=10,learning_offset=50,learning_method='batch')

    docres = lda.fit_transform(cntTf)

    # print len(docres)
    # print len(docres[0])

    # 文档的主题分布
    print docres

    # 主题和词的分布
    # print lda.components_


    #
    n_top_words = 15
    tf_feature_names = cntVector.get_feature_names()

    print_top_words(lda, tf_feature_names, n_top_words)


def test():
    for i in range(len(corpus_name)):
        print corpus_name[i] + ':' + corpus[i]


    maxindex = 0


    # for i in range(len(docres)):
    #
    #     for j in range(len(docres[0])):
    #         if docres[i][j] > docres[i][maxindex]:
    #             maxindex = j
    #     with open('../result/doc_topic','a+') as f:
    #         f.write(corpus_name[i] + ":" + bytes(maxindex) + '\n')



if __name__ == '__main__':

    cutting()
    lda_func()