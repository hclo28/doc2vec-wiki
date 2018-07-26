import gensim
from pprint import pprint
model = gensim.models.Doc2Vec.load('wikipedia.model')

p('-----単語の類似度計算-----')
exec_most_sim(['アマゾン ウェブ サービス'])
exec_most_sim(['Mac'])
exec_most_sim(['イチロー'])

p('-----単語の演算-----')
exec_calc(['王様'], ['女'], ['男'])

def exec_most_sim(doc)
    p(doc)
    pprint(model.docvecs.most_similar(doc))
    p('---------------')

def exec_calc(doc, plus, minus)
    p(doc + '-' minus + '+' plus)
    vec = [model.docvecs[doc] - model.docvecs[minus] + model.docvecs[plus]]
    pprint(model.docvecs.most_similar(vec))
    p('---------------')
