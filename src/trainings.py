# coding: UTF-8
import gensim
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, Doc2VecVocab, TaggedDocument

wiki = WikiCorpus('./data/jawiki-latest-pages-articles.xml.bz2')

class TaggedWikiDocument(object):
   def __init__(self, wiki):
       self.wiki = wiki
       self.wiki.metadata = True
   def __iter__(self):
       for content, (page_id, title) in self.wiki.get_texts():
           yield TaggedDocument([c for c in content], [title])

document = TaggedWikiDocument(wiki)
# dm：dmpv=1（デフォルト）
# size：分散表現の次元数
# window：コンテキストの文脈幅
# min_count：学習に使う単語の最低出現回数
# alpha: 学習率
# sample：頻出語を無視するしきい値
# epochs：一つの訓練データを何回繰り返して学習させるか
# seed：乱数のシード値
model = Doc2Vec(documents=document, dm=1, vector_size=400, window=8, min_count=10, epochs=10, workers=6)
model.save('model/wikipedia.model')
