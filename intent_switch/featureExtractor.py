import sys

sys.path.append('../')
import os

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home"
os.environ['ANSERINI_CLASSPATH'] = "/Volumes/ext3/arvin/anserini/target/"

from pyserini.index import pygenerator, pyutils
from pyserini.search import pysearch
from jnius import autoclass
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer
import numpy as np
from parser import parse_qrel, parse_topics
from tqdm import tqdm

JString = autoclass('java.lang.String')


class FeatureExtractor:
    """
    LTR feature extractor, base on anserini and pyserini.
    """

    def __init__(self, index_path: str, analyzer):
        """
        :param index_path: path to anserini index.
        :param analyzer: lucene analyzer, suppose to be the same as used in the indexing.
        """
        self.index_utils = pyutils.IndexReaderUtils(index_path)
        self.reader = self.index_utils.reader
        self.analyzer = analyzer
        self.feature_description = []

    def add_feature_description(self, description: str):
        self.feature_description.append(description)

    def _analyze_query(self, query: str) -> list:
        """
        stem and tokenize query string by using self lucene analyzer.
        :param query: query string
        :return: list of tokens.
        """
        return self.index_utils.analyze(query, analyzer=self.analyzer)

    def get_field_length(self, docid: str, field: str) -> int:
        """
        Extract field length by given docid.
        :param docid: docid, unique document id in index.
        :param field: document field want to extract.
        :return: field length
        """
        try:
            field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        except:
            return 0
        num_term = 0
        for v in field_vector.values():
            num_term += v
        return num_term

    def get_query_length(self, query: str) -> int:
        """
        Get query length, maybe not useful for LTR as the value won't change for a same topic.
        :param query: raw query string
        :return: query length
        """
        return len(self._analyze_query(query))

    def get_tf(self, query: str, docid: str, field: str) -> int:
        query_tokens = self._analyze_query(query)
        try:
            field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        except:
            return 0
        tf = 0
        for token in query_tokens:
            if token in field_vector.keys():
                tf += field_vector[token]
        return tf

    def get_idf(self, query: str, field: str) -> float:
        """
        Get sum idf of a given query, maybe not useful for LTR as the value won't change for a same topic.
        :param query: raw query string
        :param field: field want to compute.
        :return:
        """
        N = self.reader.getDocCount(JString(field))  # total num of docs.
        query_tokens = self._analyze_query(query)
        idf = 0
        for token in query_tokens:
            n = self.index_utils.object.getDocFreq(self.reader, token, field)
            idf += np.log(N / (1 + n))
        return idf

    def get_tfidf(self, query: str, docid: str, field: str) -> float:
        query_tokens = self._analyze_query(query)
        try:
            field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        except:
            return 0
        N = self.reader.getDocCount(JString(field))  # total num of docs.
        tfidf = 0
        for token in query_tokens:
            if token in field_vector.keys():
                tf = field_vector[token]
                n = self.index_utils.object.getDocFreq(self.reader, token, field)
                idf = np.log(N / (1 + n))
                tfidf += tf * idf
        return tfidf

    def get_query_cover_num(self, query: str, docid: str, field: str) -> float:
        tokens = self._analyze_query(query)
        try:
            field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        except:
            return 0
        cover_num = 0
        for token in tokens:
            if token in field_vector.keys():
                cover_num += 1
        return cover_num

    def get_query_cover_ratio(self, query: str, docid: str, field: str) -> float:
        tokens = self._analyze_query(query)
        try:
            field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        except:
            return 0
        cover_num = 0
        for token in tokens:
            if token in field_vector.keys():
                cover_num += 1
        return cover_num/len(tokens)

    def get_fieldFeature(self, query: str, docid: str, field: str, method: str, sdm=False, slop=1, inOder=False) -> float:
        """
        :param query: raw query string.
        :param docid: unique document id.
        :param field: document field.
        :param method: feature model, options: BM25, Dirichlet, JelinekMercer.
        :param sdm: using Sequential Dependence Model?
        :param slop: parameter of sdm: https://lucene.apache.org/core/7_3_0/core/org/apache/lucene/search/spans/SpanNearQuery.html
        :param inOder: parameter of sdm: https://lucene.apache.org/core/7_3_0/core/org/apache/lucene/search/spans/SpanNearQuery.html
        :return: feature value.
        """
        return self.index_utils.object.getFieldFeature(self.reader,
                                                       JString(docid),
                                                       JString(field),
                                                       JString(query.encode('utf-8')),
                                                       self.analyzer,JString(method),
                                                       sdm, slop, inOder)
if __name__ == "__main__":
    topic_dic = parse_topics("topics")
    qrel_dic = parse_qrel("0.txt")

    index_path = '/Volumes/ext3/arvin/anserini/lucene-index.cw09b/'
    analyzer = get_lucene_analyzer()
    extractor = FeatureExtractor(index_path, analyzer)
    fields = ["title", "url", "anchor", "contents"]
    f = open("clueweb09_intent_change.txt", "w+")

    for qid in tqdm(qrel_dic.keys()):
        for docid in qrel_dic[qid].keys():
            rel = qrel_dic[qid][docid]
            query = topic_dic[qid]
            features = []
            s = ""
            for field in fields:
                features.append(extractor.get_field_length(docid, field))
                features.append(extractor.get_tf(query, docid, field))
                features.append(extractor.get_tfidf(query, docid, field))
                features.append(extractor.get_query_cover_num(query, docid, field))
                features.append(extractor.get_query_cover_ratio(query, docid, field))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25"))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet"))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer"))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=1, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=2, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=8, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=1, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=2, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "BM25", sdm=True, slop=8, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=1, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=2, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=8, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=1, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=2, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "Dirichlet", sdm=True, slop=8, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=1, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=2, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=8, inOder=True))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=1, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=2, inOder=False))
                features.append(extractor.get_fieldFeature(query, docid, field, "JelinekMercer", sdm=True, slop=8, inOder=False))

            s += "{} qid:{} ".format(rel, qid)
            for i in range(len(features)):
                s += "{}:{:.6f} ".format(i+1, features[i])
            s += "#{}".format(docid)
            f.write(s + "\n")

    f.close()
