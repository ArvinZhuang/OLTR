import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home"
os.environ['ANSERINI_CLASSPATH'] = "/Volumes/ext3/arvin/anserini/target/"

from pyserini.index import pygenerator, pyutils
from pyserini.search import pysearch
from jnius import autoclass
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer
import numpy as np

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
        field_vector = self.index_utils.get_document_vector_by_field(docid, field)
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
        field_vector = self.index_utils.get_document_vector_by_field(docid, field)
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
            idf += np.log(N/(1 + n))
        return idf

    def get_tfidf(self, query: str, docid: str, field: str) -> float:
        query_tokens = self._analyze_query(query)
        field_vector = self.index_utils.get_document_vector_by_field(docid, field)
        N = self.reader.getDocCount(JString(field))  # total num of docs.
        tfidf = 0
        for token in query_tokens:
            if token in field_vector.keys():
                tf = field_vector[token]
                n = self.index_utils.object.getDocFreq(self.reader, token, field)
                idf = np.log(N/(1 + n))
                tfidf += tf * idf
        return tfidf

    def get_bm25(self, query: str, docid: str, field: str) -> float:
        return self.index_utils.object.getFieldBM25QueryWeight(self.reader,
                                                               JString(docid),
                                                               JString(field),
                                                               JString(query.encode('utf-8')),
                                                               self.analyzer)


if __name__ == "__main__":
    query = "obama family tree"
    docid = 'clueweb09-en0025-18-33016'
    index_path = '/Volumes/ext3/arvin/anserini/lucene-index.cw09b/'
    analyzer = get_lucene_analyzer()
    field = "title"

    extractor = FeatureExtractor(index_path, analyzer)
    print(extractor.get_field_length(docid, field))
    print(extractor.get_tf(query, docid, field))
    print(extractor.get_tfidf(query, docid, field))
    print(extractor.get_bm25(query, docid, field))
