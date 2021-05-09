import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home"
os.environ['ANSERINI_CLASSPATH'] = "/Volumes/ext3/arvin/anserini/target/"

from pyserini.collection import pycollection
from pyserini.index import pygenerator, pyutils
from pyserini.search import pysearch
from jnius import autoclass
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer



index_path = '/Volumes/ext3/arvin/anserini/lucene-index.cw09b/'


JString = autoclass('java.lang.String')
JIndexReaderUtils = autoclass('io.anserini.index.IndexReaderUtils')

# this is how to get lucene document
# reader = JIndexReaderUtils.getReader(JString(index_path))
# doc = JIndexReaderUtils.document(reader, JString("clueweb09-enwp00-00-00065"))
# # get stored field text
# print(doc.get("TEST"))
#
# # this is another way to get lucene document
searcher = pysearch.SimpleSearcher(index_path)
doc = searcher.doc_by_field("id", "clueweb09-en0003-90-15493")
print("lucene document id", doc.lucene_document().get("id"))

# this is how you search query by field, using lucene query syntax.
# hits = searcher.object.searchLuceneSyntax('id|test parser', 10)
# for i in range(0, 2):
#     print([field.name() for field in hits[i].lucene_document.getFields()])



# # this is how to iterate documents in a collection.
# collection = pycollection.Collection('ClueWeb09Collection', '/Users/s4416495/experiment_code/anserini/ClueWeb09b/')
# generator = pygenerator.Generator('DefaultLuceneDocumentGenerator')
# for (i, fs) in enumerate(collection):
#     for (j, doc) in enumerate(fs):
#         print(doc.id)

# this is how you get tf dictionary by docid and field.
index_utils = pyutils.IndexReaderUtils(index_path)
doc_vector = index_utils.get_document_vector_by_field('clueweb09-en0003-90-15493', "title")
print("tf given field:", doc_vector)

# this is how you compute field length for a document.
fieldLength = 0
for v in doc_vector.values():
    fieldLength += v
print("field length:", fieldLength)

# this is how you get document freq and collection doc freq by given term and field.
docFreq,  collectionFreq = index_utils.get_term_counts_by_field('china', "contents")
print("term:", 'china')
print("document freq:", docFreq, "collection freq", collectionFreq)


# this is how you can get lucene index reader.
reader = JIndexReaderUtils.getReader(JString(index_path))
# this is how you get total number of terms in a given field.
print("total number of terms:", reader.getSumTotalTermFreq(JString("url")))
# this is how you get number of documents contains the field
print("total number of documents:", reader.getDocCount(JString("url")))
# this is how you compute average field length
print("field length:", reader.getSumTotalTermFreq(JString("url"))/reader.getDocCount(JString("url")))
# Note: if you want to get information by term string, remember what is the analyzer used for indexing.


# this is how you tokenize and stemming a query string.
query = "what are the best cities in China, maybe is the yueyang and shanghai feed sworn villag maybe"
analyzer = get_lucene_analyzer()
analyzed_query = index_utils.analyze(query, analyzer=analyzer)
print(query)
print("analyzed query:", analyzed_query)





# get bm25 score of a document given a query string and field.
def compute_field_bm25_term_weight(reader, docid: str, field: str, term: str) -> float:
    object = JIndexReaderUtils()
    return object.getFieldBM25TermWeight(reader, JString(docid), JString(field), JString(term.encode('utf-8')))\

def compute_field_bm25_query_weight(reader, docid: str, field: str, query: str, analyzer) -> float:
    object = JIndexReaderUtils()
    return object.getFieldBM25QueryWeight(reader, JString(docid), JString(field), JString(query.encode('utf-8')), analyzer)


# this is how to get field bm25 score of a document given term by term.
docid ='clueweb09-en0025-18-33016'
bm25_score = 0
for term in analyzed_query:
    # bm25_score = index_utils.compute_bm25_term_weight(docid, term)
    bm25_score += compute_field_bm25_term_weight(reader, docid, "title", term)
print(bm25_score)

# this is how to get field bm25 score of a document given query string.
print(compute_field_bm25_query_weight(reader, docid, "title", query, analyzer))

