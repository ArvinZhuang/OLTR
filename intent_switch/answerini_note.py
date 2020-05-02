import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home"
os.environ['ANSERINI_CLASSPATH'] = "/Users/s4416495/experiment_code/anserini/target/"

from pyserini.collection import pycollection
from pyserini.index import pygenerator, pyutils
from pyserini.search import pysearch
from jnius import autoclass
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer



index_path = '/Users/s4416495/experiment_code/anserini/lucene-index.cw09b/'


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
doc = searcher.doc_by_field("id", "clueweb09-enwp00-00-00000")
print(doc.lucene_document().get("id"))

# this is how you search query by field, using lucene query syntax.
# hits = searcher.object.searchLuceneSyntax('test parser', 10)
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
doc_vector = index_utils.get_document_vector_by_field('clueweb09-enwp00-00-00000', "title")
print(doc_vector)

# this is how you compute field length for a document.
fieldLength = 0
for v in doc_vector.values():
    fieldLength += v
print(fieldLength)

# this is how you get document freq and collection doc freq by given term and field.
docFreq,  collectionFreq = index_utils.get_term_counts_by_field('china', "contents")
print(docFreq, collectionFreq)


# this is how you can get lucene index reader.
reader = JIndexReaderUtils.getReader(JString(index_path))
# this is how you get total number of terms in a given field.
print(reader.getSumTotalTermFreq(JString("url")))
# this is how you get number of documents contains the field
print(reader.getDocCount(JString("url")))
# this is how you compute average field length
print(reader.getSumTotalTermFreq(JString("url"))/reader.getDocCount(JString("url")))
# Note: if you want to get information by term string, remember what is the analyzer used for indexing.

