
class qrelRecord:
    def __init__(self, topicId, subtopicId, docId, relevance):
        self._topicId = topicId
        self._subtopicId = subtopicId
        self._docId = docId
        self._relevance = relevance

    def to_string(self):
        return "{} {} {} {}".format(self._topicId, self._subtopicId, self._docId, self._relevance)

class qrelRecords:
    def __init__(self):
        self.qrel_list = []

    def add_record(self, qrel_record):
        self.qrel_list.append(qrel_record)

    def write_to_disk(self, path):
        f = open(path, "w")
        for record in self.qrel_list:
            f.write(record.to_string()+'\n')
        f.close()

def read_adhoc(path, data):
    with open(path, "r") as f:
        for line in f:
            cols = line.strip().split()
            topicId = cols[0]
            if year == 9:
                subtopicId = 0
                docId = cols[1]
                relevance = cols[2]
            else:
                subtopicId = int(cols[1])
                docId = cols[2]
                relevance = cols[3]

            if topicId not in data.keys():
                if int(relevance) > 0:
                    data[topicId] = {docId: [subtopicId]}
                else:
                    data[topicId] = {docId: []}
            else:
                if docId not in data[topicId].keys():
                    if int(relevance):
                        data[topicId][docId] = [subtopicId]
                    else:
                        data[topicId][docId] = []
                else:
                    if int(relevance):
                        data[topicId][docId].append(subtopicId)
    return data

def read_diversity(path, data):
    with open(path, "r") as f:
        for line in f:
            cols = line.strip().split()
            topicId = cols[0]
            subtopicId = int(cols[1])
            docId = cols[2]
            relevance = cols[3]

            if topicId not in data.keys():
                if int(relevance) > 0:
                    data[topicId] = {docId: [subtopicId]}
                else:
                    data[topicId] = {docId: []}
            else:
                if docId not in data[topicId].keys():
                    if int(relevance):
                        data[topicId][docId] = [subtopicId]
                    else:
                        data[topicId][docId] = []
                else:
                    if int(relevance):
                        data[topicId][docId].append(subtopicId)
    return data

if __name__ == "__main__":
    qrel_years = [9,10,11,12]
    num_subtopics = 4
    data = {}
    for year in qrel_years:
        adhoc_path = "./accessments/qrels.adhoc-w{:02d}".format(year)
        div_path = "./accessments/qrels.diversity-w{:02d}".format(year)
        data = read_adhoc(adhoc_path, data)
        data = read_diversity(div_path, data)

    qrelRecordsList = []
    for i in range(num_subtopics):
        qrelRecordsList.append(qrelRecords())

    for topicId in data.keys():
        for docId in data[topicId].keys():
            for subtopicId in range(num_subtopics):
                if subtopicId in data[topicId][docId]:
                    newRecord = qrelRecord(topicId, subtopicId, docId, 1)
                else:
                    newRecord = qrelRecord(topicId, subtopicId, docId, 0)
                qrelRecordsList[subtopicId].add_record(newRecord)

    for i in range(num_subtopics):
        qrelRecordsList[i].write_to_disk("{}.txt".format(i))











