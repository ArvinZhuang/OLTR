import os


def parse_topics(input_folder: str) -> dict:
    files = os.listdir(input_folder)

    topic_dic = {}
    for file in files:
        # feature id in standard letor datasets start from 1.
        with open(os.path.join(input_folder, file), 'r') as f:
            for line in f:
                cols = line.split(":")
                if "-" in cols[0]:
                    cols[0] = cols[0].split("-")[1]
                topic_id = int(cols[0])
                query = cols[1].strip()
                topic_dic[topic_id] = query
    return topic_dic


def parse_qrel(path: str) -> dict:
    qrel_dic = {}
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            rel = int(rel)
            qid = int(qid)
            if qid in qrel_dic.keys():
                qrel_dic[qid][docid] = rel
            else:
                qrel_dic[qid] = {docid: rel}
    return qrel_dic


def get_relevant_docs_by_qid(qrel_dic: dict, qid: str) -> list:
    rel_docs = []
    for docid, rel in qrel_dic[qid].items():
        if rel > 0:
            rel_docs.append(docid)
    return rel_docs


def write_qrel(qrel_dic: dict, path: str, intent: int):
    s = ''
    for qid in qrel_dic.keys():
        for docid, rel in qrel_dic1[qid].items():
            s += '{} {} {} {}\n'.format(qid, intent, docid, rel)

    with open(path + "{}.txt".format(intent), "w") as f:
        f.write(s)


def make_exclusive():
    topics = ["1.txt", "2.txt", "3.txt", "4.txt"]
    for topic1 in range(len(topics)):
        qrel_dic1 = parse_qrel(topics[topic1])
        for topic2 in range(len(topics)):
            qrel_dic2 = parse_qrel(topics[topic2])
            num_overlap = 0

            for qid in qrel_dic1.keys():
                rel_docs1 = get_relevant_docs_by_qid(qrel_dic1, qid)
                rel_docs2 = get_relevant_docs_by_qid(qrel_dic2, qid)
                for docid in rel_docs1:

                    if docid in rel_docs2:
                        num_overlap += 1
        #                 if topic1 != topic2:
        #                     qrel_dic1[qid][docid] = 0
        # write_qrel(qrel_dic1, 'intents_exclusive/', topic1+1)


if __name__ == "__main__":

    # make_exclusive()
    intent_path = "intents"
    topics = ["{}/1.txt".format(intent_path),
              "{}/2.txt".format(intent_path),
              "{}/3.txt".format(intent_path),
              "{}/4.txt".format(intent_path)]

    # print overlap relevance docs fo each intents
    # for topic1 in range(len(topics)):
    #     qrel_dic1 = parse_qrel(topics[topic1])
    #     for topic2 in range(len(topics)):
    #         qrel_dic2 = parse_qrel(topics[topic2])
    #         num_overlap = 0
    #         for qid in qrel_dic1.keys():
    #             rel_docs1 = get_relevant_docs_by_qid(qrel_dic1, qid)
    #             rel_docs2 = get_relevant_docs_by_qid(qrel_dic2, qid)
    #             for docid in rel_docs1:
    #                 if docid in rel_docs2:
    #                     num_overlap += 1
    #
    #         print(topic1, topic2, num_overlap)

    # print exclusive relevance docs fo each intents
    for topic1 in range(len(topics)):
        qrel_dic1 = parse_qrel(topics[topic1])
        for topic2 in range(len(topics)):
            qrel_dic2 = parse_qrel(topics[topic2])
            num_overlap = 0

            for qid in qrel_dic1.keys():
                rel_docs1 = get_relevant_docs_by_qid(qrel_dic1, qid)
                rel_docs2 = get_relevant_docs_by_qid(qrel_dic2, qid)
                for docid in rel_docs1:
                    if docid in rel_docs2:
                        num_overlap += 1
                        if topic1 != topic2:
                            qrel_dic1[qid][docid] = 0
        exclusive = 0
        for qid in qrel_dic1.keys():
            rel_docs1 = get_relevant_docs_by_qid(qrel_dic1, qid)
            exclusive += len(rel_docs1)
        print(topic1, exclusive)



    # qrel_dic1 = parse_qrel(topics[0])
    # qrel_dic2 = parse_qrel(topics[1])
    # qrel_dic3 = parse_qrel(topics[2])
    # qrel_dic4 = parse_qrel(topics[3])
    # total = 0
    # for qid in qrel_dic1.keys():
    #     total += len(qrel_dic1[qid].keys())
    #     rel_docs1 = get_relevant_docs_by_qid(qrel_dic1, qid)
    #     rel_docs2 = get_relevant_docs_by_qid(qrel_dic2, qid)
    #     rel_docs3 = get_relevant_docs_by_qid(qrel_dic3, qid)
    #     rel_docs4 = get_relevant_docs_by_qid(qrel_dic4, qid)
    #
    #     num_docs = [len(rel_docs1), len(rel_docs2),
    #                 len(rel_docs3), len(rel_docs4)]
    #
    #     print(qid, num_docs)



