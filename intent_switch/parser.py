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


if __name__ == "__main__":
    topic_dic = parse_topics("topics")
    qrel_dic = parse_qrel("1.txt")
    print(qrel_dic)
