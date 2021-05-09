import math
def write_to_disk(path, query_doc_dict):

    with open(path, "w") as f:
        for query in query_doc_dict.keys():
            for docid in query_doc_dict[query].keys():
                s = ""
                label, features= query_doc_dict[query][docid]


                features_str = ""
                for i, feature in enumerate(features):
                    # if feature == 0:
                    #     continue
                    features_str += "{}:{} ".format(i + 1, feature)
                s += "{} qid:{} {}#{}\n".format(label, query, features_str, docid)
                f.write(s)


def load_data_set(path):
    query_doc_dict = {}
    with open(path, "r") as fin:
        current_query = None
        for line in fin:
            cols = line.strip().split()
            query = cols[1].split(':')[1]
            if query == current_query:
                old_query = True

            else:
                old_query = False
                current_query = query

            relevence = cols[0]

            features = []
            for i in range(2, len(cols)):
                feature_id = cols[i].split(':')[0]

                if not feature_id.isdigit():
                    if feature_id[0] == "#":
                        docid = cols[i][1:]
                    break

                feature_value = float(cols[i].split(':')[1])
                if math.isnan(feature_value):
                    feature_value = 0

                features.append(feature_value)
            if len(features) >= 106:
                features = features[:105]

                # features.append(-22.2287221840378)
            if old_query:
                query_doc_dict[query][docid] = (relevence, features)
            else:
                query_doc_dict[query] = {docid: (relevence, features)}
    return query_doc_dict

if __name__ == "__main__":
    num_added = 0
    query_doc_dict = load_data_set("clueweb09_intent_change.txt")

    # with open("../datasets/clueweb09/ClueWeb09-En-PR.prior.txt", "r") as fin:
    #     for line in fin:
    #         cols = line.strip().split()
    #         docid = cols[0]
    #         value = cols[1]
    #         for qid in query_doc_dict.keys():
    #             try:
    #                 query_doc_dict[qid][docid][1].append(value)
    #                 num_added += 1
    #                 print(num_added)
    #             except:
    #                 pass
    #
    write_to_disk("clueweb09_intent_change.txt", query_doc_dict)