from dataset.AbstractDataset import AbstractDataset
import numpy as np
import math
import os


class LetorDataset(AbstractDataset):

    def __init__(self, path,
                 feature_size,
                 query_level_norm=False,
                 binary_label=False):
        super().__init__(path, feature_size, query_level_norm)
        self._binary_label = binary_label
        self._comments = {}
        self._docid_map = {}

        self._load_data()

    def _load_data(self):
        with open(self._path, "r") as fin:
            current_query = None
            for line in fin:
                cols = line.strip().split()
                query = cols[1].split(':')[1]
                if query == current_query:
                    docid = len(self._query_get_docids[query])
                    old_query = True

                else:
                    if current_query != None and self._query_level_norm:
                        self._normalise(current_query)
                    old_query = False
                    docid = 0
                    current_query = query
                    self._docid_map[query] = {}
                    self._query_pos_docids[query] = []

                comments_part = line.split("#")
                if len(comments_part) == 2:
                    if query not in self._comments:
                        self._comments[query] = []
                    self._comments[query].append(comments_part[1].strip())

                relevence = float(cols[0])  # Sometimes the relevance label can be a float.
                if relevence.is_integer():
                    relevence = int(relevence)  # But if it is indeed an int, cast it into one.
                    if self._binary_label and relevence > 0:
                        relevence = 1

                features = [0] * self._feature_size

                for i in range(2, len(cols)):
                    feature_id = cols[i].split(':')[0]

                    if not feature_id.isdigit():
                        if feature_id[0] == "#":
                            self._docid_map[query][docid] = feature_id[1:]
                        break

                    feature_id = int(feature_id) - 1
                    feature_value = float(cols[i].split(':')[1])
                    if math.isnan(feature_value):
                        feature_value = 0

                    features[feature_id] = feature_value

                if relevence > 0:
                    self._query_pos_docids[query].append(docid)

                if old_query:
                    self._query_docid_get_features[query][docid] = np.array(features)
                    self._query_get_docids[query].append(docid)
                    self._query_get_all_features[query] = np.vstack((self._query_get_all_features[query], features))
                    self._query_docid_get_rel[query][docid] = relevence
                    self._query_relevant_labels[query].append(relevence)
                else:
                    self._query_docid_get_features[query] = {docid: np.array(features)}
                    self._query_get_docids[query] = [docid]
                    self._query_get_all_features[query] = np.array([features])
                    self._query_docid_get_rel[query] = {docid: relevence}
                    self._query_relevant_labels[query] = [relevence]

        if self._query_level_norm:
            self._normalise(current_query)

    def _normalise(self, query):
        norm = np.zeros((len(self._query_get_docids[query]), self._feature_size))
        # if there is more than 1 candidate docs, do the norm
        if norm.shape[0] != 1:
            query_features = self._query_get_all_features[query]
            min = np.amin(query_features, axis=0)
            max = np.amax(query_features, axis=0)
            safe_ind = max - min != 0
            norm[:, safe_ind] = (query_features[:, safe_ind] - min[safe_ind]) / (
                    max[safe_ind] - min[safe_ind])
            self._query_get_all_features[query] = norm

    def update_relevance_label(self, qrel_dic: dict):
        for qid in self._query_docid_get_rel.keys():

            self._query_pos_docids[qid] = []
            ind = 0
            for docid in self._query_docid_get_rel[qid].keys():
                rel = qrel_dic[qid][self._docid_map[qid][docid]]
                self._query_docid_get_rel[qid][docid] = rel
                self._query_relevant_labels[qid][ind] = rel
                if rel > 0:
                    self._query_pos_docids[qid].append(docid)
                ind += 1


    def get_features_by_query_and_docid(self, query, docid):
        return self._query_docid_get_features[query][docid]

    def get_candidate_docids_by_query(self, query):
        return self._query_get_docids[query]

    def get_all_features_by_query(self, query):
        return self._query_get_all_features[query]

    def get_relevance_label_by_query_and_docid(self, query, docid):
        return self._query_docid_get_rel[query][docid]

    def get_all_relevance_label_by_query(self, query):
        return self._query_relevant_labels[query]

    def get_relevance_docids_by_query(self, query):
        return self._query_pos_docids[query]

    def get_all_querys(self):
        return np.array(list(self._query_get_all_features.keys()))

    def get_all_comments_by_query(self, query):
        return self._comments[query]

    def write(self, output_file):
        s = ""
        for query in self.get_all_querys():
            comments = self.get_all_comments_by_query(query)
            for i, features in enumerate(self.get_all_features_by_query(query)):
                comment = comments[i]
                label = self.get_relevance_label_by_query_and_docid(query, i)
                features_str = ""
                for i, feature in enumerate(features):
                    # if feature == 0:
                    #     continue
                    features_str += "{}:{} ".format(i + 1, feature)
                s += "{} qid:{} {}# {}\n".format(label, query, features_str, comment)
        with open(output_file, "w") as f:
            f.write(s)

    def write_cross_validation_datasets(self, path: str, fold_num: int):
        """
        :param fold_num: number of fold to do cross validation.
        :param path: folder address to store the cross sets.
        :return:
        """

        for fold in range(fold_num):
            fold_path = f"{path}/Fold{fold+1}"
            # Create target Directory if don't exist
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
                print("Directory ", fold_path, " Created ")
            else:
                print("Directory ", fold_path, " already exists")

