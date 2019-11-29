from dataset.AbstractDataset import AbstractDataset
import numpy as np


class LetorDataset(AbstractDataset):

    def __init__(self, path,
                 feature_size,
                 query_level_norm=False):
        super().__init__(path, feature_size, query_level_norm)
        self._load_data()

    def _load_data(self):
        with open(self._path) as fin:
            current_query = None
            for line in fin:
                cols = line.strip().split()
                query = cols[1].split(':')[1]
                if query == current_query:
                    docid = len(self._query_get_docids[query])
                    old_query = True

                else:
                    if current_query != None and self._query_level_norm:
                        norm = np.zeros((len(self._query_get_docids[current_query]), self._feature_size))
                        # if there is more than 1 candidate docs, do the norm
                        if norm.shape[0] != 1:
                            query_features = self._query_get_all_features[current_query]
                            min = np.amin(query_features, axis=0)
                            max = np.amax(query_features, axis=0)
                            safe_ind = max - min != 0
                            norm[:, safe_ind] = (query_features[:, safe_ind] - min[safe_ind]) / (
                                    max[safe_ind] - min[safe_ind])
                            self._query_get_all_features[current_query] = norm
                    old_query = False
                    docid = 0
                    current_query = query

                relevence = int(cols[0])
                features = [0] * self._feature_size

                for i in range(2, len(cols)):
                    feature_id = cols[i].split(':')[0]

                    if not feature_id.isdigit():
                        break

                    feature_id = int(feature_id) - 1
                    feature_value = float(cols[i].split(':')[1])

                    features[feature_id] = feature_value

                if relevence > 0:
                    if query in self._query_pos_docids:
                        self._query_pos_docids[query].append(docid)
                    else:
                        self._query_pos_docids[query] = [docid]

                if old_query:
                    self._query_docid_get_features[query][docid] = np.array(features)
                    self._query_get_docids[query].append(docid)
                    self._query_get_all_features[query] = np.vstack((self._query_get_all_features[query], features))
                    self._query_docid_get_rel[query][docid] = relevence
                else:
                    self._query_docid_get_features[query] = {docid: np.array(features)}
                    self._query_get_docids[query] = [docid]
                    self._query_get_all_features[query] = np.array([features])
                    self._query_docid_get_rel[query] = {docid: relevence}

    def get_features_by_query_and_docid(self, query, docid):
        return self._query_docid_get_features[query][docid]

    def get_candidate_docids_by_query(self, query):
        return self._query_get_docids[query]

    def get_all_features_by_query(self, query):
        return self._query_get_all_features[query]

    def get_relevance_label_by_query_and_docid(self, query, docid):
        return self._query_docid_get_rel[query][docid]

    def get_relevance_docids_by_query(self, query):
        return self._query_pos_docids[query]

    def get_all_querys(self):
        return np.array(list(self._query_get_all_features.keys()))
