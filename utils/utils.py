from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

class Recorder():

    def __init__(self, early_step):
        self.max = {'fscore': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['症状']['fscore'] > self.max['fscore']:
            self.max['fscore'] = self.cur['症状']['fscore']
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

def metrics(y_true, y_pred, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    category = [[0,1,2,3]]*len(y_true)
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c[0]]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

    # metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    # y_pred = np.around(np.array(y_pred)).astype(int)
    # metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    # metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    # metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    # metrics_by_category['acc'] = accuracy_score(y_true, y_pred)
    
    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'auc': metrics_by_category[c]['auc'],
                'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
            }
        except Exception as e:
            metrics_by_category[c] = {
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }
    return metrics_by_category


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': [sample[0].cuda() for sample in batch],
            'content_masks': [sample[1].cuda() for sample in batch],
            'label': [sample[2].cuda() for sample in batch]
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'label': batch[2],
            }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
