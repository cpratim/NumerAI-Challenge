import warnings


class ModeClustering(object):

    def __init__(self, n_feat_samples=100, n_target_acc=3, disp_warnings=False):

        self.fitted = False
        self.mode_distributions = {}
        self.n_feat_samples = n_feat_samples
        self.n_target_acc = n_target_acc
        if not disp_warnings:
            warnings.filterwarnings('ignore')

    def fit(self, X, y):

        targets = sorted(list(set(y_train)))
        target_feat = {}
        for _x, _y in zip(X, y):
            if _y not in target_feat:
                target_feat[_y] = []
            target_feat[_y].append(_x)
        feat_dists = {}
        for t in targets:
            feat = np.array(target_feat[t]).T
            mean_dist = sorted(
                [(np.mean(i), np.std(i), stats.mode(i)[0][0], n) for n, i in enumerate(feat)],
                key = lambda x: x[1],
                reverse = True,
            )[:self.n_feat_samples]
            feat_dists[t] = mean_dist

        for t, d in feat_dists.items():
            atr = np.array(d).T
            self.mode_distributions[t] = (atr[-1], atr[2])

        self.fitted = True

    def predict(self, X):

        assert self.fitted == True
        y_pred = []
        for x in tqdm(X, desc="Processing..."):
            sim = []
            for t, v in self.mode_distributions.items():
                pred = 0
                _mp = [x[int(i)] for i in v[0]]
                match = match_index(_mp, v[1])
                sim.append((t, match, len(match)))
            sp = sorted(sim, key = lambda x: x[2], reverse=True)[:self.n_target_acc]
            ts = np.sum(sp, axis=0)[2]
            for n, (t, m, ml) in enumerate(sp):
                if ts != 0:
                    pred += t * (ml/ts)
            y_pred.append(pred)
        return y_pred
