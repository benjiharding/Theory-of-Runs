import numpy as np
from scipy import stats
from cdf import cdf, percentile_from_cdf


class NormalScoreTransformer:
    def __init__(self):
        self.data = None
        self.ns_data = None
        self.wts = None
        self.lower_tail = None
        self.upper_tail = None
        self.transform_table = None
        self.cdf_x = None
        self.cdfvals = None
        self.order = None

    # this needs a fit method, then transform

    def transform(self, data, wts=None, tail_values=(None, None)):

        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
                data = data[np.isfinite(data)]
            except:
                raise ValueError("data must be a np.ndarray")
        assert data.ndim == 1, "data must be a 1D np.ndarray"

        if any(item is None for item in tail_values):
            self.lower_tail = min(data) - 0.001
            self.upper_tail = max(data) + 0.001
        else:
            self.lower_tail = tail_values[0]
            self.upper_tail = tail_values[1]

        self.data = data[np.isfinite(data)]
        self.order = self.data.argsort()
        self.wts = wts

        # build and forward transform the weighted empirical cdf
        self.cdf_x, self.cdfvals = self._build_cdf(
            tail_values=(self.lower_tail, self.upper_tail)
        )
        self.ns_data = stats.norm.ppf(self.cdfvals, loc=0, scale=1)

        self.transform_table = np.concatenate(
            [self.cdf_x[1:-1].reshape(-1, 1), self.ns_data[1:-1].reshape(-1, 1)], axis=1
        )
        return self.ns_data[1:-1][self.order.argsort()]

    def inverse_transform(self, ns_data):

        if not isinstance(ns_data, np.ndarray):
            try:
                ns_data = np.array(ns_data)
                ns_data = ns_data[np.isfinite(ns_data)]
            except:
                raise ValueError("data must be a np.ndarray")
        assert ns_data.ndim == 1, "data must be a 1D np.ndarray"

        # calculate quantiles of the Gaussian data
        q = stats.norm.cdf(ns_data, loc=0, scale=1) * 100

        # get the original units cdf consiering tail values for interpolation
        xvals = percentile_from_cdf(self.cdf_x, self.cdfvals, q)

        return xvals

    def _build_cdf(self, tail_values=(None, None)):

        if self.wts is None:
            self.wts = np.ones(len(self.data))
        else:
            self.wts = np.array(self.wts)

        cdf_x, cdfvals = cdf(
            self.data, weights=self.wts, lower=tail_values[0], upper=tail_values[1]
        )
        return cdf_x, cdfvals
