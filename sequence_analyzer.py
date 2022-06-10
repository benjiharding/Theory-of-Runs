import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nscore import NormalScoreTransformer
from lu_simulator import LUSimulator1D
import cdf


class SequenceAnalyzer:
    """Summarize high-order statistics and measures of non-Gaussianity
    for 1D sequences of data.
    """

    def __init__(
        self, data, dhid, var, quantiles, scale_factors, nreals, seed, runs_above=False,
    ):
        """Initialize SequenceAnalyzer class"""

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a `pd.DataFrame`")
        if not isinstance(dhid, str):
            raise ValueError("dhid must be a string")
        if not dhid in data.columns:
            raise ValueError("dhid must be a column in data")
        if not isinstance(var, str):
            raise ValueError("var must be a string")

        self.data = data.loc[~data[var].isna()]
        self.dhid = dhid
        self.var = var
        self.dhids = self.data[self.dhid].unique()
        self.ndh = len(self.dhids)
        self.quantiles = quantiles
        self.thresholds = {
            q / 100: t
            for q, t in zip(quantiles, np.percentile(self.data[self.var], quantiles))
        }
        self.scale_factors = scale_factors
        self.seed = seed
        self.nreals = nreals
        self.runs_above = runs_above
        self.metrics = [
            "scale_continuous",
            "total_runs",
            "run_length_freqs",
            "n_pt_connectivity",
        ]
        self._initialize_dict()

    def _initialize_dict(self):
        """Initialize data for subsequent tests and measures"""

        self.dh_dict = {dh: {} for dh in self.dhids}

        for dh in self.dhids:

            x = self.data.loc[self.data[self.dhid] == dh][[self.var]].copy()
            self.dh_dict[dh]["x_numpy"] = x.values.flatten()
            self.dh_dict[dh]["x_pandas"] = x

            # downsample the drillhole
            self.dh_dict[dh]["ds_x"] = {
                f: self.downsample(self.dh_dict[dh]["x_numpy"], f=f)
                for f in self.scale_factors
            }
            # indicator transform
            self.dh_dict[dh]["indicators"] = self.indicator_transform(x)
            self.dh_dict[dh]["ds_indicators"] = {
                f: self.indicator_transform(
                    pd.DataFrame(self.dh_dict[dh]["ds_x"][f], columns=[self.var])
                )
                for f in self.scale_factors
            }
            # frequencies of runs - drillhole
            self.dh_dict[dh]["runs"] = {
                q: self.binary_runs(self.dh_dict[dh]["indicators"][f"{q} Indicator"])
                for q in self.thresholds.keys()
            }
            # normal score transform
            nst = NormalScoreTransformer()
            self.dh_dict[dh]["ns_transformer"] = nst
            self.dh_dict[dh]["ns_data"] = nst.transform(x)

            # simulate and back transform
            self.dh_dict[dh]["gauss_reals"] = self.simulate(self.dh_dict[dh]["ns_data"])
            self.dh_dict[dh]["reals"] = np.zeros_like(self.dh_dict[dh]["gauss_reals"])
            for ireal in range(self.nreals):
                self.dh_dict[dh]["reals"][:, ireal] = nst.inverse_transform(
                    self.dh_dict[dh]["gauss_reals"][:, ireal]
                )
            # realization indicator transform
            self.dh_dict[dh]["ind_reals"] = self._realization_indicator_transform(
                self.dh_dict[dh]["reals"]
            )
            # frequencies of runs - realizations
            self.dh_dict[dh]["runs_reals"] = {
                q: self._binary_runs_realizations(self.dh_dict[dh]["ind_reals"][q])
                for q in self.thresholds.keys()
            }
            # downsample realizations
            self.dh_dict[dh]["ds_reals"] = {
                f: self._downsample_realizations(self.dh_dict[dh]["reals"], f=f)
                for f in self.scale_factors
            }
            self.dh_dict[dh]["ds_ind_reals"] = {
                f: self._realization_indicator_transform(
                    self.dh_dict[dh]["ds_reals"][f]
                )
                for f in self.scale_factors
            }
            # original drillhole sensitivity to scale
            cont_var, cont_mean = self.drillhole_scale_sensitivity(dh)
            self.dh_dict[dh]["cont_var"] = cont_var
            self.dh_dict[dh]["cont_means"] = cont_mean

            # realization sensitivity to scale
            sim_cont_var, sim_cont_mean = self.realization_scale_sensitivity(dh)
            self.dh_dict[dh]["sim_cont_var"] = sim_cont_var
            self.dh_dict[dh]["sim_cont_means"] = sim_cont_mean

    #
    # FUNCTIONS
    #

    def simulate(self, x):
        """LU simulation of 1D array (m, 1) where covariance is calculated from x."""
        return LUSimulator1D().lusim(x, self.seed, self.nreals)

    @staticmethod
    def _cdf_repr(reference, reals, log):
        reference = np.asarray(reference)
        nreals = reals.shape[1]
        fig, ax = plt.subplots()
        x, y = cdf.cdf(reference)
        ax.plot(x, y, c="r")
        for ireal in range(nreals):
            rx, ry = cdf.cdf(reals[:, ireal])
            ax.plot(rx, ry, lw=1, c="0.8", zorder=-1)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")
        return fig, ax

    def check_reals(self, dh, gauss=True, log=False):
        """Check histogram reprodcution in original or Gaussian units"""
        reference = (
            self.dh_dict[dh]["ns_data"] if gauss else self.dh_dict[dh]["x_numpy"]
        )
        reals = self.dh_dict[dh]["gauss_reals"] if gauss else self.dh_dict[dh]["reals"]
        return self._cdf_repr(reference, reals, log)

    def indicator_transform(self, df):
        """Transform var to indicators based on thresholds"""
        ind_cols = [f"{q} Indicator" for q in self.thresholds.keys()]
        df = df.reindex(columns=df.columns.tolist() + ind_cols)
        df[ind_cols] = 0
        for q, t in self.thresholds.items():
            df.loc[df[self.var] <= t, f"{q} Indicator"] = 1
        df = df.astype({col: int for col in ind_cols})
        return df

    def _realization_indicator_transform(self, reals):
        """Transform Gaussian realizations to indicators"""
        Z_indicators = {}
        for q, t in self.thresholds.items():
            Z_ind = np.zeros((reals.shape[0], self.nreals))
            Z_ind[reals <= t] = 1
            Z_indicators[q] = Z_ind
        return Z_indicators

    def binary_runs(self, x):
        """Calcualte runs and cumulative runs in binary array x"""
        runs_data = {}
        x = np.asarray(x)
        first_run = x[0]  # 1 or 0
        runstart = np.nonzero(np.diff(np.r_[[-np.inf], x, [np.inf]]))[0]
        runs = np.diff(runstart)
        runs = self._check_runs_above(first_run, runs)
        cum_runs = []
        for run in runs:
            for i in range(run):
                sub_run_length = run - i
                num_sub_runs = i + 1
                cum_runs.append([*[sub_run_length] * num_sub_runs])

        runs_data["runs"] = runs
        runs_data["cum_runs"] = np.array([a for b in cum_runs for a in b])
        runs_data["run_idxs"] = runstart
        runs_data["n_runs"] = len(runs)

        try:  # catch situation where all runs are below/above?
            runs_data["cum_runs_freqs"] = np.bincount(runs_data["cum_runs"])[1:]
            runs_data["runs_freqs"] = np.bincount(runs)[1:]
        except:
            runs_data["cum_runs_freqs"] = np.array([])
            runs_data["runs_freqs"] = np.array([])

        return runs_data

    def _check_runs_above(self, first_run, runs):
        if self.runs_above:
            if first_run:
                runs = runs[1::2]
            else:
                runs = runs[0::2]
            return runs
        else:
            return runs

    def _binary_runs_realizations(self, reals):
        """Calcualte runs and cumulative runs in binary array x for all realizations"""
        runs = {"runs": [], "cum_runs": [], "runs_freqs": [], "cum_runs_freqs": []}
        for ireal in range(self.nreals):
            runs_data = self.binary_runs(reals[:, ireal])
            runs["runs"].append(runs_data["runs"])
            runs["cum_runs"].append(runs_data["cum_runs"])
            runs["runs_freqs"].append(runs_data["runs_freqs"])
            runs["cum_runs_freqs"].append(runs_data["cum_runs_freqs"])
        return runs

    def downsample(self, x, f):
        """Average every f samples in x"""
        x = np.asarray(x)
        xp = np.r_[x, np.nan + np.zeros((-len(x) % f))]
        return np.nanmean(xp.reshape(-1, f), axis=-1)

    def _downsample_realizations(self, reals, f):
        """Average every f samples in x for all realizations"""
        ds_len = len(self.downsample(reals[:, 0], f=f))
        ds_reals = np.zeros((ds_len, self.nreals))
        for ireal in range(self.nreals):
            ds_reals[:, ireal] = self.downsample(reals[:, ireal], f=f)
        return ds_reals

    def oned_variogram(self, x):
        """Calculate 1D variogram along x"""
        x = np.asarray(x)
        nx = x.shape[0]
        gx = np.zeros(nx - 1)
        for i in range(1, nx - 1):
            z0 = x[0 : nx - i]
            z1 = x[i:nx]
            dz = (z1 - z0) ** 2
            gx[i] = np.sum(dz) / (2 * (nx - i))
        return gx

    def oned_covariance(self, x):
        """Calculate 1D covariance along x"""
        x = np.asarray(x)
        nx = x.shape[0]
        cx = np.zeros(nx - 1)
        for i in range(1, nx - 1):
            z0 = x[0 : nx - i]
            z1 = x[i:nx]
            dz = (z1 * z0) - np.mean(z0) * np.mean(z1)
            cx[i] = np.sum(dz) / (nx - i)
        return cx

    def n_pt_conn(self, x, nstep):
        """n-point connectivity fucntion of binary array x"""
        x = np.asarray(x)
        nx = x.shape[0]
        phi_n = []
        for n in range(1, nstep + 1):
            prod = []
            for i in range(nx - n + 1):
                idxs = [i] + [j + i for j in range(n)]
                a = [x[idx] for idx in idxs]
                prod.append(np.prod(a))
            phi_n.append(np.mean(prod))
        return phi_n

    def drillhole_scale_sensitivity(self, dh):
        """Calculate continuous and indicator variance at scale factors"""
        cont_var = {}
        cont_mean = {}
        for f in self.scale_factors:
            ds_cont = self.dh_dict[dh]["ds_x"][f]
            cont_var[f] = np.var(ds_cont)
            cont_mean[f] = np.mean(ds_cont)
        return cont_var, cont_mean

    def realization_scale_sensitivity(self, dh):
        """Calculate continuous and indicator variance at scale factors"""
        cont_var = {f: [] for f in self.scale_factors}
        cont_mean = {f: [] for f in self.scale_factors}
        for f in self.scale_factors:
            for ireal in range(self.nreals):
                cont_var[f].append(np.var(self.dh_dict[dh]["ds_reals"][f][:, ireal]))
                cont_mean[f].append(np.mean(self.dh_dict[dh]["ds_reals"][f][:, ireal]))
        return cont_var, cont_mean

    #
    # METRICS
    #

    # @staticmethod
    # def standardize(x, mu, sigma):
    #     EPS = 1e-6
    #     x = np.asarray(x)
    #     return (x - mu) / (sigma + EPS)

    @staticmethod
    def standardize(x, d, robust=False):
        EPS = 1e-6
        x = np.asarray(x)
        if robust:
            mdn = np.percentile(d, 50)
            iqr = np.percentile(d, 75) - np.percentile(d, 25)
            return (x - mdn) / (iqr + EPS)
        else:
            mu = np.mean(d)
            sigma = np.std(d)
            return (x - mu) / (sigma + EPS)

    def nongauss_measure(self, metric, max_runs=None, nstep=None):
        """Calculate measures of non-Gaussianity.
        Metrics: scale_continuous, total_runs, run_length_freqs, n_pt_connectivity
        """
        if metric not in self.metrics:
            raise ValueError(f"metric must be one of {self.metrics}")

        if metric == "scale_continuous":
            df = self._ng_scale_continuous()

        if metric == "total_runs":
            df = self._ng_total_runs()

        if metric == "run_length_freqs":
            df = self._ng_run_freqs(max_runs)

        if metric == "n_pt_connectivity":
            df = self._ng_n_pt_conn(nstep)

        return df

    def _ng_scale_continuous(self):
        """Scale based measure of non-Gaussianity from continuous values"""
        df = pd.DataFrame(columns=[f"Scale Factor {f}" for f in self.scale_factors])
        df.index.name = self.dhid
        for f in self.scale_factors:
            for dh in self.dhids:
                df.loc[dh, f"Scale Factor {f}"] = self._ng_scale_continuous_metric(
                    dh, f
                )
        return df

    # def _ng_scale_continuous_metric(self, dh, f):
    #     x = self.dh_dict[dh]["cont_var"][f]
    #     mu = np.mean(self.dh_dict[dh]["sim_cont_var"][f])
    #     sigma = np.std(self.dh_dict[dh]["sim_cont_var"][f])
    #     return np.abs(self.standardize(x, mu, sigma))

    def _ng_scale_continuous_metric(self, dh, f):
        x = self.dh_dict[dh]["cont_var"][f]
        d = self.dh_dict[dh]["sim_cont_var"][f]
        return np.abs(self.standardize(x, d))

    def _ng_total_runs(self):
        """Run based measure of non-Gaussianity from indicators"""
        df = pd.DataFrame(columns=[f"{q} Indicator" for q in self.thresholds.keys()])
        df.index.name = self.dhid
        for q, t in self.thresholds.items():
            for dh in self.dhids:
                df.loc[dh, f"{q} Indicator"] = self._ng_total_runs_metric(dh, q)
        return df

    # def _ng_total_runs_metric(self, dh, q, return_score=True):
    #     tot_runs = []
    #     for ireal in range(self.nreals):
    #         tot_runs.append(len(self.dh_dict[dh]["runs_reals"][q]["cum_runs"][ireal]))
    #         # tot_runs.append(len(self.dh_dict[dh]["runs_reals"][q]["runs"][ireal]))
    #     x = len(self.dh_dict[dh]["runs"][q]["cum_runs"])
    #     # x = len(self.dh_dict[dh]["runs"][q]["runs"])
    #     mu = np.mean(tot_runs)
    #     sigma = np.std(tot_runs)
    #     return np.abs(self.standardize(x, mu, sigma)) if return_score else tot_runs

    def _ng_total_runs_metric(self, dh, q, return_score=True):
        tot_runs = []
        for ireal in range(self.nreals):
            tot_runs.append(len(self.dh_dict[dh]["runs_reals"][q]["cum_runs"][ireal]))
        x = len(self.dh_dict[dh]["runs"][q]["cum_runs"])
        return np.abs(self.standardize(x, tot_runs)) if return_score else tot_runs

    def _ng_run_freqs(self, max_runs):
        """Run length frequency measure of non-Gaussianity from indicators        
        """
        if max_runs is None:
            raise ValueError("max_runs must be specified")
        df = pd.DataFrame(columns=[f"{q} Indicator" for q in self.thresholds.keys()])
        df.index.name = self.dhid
        for q, t in self.thresholds.items():
            for dh in self.dhids:
                df.loc[dh, f"{q} Indicator"] = self._ng_run_freqs_metric(
                    dh, q, max_runs
                )
        return df

    # def _ng_run_freqs_metric(self, dh, q, max_runs, return_score=True):
    #     dh_runs = self.dh_dict[dh]["runs"][q]["cum_runs_freqs"]
    #     nruns = min(len(dh_runs), max_runs)
    #     temp = np.zeros((self.nreals, len(self.dh_dict[dh]["x_numpy"])))
    #     for ireal in range(self.nreals):
    #         real_run_freqs = self.dh_dict[dh]["runs_reals"][q]["cum_runs_freqs"][ireal]
    #         idxs = np.arange(len(real_run_freqs))
    #         temp[ireal, idxs] = real_run_freqs
    #     x = dh_runs[:nruns]
    #     mu = np.mean(temp[:, :nruns], axis=0)
    #     sigma = np.std(temp[:, :nruns], axis=0)
    #     return (
    #         np.mean(np.abs(self.standardize(x, mu, sigma)))
    #         if return_score
    #         else temp
    #     )

    def _ng_run_freqs_metric(self, dh, q, max_runs, return_score=True):
        dh_runs = self.dh_dict[dh]["runs"][q]["cum_runs_freqs"]
        nruns = min(len(dh_runs), max_runs)
        temp = np.zeros((self.nreals, len(self.dh_dict[dh]["x_numpy"])))
        for ireal in range(self.nreals):
            real_run_freqs = self.dh_dict[dh]["runs_reals"][q]["cum_runs_freqs"][ireal]
            idxs = np.arange(len(real_run_freqs))
            temp[ireal, idxs] = real_run_freqs
        x = dh_runs[:nruns]
        d = temp[:, :nruns]
        scores = [self.standardize(x[i], d[:, i]) for i in range(nruns)]
        score = np.mean(np.abs(scores))
        return score if return_score else temp

    def _ng_n_pt_conn(self, nstep):
        """n-point connectivity function measure of non-Gaussianity from indicators"""
        if nstep is None:
            raise ValueError("nstep must be specified")
        df = pd.DataFrame(columns=[f"{q} Indicator" for q in self.thresholds.keys()])
        df.index.name = self.dhid
        for q, t in self.thresholds.items():
            for dh in self.dhids:
                df.loc[dh, f"{q} Indicator"] = self._ng_n_pt_conn_metric(dh, q, nstep)
        return df

    def _ng_n_pt_conn_metric(self, dh, q, nstep, return_score=True):
        x = self.dh_dict[dh]["indicators"][f"{q} Indicator"].values
        dh_conn = self.n_pt_conn(x, nstep)
        temp = np.zeros((self.nreals, nstep))
        for ireal in range(self.nreals):
            ind_real = self.dh_dict[dh]["ind_reals"][q][:, ireal]
            temp[ireal, :] = self.n_pt_conn(ind_real, nstep)
        scores = [self.standardize(dh_conn[i], temp[:, i]) for i in range(nstep)]
        score = np.mean(np.abs(scores))
        return score if return_score else temp

    # def _ng_n_pt_conn_metric(self, dh, q, nstep, return_score=True):
    #     x = self.dh_dict[dh]["indicators"][f"{q} Indicator"].values
    #     ind_reals = self.dh_dict[dh]["ind_reals"][q]
    #     if self.runs_above:
    #         x = 1 - x
    #         ind_reals = 1 - ind_reals
    #     dh_conn = self.n_pt_conn(x, nstep)
    #     temp = np.zeros((self.nreals, nstep))
    #     for ireal in range(self.nreals):
    #         ind_real = ind_reals[:, ireal]
    #         temp[ireal, :] = self.n_pt_conn(ind_real, nstep)
    #     scores = [self.standardize(dh_conn[i], temp[:, i]) for i in range(nstep)]
    #     score = np.mean(np.abs(scores))
    #     return score if return_score else temp

    #
    # PLOTTING
    #

    def plot_dh(self, dh, gauss=True, figsize=(20, 1), **kwargs):
        """Plot continuous values from a drillhole"""
        fig, ax = plt.subplots(figsize=figsize)
        c = self.dh_dict[dh]["ns_data"] if gauss else self.dh_dict[dh]["x_numpy"]
        x = np.arange(len(c))
        y = np.ones_like(x) * 0.5
        im = ax.scatter(x, y, c=c, **kwargs)
        ax.yaxis.set_ticklabels([])
        plt.colorbar(im, ax=ax)
        return fig, ax

    def plot_downscaled_dh(self, dh, figsize=(15, 10), **kwargs):
        """Plot downsacled continuous values from a drillhole"""
        ds = self.dh_dict[dh]["ds_x"]
        fig, axes = plt.subplots(len(self.scale_factors), 1, figsize=figsize)
        for ax, f in zip(axes, self.scale_factors):
            c = ds[f]
            x = np.arange(len(c))
            y = np.ones_like(x) * 0.5
            im = ax.scatter(x, y, c=c, **kwargs)
            ax.yaxis.set_ticklabels([])
            plt.colorbar(im, ax=ax)
        return fig, axes

    def plot_indicators(self, dh, figsize=(15, 5), **kwargs):
        """Plot indicator values from a drillhole"""
        nrows = len(self.thresholds)
        ind = self.dh_dict[dh]["indicators"]
        fig, axes = plt.subplots(nrows, 1, figsize=figsize)
        for ax, q in zip(axes, self.thresholds.keys()):
            x = np.arange(len(ind))
            y = np.ones_like(x) * 0.5
            im = ax.scatter(
                x, y, c=ind[f"{q} Indicator"].values, vmin=0, vmax=1, **kwargs
            )
            ax.set_title(f"{q} Indicator")
            ax.yaxis.set_ticklabels([])
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
            cbar.set_ticklabels([0, 1])
        return fig, ax

    def _plot_run_freqs(self, dh, q, max_runs, figsize, **kwargs):
        """Plot frequencies of cumualtive run lengths"""
        fig, ax = plt.subplots(figsize=figsize)
        dh_runs = self.dh_dict[dh]["runs"][q]["cum_runs_freqs"]
        nruns = min(len(dh_runs), max_runs)
        xx = np.arange(nruns)
        ax.plot(dh_runs[:nruns], marker=".", c="C3", label="Drillhole Runs")
        real_values = self._ng_run_freqs_metric(dh, q, max_runs, return_score=False)
        for n in range(nruns):
            if n == nruns - 1:
                label = "Gaussian Reals"
            else:
                label = None
            ax.scatter(
                np.ones(self.nreals) * n,
                real_values[:, n],
                c="k",
                s=15,
                alpha=0.1,
                label=label,
            )
        ax.set_xticks(np.arange(nruns))
        ax.set_xticklabels(np.arange(nruns) + 1)
        ax.set_xlabel("Cumulative Run Length")
        ax.set_ylabel("Frequency")
        ax.legend()
        return fig, ax

    def _plot_n_pt_conn(self, dh, q, nstep, figsize, **kwargs):
        """Plot n-point connectivity function"""
        fig, ax = plt.subplots(figsize=figsize)
        x = self.dh_dict[dh]["indicators"][f"{q} Indicator"].values
        dh_conn = self.n_pt_conn(x, nstep)
        xx = np.arange(nstep)
        ax.plot(dh_conn, marker=".", c="C2", label="Drillhole n-Point Connectivity")
        real_values = self._ng_n_pt_conn_metric(dh, q, nstep, return_score=False)
        for n in range(nstep):
            if n == nstep - 1:
                label = "Gaussian Reals"
            else:
                label = None
            ax.scatter(
                np.ones(self.nreals) * n,
                real_values[:, n],
                c="k",
                s=15,
                alpha=0.1,
                label=label,
            )
        ax.set_xticks(np.arange(nstep))
        ax.set_xticklabels(np.arange(nstep) + 1)
        ax.set_xlabel("Connected Steps")
        ax.set_ylabel("Prob. of Connection")
        ax.legend()
        return fig, ax

    def _plot_total_runs(self, dh, q, figsize, **kwargs):
        """Plot distribution of total runs"""
        fig, ax = plt.subplots(figsize=figsize)
        dh_tot_runs = len(self.dh_dict[dh]["runs"][q]["cum_runs"])
        # dh_tot_runs = len(self.dh_dict[dh]["runs"][q]["runs"])
        real_tot_runs = self._ng_total_runs_metric(dh, q, return_score=False)
        mu = np.mean(real_tot_runs)
        sigma = np.std(real_tot_runs)
        ax.hist(real_tot_runs, histtype="step", color="C0")
        ax.axvline(dh_tot_runs, c="k", label="Drillhole Value", ls=":")
        ax.axvline(mu, c="C0", label="Expected Value", ls=":")
        ax.set_xlabel("Total Runs")
        ax.set_ylabel("Frequency")
        ax.legend(loc=1)
        return fig, ax

    def _plot_variance(self, dh, figsize, **kwargs):
        """Plot variance vs scale factor"""
        fig, ax = plt.subplots()
        dh_var = self.dh_dict[dh]["cont_var"]
        sim_var = self.dh_dict[dh]["sim_cont_var"]
        ax.plot(
            list(dh_var.keys()), list(dh_var.values()), marker=".", label="Drillhole"
        )
        for f in sim_var.keys():
            if f == list(sim_var.keys())[-1]:
                label = "Gaussian Reals"
            else:
                label = None
            ax.scatter(
                np.ones(self.nreals) * f,
                sim_var[f],
                c="k",
                s=15,
                alpha=0.1,
                label=label,
            )
        ax.legend()
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("Variance")
        return fig, ax

    def diagnostic_plot(
        self,
        metric,
        dh,
        q=None,
        f=None,
        max_runs=None,
        nstep=None,
        figsize=None,
        **kwargs,
    ):
        """Plot measures of non-Gaussianity.
        Metrics: scale_continuous, total_runs, run_length_freqs, n_pt_connectivity
        `scale_continuous` args: dh
        `total_runs` args: dh, q
        `run_length_freqs` args: dh, q, max_runs
        `n_pt_connectivity` args: dh, q, nstep
        """
        if metric not in self.metrics:
            raise ValueError(f"metric must be one of {self.metrics}")

        if metric == "scale_continuous":
            fig, ax = self._plot_variance(dh, figsize, **kwargs)

        if metric == "total_runs":
            fig, ax = self._plot_total_runs(dh, q, figsize, **kwargs)

        if metric == "run_length_freqs":
            fig, ax = self._plot_run_freqs(dh, q, max_runs, figsize, **kwargs)

        if metric == "n_pt_connectivity":
            fig, ax = self._plot_n_pt_conn(dh, q, nstep, figsize, **kwargs)

        return fig, ax
