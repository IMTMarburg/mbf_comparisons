import functools
import pandas as pd
import pypipegraph as ppg
from mbf_qualitycontrol import register_qc, qc_disabled
from dppd import dppd
import dppd_plotnine  # noqa: F401
from mbf_genomics.util import parse_a_or_c
from mbf_genomics import DelayedDataFrame
from .annotator import ComparisonAnnotator

dp, X = dppd()


class Comparisons:
    """A ddf + comparison groups,
    ready for actually doing comparisons

    Paramaters:

        groups_to_samples: { keyX: [columnA, annoB, (annoC, column_name), (annoC, 2),
                                keyY: ..., ...}
            keyX: one of the keys of groups_to_samples
            keyY: one of the keys of groups_to_samples
    """

    def __init__(self, ddf, groups_to_samples):
        if not isinstance(ddf, DelayedDataFrame):
            raise ValueError("Ddf must be a DelayedDataFrame")
        self.ddf = ddf
        self.groups_to_samples = self._check_input_dict(groups_to_samples)
        self.sample_column_to_group = self._sample_columns_to_group()
        self.samples = functools.reduce(
            list.__add__, [x[1] for x in sorted(self.groups_to_samples.items())]
        )
        self.name = "comparison__" + "_".join(sorted(self.groups_to_samples.keys()))
        self.result_dir = self.ddf.result_dir / self.name
        self.result_dir.mkdir(exist_ok=True, parents=True)
        if ppg.inside_ppg():
            ppg.assert_uniqueness_of_object(self)
            if not hasattr(ppg.util.global_pipegraph, "_mbf_comparisons_name_dedup"):
                ppg.util.global_pipegraph._mbf_comparisons_name_dedup = set()
            for name in self.groups_to_samples:
                if name in ppg.util.global_pipegraph._mbf_comparisons_name_dedup:
                    raise ValueError(
                        f"Comparisons group {name} defined in multiple Comparisons - not supported"
                    )

        self.register_qc()

    def a_vs_b(self, a, b, method, laplace_offset=1 / 1e6):
        if a not in self.groups_to_samples:
            raise KeyError(a)
        if b not in self.groups_to_samples:
            raise KeyError(a)
        if not hasattr(method, "compare"):
            raise TypeError(f"{method} had no method compare")
        res = ComparisonAnnotator(self, a, b, method, laplace_offset)
        self.ddf += res
        return res

    def all_vs_b(self, b, method, laplace_offset=1 / 1e6):
        res = {}
        for a in self.groups_to_samples:
            if a != b:
                res[a] = self.a_vs_b(a, b, method, laplace_offset)
        return res

    def _check_input_dict(self, groups_to_samples):
        if not isinstance(groups_to_samples, dict):
            raise ValueError("groups_to_samples must be a dict")
        for k, v in groups_to_samples.items():
            if not isinstance(k, str):
                raise ValueError("keys must be str, was %s %s" % (k, type(k)))
            v = [parse_a_or_c(x) for x in v]
            groups_to_samples[k] = v

        return groups_to_samples

    def _sample_columns_to_group(self):
        result = {}
        for group, samples in self.groups_to_samples.items():
            for ac in samples:
                c = ac[1]
                if c in result:
                    raise ValueError(
                        f"Sample in multiple groups - not supported {ac}, {group}, {result[ac]}"
                    )
                result[c] = group
        return result

    def register_qc(self):
        if not qc_disabled():
            self.register_qc_distribution()
            self.register_qc_pca()
            self.register_qc_correlation()

    def find_variable_name(self):
        for anno, column in self.samples:
            if anno is not None and hasattr(anno, "unit"):
                return anno.unit
        return "value"

    def get_plot_name(self, column):
        for ac in self.samples:
            if ac[1] == column:
                if ac[0] is not None:
                    return getattr(ac[0], "plot_name", column)
                else:
                    return column
        raise KeyError(column)

    def get_df(self):
        return self.ddf.df[[column for anno, column in self.samples]]

    def register_qc_distribution(self):
        output_filename = self.result_dir / "distribution.png"

        def plot(output_filename):
            return (
                dp(self.get_df())
                .melt(var_name="sample", value_name="y")
                .assign(
                    var_name=[self.get_plot_name(x) for x in X["sample"]],
                    group=[self.sample_column_to_group[x] for x in X["sample"]],
                )
                .p9()
                .theme_bw()
                .annotation_stripes()
                .geom_violin(dp.aes("sample", "y"), width=0.5)
                .add_boxplot(x="sample", y="y", _width=0.1, _fill=None, color="group")
                .scale_color_many_categories()
                .scale_y_continuous(trans="log10", name=self.find_variable_name())
                .turn_x_axis_labels()
                .hide_x_axis_title()
                .render(output_filename)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(self.deps())
        )

    def deps(self):
        return [
            self.ddf.add_annotator(ac[0]) for ac in self.samples if ac[0] is not None
        ] + [
            self.ddf.load()
        ]  # you might be working with an anno less ddf afterall

    def register_qc_pca(self):
        output_filename = self.result_dir / "pca.png"

        def plot(output_filename):
            import sklearn.decomposition as decom

            pca = decom.PCA(n_components=2, whiten=False)
            data = self.get_df()
            # min max scaling 0..1 per gene
            data = data.sub(data.min(axis=1), axis=0)
            data = data.div(data.max(axis=1), axis=0)

            data = data[~pd.isnull(data).any(axis=1)]  # can' do pca on NAN values
            pca.fit(data.T)
            xy = pca.transform(data.T)
            title = "PCA %s (%s)\nExplained variance: x %.2f%%, y %.2f%%" % (
                self.ddf.name,
                self.find_variable_name(),
                pca.explained_variance_ratio_[0] * 100,
                pca.explained_variance_ratio_[1] * 100,
            )
            plot_df = pd.DataFrame(
                {
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "label": [self.get_plot_name(c) for (a, c) in self.samples],
                    "group": [
                        self.sample_column_to_group[c] for (a, c) in self.samples
                    ],
                }
            )
            (
                dp(plot_df)
                .p9()
                .theme_bw()
                .add_scatter("x", "y", color="group")
                .add_text(
                    "x",
                    "y",
                    "label",
                    _adjust_text={
                        "expand_points": (2, 2),
                        "arrowprops": {"arrowstyle": "->", "color": "darkgrey"},
                    },
                )
                .scale_color_many_categories()
                .title(title)
                .render(output_filename, width=8, height=6)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(self.deps())
        )

    def register_qc_correlation(self):
        output_filename = self.result_dir / "pearson_correlation.png"

        def plot(output_filename):
            data = self.get_df()
            data = data.sub(data.min(axis=1), axis=0)
            data = data.div(data.max(axis=1), axis=0)
            # data -= data.min()  # min max scaling 0..1 per gene
            # data /= data.max()
            data = data[
                ~pd.isnull(data).any(axis=1)
            ]  # can' do correlation on NAN values
            data.columns = [self.get_plot_name(x) for x in data.columns]
            pdf = pd.melt(data.corr().reset_index(), "index")
            (
                dp(pdf)
                .p9()
                .add_tile("index", "variable", fill="value")
                .scale_fill_gradient2(
                    "blue", "white", "red", limits=[-1, 1], midpoint=0
                )
                .hide_x_axis_title()
                .hide_y_axis_title()
                .turn_x_axis_labels()
                .render(output_filename)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(self.deps())
        )
