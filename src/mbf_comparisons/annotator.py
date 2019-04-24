from mbf_genomics.annotator import Annotator
import itertools
import pypipegraph as ppg
import numpy as np
import pandas as pd
import collections
from mbf_qualitycontrol import register_qc
from mbf_genomics.util import (
    parse_a_or_c_to_column,
    parse_a_or_c_to_anno,
    parse_a_or_c_to_plot_name,
)
import dppd
import dppd_plotnine  # noqa: F401

dp, X = dppd.dppd()

# import pypipegraph as ppg


class Comparison(Annotator):
    def __init__(
        self, comparison_strategy, groups_to_samples, a, b, laplace_offset=1 / 1e6
    ):
        """Create a comparison (a - b)

        Parameters:
            comparison_strategy:  a ComparisonStrategy - see mbf_comparisons.comparisons
            groups_to_samples: { keyX: [columnA, annoB, (annoC, column_name), (annoC, 2),
                                keyY: ...}
            a: one of the keys of groups_to_samples
            b: one of the keys of groups_to_samples
            """

        if hasattr(comparison_strategy, "__call__"):
            self.comparison_strategy = comparison_strategy()
        else:
            self.comparison_strategy = comparison_strategy
        if isinstance(
            self.comparison_strategy.columns, str
        ):  # pragma: no cover definsive
            raise ValueError(
                "ComparisonStrategy %s had a string as columns, must be a list"
                % self.comparison_strategy
            )
        self.groups_to_samples = self._check_input_dict(groups_to_samples)
        self._check_comparison_groups(a, b)
        self.comp = (a, b)
        self.columns = []
        self.column_lookup = {}
        for col in sorted(self.comparison_strategy.columns):
            cn = self.name_column(col)
            self.columns.append(cn)
            self.column_lookup[col] = cn
        self.laplace_offset = laplace_offset

    def name_column(self, col):
        return f"Comp. {self.comp[0]} - {self.comp[1]} {col} ({self.comparison_strategy.name})"

    def __getitem__(self, itm):
        """look up the full column name from log2FC, p, FDR, etc"""
        return self.column_lookup[itm]

    def filter(self, genes, filter_definition, new_name=None):
        """Turn a filter definition [(column, operator, threshold)...]
        into a filtered genes object.

        Example:
        comp.filter(genes, '2x', [
            (log2FC, '|>', 1),  #absolute
            ('FDR', '<=', 0.05)
            ]
        """
        if new_name is None:
            filter_str = []
            for column, op, threshold in sorted(filter_definition):
                filter_str.append(f"{column}_{op}_{threshold:.2f}")
            filter_str = "__".join(filter_str)
            new_name = f"Filtered_{self.comp[0]}-{self.comp[1]}_{filter_str}"

        filter_func = self.definition_to_function(filter_definition)
        res = genes.filter(new_name, filter_func, annotators=self)
        if ppg.inside_ppg():
            self.register_qc_volcano(genes, res, filter_func)
            self.register_qc_ma_plot(genes, res, filter_func)
        return res

    def definition_to_function(self, definition):
        functors = []
        for column_name, op, threshold in definition:
            if column_name in self.columns:
                pass
            elif column_name in self.column_lookup:
                column_name = self.column_lookup[column_name]
            else:
                raise KeyError(
                    f"unknown column {column_name}", "available", self.column_lookup
                )
            if op == "==":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name] == threshold
                )  # noqa: E03
            elif op == ">":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name] > threshold
                )  # noqa: E03
            elif op == "<":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name] < threshold
                )  # noqa: E03
            elif op == ">=":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name] >= threshold
                )  # noqa: E03
            elif op == "<=":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name] <= threshold
                )  # noqa: E03
            elif op == "|>":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name].abs()
                    > threshold  # noqa: E03
                )
            elif op == "|<":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name].abs()
                    < threshold
                )  # noqa: E03
            elif op == "|>=":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name].abs()
                    >= threshold
                )  # noqa: E03
            elif op == "|<=":
                f = (
                    lambda df, column_name=column_name, threshold=threshold: df[column_name].abs()
                    <= threshold
                )  # noqa: E03
            else:
                raise ValueError(f"invalid operator {op}")
            functors.append(f)

        def filter_func(df):
            keep = np.ones(len(df), bool)
            for f in functors:
                keep &= f(df)
            return keep

        return filter_func

    def calc(self, df):
        columns_a = self.sample_columns(self.comp[0])
        columns_b = self.sample_columns(self.comp[1])
        comp = self.comparison_strategy.compare(
            df, columns_a, columns_b, self.laplace_offset
        )
        res = {}
        for col in sorted(self.comparison_strategy.columns):
            res[self.name_column(col)] = comp[col]
        return pd.DataFrame(res)

    def dep_annos(self):
        """Return other annotators"""
        res = []
        for k in self.samples_used():
            a = parse_a_or_c_to_anno(k)
            if a is not None:
                res.append(a)
        return res

    def deps(self, ddf):
        from mbf_genomics.util import freeze

        sample_info = []
        for group, samples in self.groups_to_samples.items():
            for s in samples:
                sample_info.append(
                    (group, str(parse_a_or_c_to_anno(s)), parse_a_or_c_to_column(s))
                )
        sample_info.sort()

        parameters = freeze(
            [
                (
                    # self.comparison_strategy.__class__.__name__ , handled by column name
                    sample_info,
                    #   self.comp, # his handled by column name
                    self.laplace_offset,
                )
            ]
        )
        res = [
            ppg.ParameterInvariant(
                "mbf_comparison.Comparison_%i" % hash(parameters), ()
            )
        ]
        res.extend(getattr(self.comparison_strategy, "deps", lambda: [])())
        return res

    def samples_used(self):
        for x in self.comp:
            for s in self.groups_to_samples[x]:
                yield s

    def sample_columns(self, coldef):
        res = []
        for k in self.groups_to_samples[coldef]:
            res.append(parse_a_or_c_to_column(k))
        return res

    def plot_name(self, sample_column):
        for g, samples in self.groups_to_samples.items():
            for k in samples:
                if parse_a_or_c_to_column(k) == sample_column:
                    return parse_a_or_c_to_plot_name(k)

        raise KeyError(sample_column)  # pragma: no cover

    def _check_input_dict(self, groups_to_samples):
        if not isinstance(groups_to_samples, dict):
            raise ValueError("groups_to_samples must be a dict")
        counter = collections.Counter()
        for k, v in groups_to_samples.items():
            if not isinstance(k, str):
                raise ValueError("keys must be str, was %s %s" % (k, type(k)))
            v = list(v)
            for c in v:
                ok = True
                if isinstance(c, str):
                    counter[c] += 1
                elif isinstance(c, Annotator):
                    counter[c.get_cache_name()] += 1
                elif isinstance(c, tuple) and isinstance(c[0], Annotator):
                    if c[1] in c[0].columns:
                        counter[c[0].get_cache_name(), c[1]] += 1
                    elif isinstance(c[1], int) and c[1] < len(c[0].columns):
                        counter[c[0].get_cache_name(), c[0].columns[c[1]]] += 1
                    else:
                        ok = False
                else:
                    ok = False
                if not ok:
                    raise ValueError(
                        "groups_to_samples values must be str, annotator, "
                        "(annotator, columns) or "
                        "(annotator, column_number_in_annotator)"
                    )
            groups_to_samples[k] = v

        return groups_to_samples

    def _check_comparison_groups(self, a, b):
        for x in [a, b]:
            if x not in self.groups_to_samples:
                raise ValueError(f"Comparison group {x} not found")
            if (
                len(self.groups_to_samples[x])
                < self.comparison_strategy.min_sample_count
            ):
                raise ValueError(
                    "Too few samples in %s for %s" % (x, self.comparison_strategy)
                )

    def register_qc_volcano(self, genes, filtered, filter_func):
        """perform a volcano plot - not a straight annotator.register_qc function,
        but called by .filter
        """
        output_filename = filtered.result_dir / "volcano.png"

        def plot(output_filename):
            (
                dp(genes.df)
                .mutate(significant=filter_func(genes.df))
                .p9()
                .scale_color_many_categories(name="significant", shift=3)
                .scale_y_continuous(
                    name="p",
                    trans=dp.reverse_transform("log10"),
                    labels=lambda xs: ["%.2g" % x for x in xs],
                )
                .add_vline(xintercept=1, _color="blue")
                .add_vline(xintercept=-1, _color="blue")
                .add_hline(yintercept=0.05, _color="blue")
                .add_rect(  # shade 'simply' significant regions
                    xmin="xmin",
                    xmax="xmax",
                    ymin="ymin",
                    ymax="ymax",
                    _fill="lightgrey",
                    data=pd.DataFrame(
                        {
                            "xmin": [-np.inf, 1],
                            "xmax": [-1, np.inf],
                            "ymin": [0, 0],
                            "ymax": [0.05, 0.05],
                        }
                    ),
                    _alpha=0.8,
                )
                .add_scatter(self["log2FC"], self["p"], color="significant")
                # .coord_trans(x="reverse", y="reverse")  # broken as of 2019-01-31
                .render(output_filename, width=8, height=6, dpi=300)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(
                genes.add_annotator(self)
            )
        )

    def register_qc_ma_plot(self, genes, filtered, filter_func):
        """perform an MA plot - not a straight annotator.register_qc function,
        but called by .filter
        """
        output_filename = filtered.result_dir / "ma_plot.png"

        def plot(output_filename):
            from statsmodels.nonparametric.smoothers_lowess import lowess

            df = genes.df[
                self.sample_columns(self.comp[0]) + self.sample_columns(self.comp[1])
            ]
            df = df.assign(significant=filter_func(genes.df))
            pdf = []
            loes_pdfs = []
            # Todo: how many times can you over0lopt this?
            for a, b in itertools.combinations(
                [x for x in df.columns if not "significant" == x], 2
            ):
                np_a = np.log2(df[a] + self.laplace_offset)
                np_b = np.log2(df[b] + self.laplace_offset)
                A = (np_a + np_b) / 2
                M = np_a - np_b
                pdf.append(
                    pd.DataFrame(
                        {
                            "A": A,
                            "M": M,
                            "a": self.plot_name(a),
                            "b": self.plot_name(b),
                            "significant": df["significant"],
                        }
                    )
                )
                fitted = lowess(M, A, is_sorted=False)
                loes_pdfs.append(
                    pd.DataFrame(
                        {
                            "a": self.plot_name(a),
                            "b": self.plot_name(b),
                            "A": fitted[:, 0],
                            "M": fitted[:, 1],
                        }
                    )
                )
            pdf = pd.concat(pdf)
            pdf = pdf.assign(ab=[a + ":" + b for (a, b) in zip(pdf["a"], pdf["b"])])
            loes_pdf = pd.concat(loes_pdfs)
            loes_pdf = loes_pdf.assign(
                ab=[a + ":" + b for (a, b) in zip(loes_pdf["a"], loes_pdf["b"])]
            )
            (
                dp(pdf)
                .p9()
                .theme_bw(10)
                .add_hline(yintercept=0, _color="lightblue")
                .add_hline(yintercept=1, _color="lightblue")
                .add_hline(yintercept=-1, _color="lightblue")
                .scale_color_many_categories(name="significant", shift=3)
                .add_point("A", "M", color="significant", _size=1, _alpha=0.3)
                .add_line("A", "M", _color="blue", data=loes_pdf)
                .facet_wrap(["ab"])
                .title(f"MA {filtered.name}\n{a}")
                .render(output_filename, width=8, height=6)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(
                genes.add_annotator(self)
            )
        )
