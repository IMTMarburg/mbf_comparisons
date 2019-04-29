import itertools
import pypipegraph as ppg
import numpy as np
import pandas as pd
from mbf_qualitycontrol import register_qc, qc_disabled
from mbf_genomics.util import parse_a_or_c_to_anno
from mbf_genomics.annotator import Annotator
import dppd
import dppd_plotnine  # noqa: F401

dp, X = dppd.dppd()

# import pypipegraph as ppg


class ComparisonAnnotator(Annotator):
    def __init__(
        self, comparisons, group_a, group_b, comparison_strategy, laplace_offset=1 / 1e6
    ):
        """Create a comparison (a - b)

            """
        self.comparisons = comparisons

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
        self.comp = (group_a, group_b)
        self.columns = []
        self.column_lookup = {}
        for col in sorted(self.comparison_strategy.columns):
            cn = self.name_column(col)
            self.columns.append(cn)
            self.column_lookup[col] = cn
        self.laplace_offset = laplace_offset
        self.result_dir = self.comparisons.result_dir / f"{group_a}_vs_{group_b}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self._check_comparison_groups(group_a, group_b)

    def name_column(self, col):
        return f"Comp. {self.comp[0]} - {self.comp[1]} {col} ({self.comparison_strategy.name})"

    def __getitem__(self, itm):
        """look up the full column name from log2FC, p, FDR, etc"""
        return self.column_lookup[itm]

    def filter(self, filter_definition, new_name=None):
        """Turn a filter definition [(column, operator, threshold)...]
        into a filtered genes object.

        Example:
        comp.filter(genes, '2x', [
            ('FDR', '<=', 0.05) # a name from our comparison strategy - inspect column_lookup to list
            ('log2FC', '|>', 1),  #absolute
            ...
            (anno, '>=', 50),
            ((anno, 1), '>=', 50),  # for the second column of the annotator
            ((anno, 'columnX'), '>=', 50),  # for the second column of the annotator
            ('annotator_columnX', '=>' 50), # search for an annotator with that column. Use if exactly one, complain otherwise



            ]
        """
        if new_name is None:
            filter_str = []
            for column, op, threshold in sorted(filter_definition):
                filter_str.append(f"{column}_{op}_{threshold:.2f}")
            filter_str = "__".join(filter_str)
            new_name = f"Filtered_{self.comp[0]}-{self.comp[1]}_{filter_str}"

        lookup = self.column_lookup.copy()
        for c in self.columns:
            lookup[c] = c

        # we need the filter func for the plotting
        filter_func, annos = self.comparisons.ddf.definition_to_function(
            filter_definition, lookup
        )
        res = self.comparisons.ddf.filter(
            new_name,
            filter_func,
            annotators=annos,
            column_lookup=lookup,
            result_dir=self.result_dir / new_name,
        )
        if not qc_disabled():
            self.register_qc_volcano(self.comparisons.ddf, res, filter_func)
            self.register_qc_ma_plot(self.comparisons.ddf, res, filter_func)
        res.plot_columns = self.samples()
        return res

    def calc(self, df):
        columns_a = list(self.sample_columns(self.comp[0]))
        columns_b = list(self.sample_columns(self.comp[1]))
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
        for k in self.samples():
            a = parse_a_or_c_to_anno(k)
            if a is not None:
                res.append(a)
        return res

    def deps(self, ddf):
        from mbf_genomics.util import freeze

        sample_info = []
        for ac in self.samples():
            group = self.comparisons.sample_column_to_group[ac[1]]
            sample_info.append(
                (group, ac[0].get_cache_name() if ac[0] is not None else "None", ac[1])
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
        res = [ppg.ParameterInvariant(self.get_cache_name(), parameters)]
        res.extend(getattr(self.comparison_strategy, "deps", lambda: [])())
        return res

    def samples(self):
        """Return anno, column for samples used"""
        for x in self.comp:
            for s in self.comparisons.groups_to_samples[x]:
                yield s

    def sample_columns(self, group):
        for s in self.comparisons.groups_to_samples[group]:
            yield s[1]

    def _check_comparison_groups(self, a, b):
        for x in [a, b]:
            if x not in self.comparisons.groups_to_samples:
                raise ValueError(f"Comparison group {x} not found")
            if (
                len(self.comparisons.groups_to_samples[x])
                < self.comparison_strategy.min_sample_count
            ):
                raise ValueError(
                    "Too few samples in %s for %s" % (x, self.comparison_strategy)
                )

    def register_qc_volcano(self, genes, filtered=None, filter_func=None):
        """perform a volcano plot
        """
        if filtered is None:
            output_filename = genes.result_dir / "volcano.png"
        else:
            output_filename = filtered.result_dir / "volcano.png"

        def plot(output_filename):
            (
                dp(genes.df)
                .mutate(
                    significant=filter_func(genes.df)
                    if filter_func is not None
                    else "tbd."
                )
                .p9()
                .scale_color_many_categories(name="regulated", shift=3)
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
                list(self.sample_columns(self.comp[0]))
                + list(self.sample_columns(self.comp[1]))
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
                local_pdf = pd.DataFrame(
                    {
                        "A": A,
                        "M": M,
                        "a": self.comparisons.get_plot_name(a),
                        "b": self.comparisons.get_plot_name(b),
                        "significant": df["significant"],
                    }
                ).sort_values("M")
                chosen = np.zeros(len(local_pdf), bool)
                chosen[:500] = True
                chosen[-500:] = True
                chosen[np.random.randint(0, len(chosen), 1000)] = True
                pdf.append(local_pdf)
                fitted = lowess(M, A, is_sorted=False)
                loes_pdfs.append(
                    pd.DataFrame(
                        {
                            "a": self.comparisons.get_plot_name(a),
                            "b": self.comparisons.get_plot_name(b),
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
                .title(f"MA {filtered.name}\n{self.comparisons.find_variable_name()}")
                .render(output_filename, width=8, height=6)
            )

        return register_qc(
            ppg.FileGeneratingJob(output_filename, plot).depends_on(
                genes.add_annotator(self)
            )
        )
