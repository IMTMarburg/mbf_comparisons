from mbf_genomics.annotator import Annotator
import numpy as np
import pandas as pd
import collections

# import pypipegraph as ppg


class Comparison(Annotator):
    def __init__(self, comparison_strategy, groups_to_samples, a, b):
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

    def name_column(self, col):
        return f"Comp. {self.comp[0]} - {self.comp[1]} {col} ({self.comparison_strategy.name})"

    def __getitem__(self, itm):
        """look up the full column name from log2FC, p, FDR, etc"""
        return self.column_lookup[itm]

    def filter(self, genes, new_name, filter_definition):
        """Turn a filter definition [(column, operator, threshold)...]
        into a filtered genes object.

        Example:
        comp.filter(genes, '2x', [
            (log2FC, '|>', 1),  #absolute
            ('FDR', '<=', 0.05)
            ]
        """
        return genes.filter(
            new_name, self.definition_to_function(filter_definition), annotators=self
        )

    def definition_to_function(self, definition):
        functors = []
        for column_name, op, threshold in definition:
            if column_name in self.columns:
                pass
            elif column_name in self.column_lookup:
                column_name = self.column_lookup[column_name]
            else:
                raise ValueError(f"unknown column {column_name}", 'available', self.column_lookup)
            if op == "==":
                f = (
                    lambda df, column_name=column_name: df[column_name] == threshold
                )  # noqa: E03
            elif op == ">":
                f = (
                    lambda df, column_name=column_name: df[column_name] > threshold
                )  # noqa: E03
            elif op == "<":
                f = (
                    lambda df, column_name=column_name: df[column_name] < threshold
                )  # noqa: E03
            elif op == ">=":
                f = (
                    lambda df, column_name=column_name: df[column_name] >= threshold
                )  # noqa: E03
            elif op == "<=":
                f = (
                    lambda df, column_name=column_name: df[column_name] <= threshold
                )  # noqa: E03
            elif op == "|>":
                f = (
                    lambda df, column_name=column_name: df[column_name].abs()
                    > threshold  # noqa: E03
                )
            elif op == "|<":
                f = (
                    lambda df, column_name=column_name: df[column_name].abs()
                    < threshold
                )  # noqa: E03
            elif op == "|>=":
                f = (
                    lambda df, column_name=column_name: df[column_name].abs()
                    >= threshold
                )  # noqa: E03
            elif op == "|<=":
                f = (
                    lambda df, column_name=column_name: df[column_name].abs()
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
        comp = self.comparison_strategy.compare(df, columns_a, columns_b)
        res = {}
        for col in sorted(self.comparison_strategy.columns):
            res[self.name_column(col)] = comp[col]
        return pd.DataFrame(res)

    def dep_annos(self):
        """Return other annotators"""
        res = []
        for c in self.samples_used():
            if isinstance(c, Annotator):
                res.append(c)
            if isinstance(c, tuple) and isinstance(c[0], Annotator):
                res.append(c[0])
        return res

    def samples_used(self):
        for x in self.comp:
            for s in self.groups_to_samples[x]:
                yield s

    def sample_columns(self, coldef):
        res = []
        for k in self.groups_to_samples[coldef]:
            if isinstance(k, str):
                res.append(k)
            elif isinstance(k, Annotator):
                res.append(k.columns[0])
            elif isinstance(k, tuple):
                if isinstance(k[1], int):
                    res.append(k[0].columns[k[1]])
                else:
                    res.append(k[1])
            else:  # pragma: no cover - should have been handled by _check_input_dict
                raise NotImplementedError(
                    "Sample_columns encountered a case that should have been covered by _check_input_dict"
                )
        return res

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
