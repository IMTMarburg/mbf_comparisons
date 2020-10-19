from typing import List, Tuple, Dict, Callable
from pandas import DataFrame
from pypipegraph import Job
from statsmodels.stats.multitest import multipletests
from mbf_genomics.genes.anno_tag_counts import IntervalStrategyGene
from mbf_genomes import GenomeBase
import pypipegraph as ppg
import pandas as pd
import scipy.stats as ss
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
import mbf_r
import re


class Log2FC:
    min_sample_count = 0
    supports_other_samples = False

    def __init__(self):
        self.columns = ["log2FC", "minExpression"]
        self.name = "simple"

    def compare(self, df, columns_a, columns_b, columns_other, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        min_expression = df[columns_a + columns_b].min(axis=1)
        return pd.DataFrame({"log2FC": logFC, "minExpression": min_expression})


class TTest:
    """Standard students t-test, independent on log2FC + benjamini hochberg"""

    min_sample_count = 3
    supports_other_samples = False

    def __init__(self, equal_variance=False):
        self.equal_var = equal_variance
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest"

    def compare(self, df, columns_a, columns_b, columns_other, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_ind(a, b, axis=1, equal_var=self.equal_var, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})


class TTestPaired:
    """Standard students t-test, paired, on log2FC + benjamini hochberg"""

    min_sample_count = 3
    supports_other_samples = False

    def __init__(self):
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest_paired"

    def compare(self, df, columns_a, columns_b, columns_other, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_rel(a, b, axis=1, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})


class EdgeRUnpaired:

    min_sample_count = 3
    name = "edgeRUnpaired"
    columns = ["log2FC", "p", "FDR"]
    supports_other_samples = False

    def __init__(self, ignore_if_max_count_less_than=None, manual_dispersion_value=0.4):
        self.ignore_if_max_count_less_than = ignore_if_max_count_less_than
        self.manual_dispersion_value = manual_dispersion_value

    def deps(self):
        import rpy2.robjects as ro

        ro.r("library('edgeR')")
        version = str(ro.r("packageVersion")("edgeR"))
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name,
            (version, self.ignore_if_max_count_less_than),
        )

    def edgeR_comparison(
        self, df, columns_a, columns_b, library_sizes=None, manual_dispersion_value=0.4
    ):
        """Call edgeR exactTest comparing two groups.
        Resulting dataframe is in df order.
        """
        import mbf_r
        import math
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri

        ro.r("library(edgeR)")
        input_df = df[columns_a + columns_b]
        input_df.columns = ["X_%i" % x for x in range(len(input_df.columns))]
        if library_sizes is not None:  # pragma: no cover
            samples = pd.DataFrame({"lib.size": library_sizes})
        else:
            samples = pd.DataFrame({"lib.size": input_df.sum(axis=0)})
        # this looks like it inverts the columns,
        # but it doesnt'
        samples.insert(0, "group", ["z"] * len(columns_b) + ["x"] * len(columns_a))
        r_counts = mbf_r.convert_dataframe_to_r(input_df)
        r_samples = mbf_r.convert_dataframe_to_r(samples)
        y = ro.r("DGEList")(
            counts=r_counts,
            samples=r_samples,
            **{
                "lib.size": ro.r("as.vector")(
                    numpy2ri.py2rpy(np.array(samples["lib.size"]))
                )
            },
        )
        # apply TMM normalization
        y = ro.r("calcNormFactors")(y)
        if len(columns_a) == 1 and len(columns_b) == 1:  # pragma: no cover
            # not currently used.
            z = manual_dispersion_value
            e = ro.r("exactTest")(y, dispersion=math.pow(manual_dispersion_value, 2))
            """
            you are attempting to estimate dispersions without any replicates.
            Since this is not possible, there are several inferior workarounds to come up with something
            still semi-useful.
            1. pick a reasonable dispersion value from "Experience": 0.4 for humans, 0.1 for genetically identical model organisms, 0.01 for technical replicates. We'll try this for now.
            2. estimate dispersions on a number of genes that you KNOW to be not differentially expressed.
            3. In case of multiple factor experiments, discard the least important factors and treat the samples as replicates.
            4. just use logFC and forget about significance.
            """
        else:
            z = ro.r("estimateDisp")(y, robust=True)
            e = ro.r("exactTest")(z)
        res = ro.r("topTags")(e, n=len(input_df), **{"sort.by": "none"})
        result = mbf_r.convert_dataframe_from_r(res[0])
        return result

    def compare(self, df, columns_a, columns_b, columns_other, _laplace_offset):
        # laplace offset is ignored, edgeR works on raw data
        value_columns = columns_a + columns_b
        # we need to go by key, since filter out nan rows.
        idx = ["G%i" % ii for ii in range(len(df))]
        input_df = df[value_columns]
        input_df = input_df.assign(idx=idx)
        input_df = input_df.set_index("idx")
        if pd.isnull(input_df).any().any():  # pragma: no cover
            raise ValueError("Nans before filtering in edgeR input")

        if self.ignore_if_max_count_less_than is not None:
            max_raw_count_per_gene = input_df.max(axis=1)
            input_df.loc[
                max_raw_count_per_gene < self.ignore_if_max_count_less_than, :
            ] = np.nan
        # does not matter any or all since we set them all above.
        input_df = input_df[~pd.isnull(input_df[value_columns]).all(axis=1)]

        differential = self.edgeR_comparison(
            input_df,
            columns_a,
            columns_b,
            manual_dispersion_value=self.manual_dispersion_value,
        )
        result = {"FDR": [], "p": [], "log2FC": []}
        for key in idx:
            try:
                row = differential.loc[key]
                result["FDR"].append(row["FDR"])
                result["p"].append(row["PValue"])
                result["log2FC"].append(row["logFC"])
            except KeyError:
                result["FDR"].append(np.nan)
                result["p"].append(np.nan)
                result["log2FC"].append(np.nan)
        return pd.DataFrame(result)


class EdgeRPaired(EdgeRUnpaired):

    min_sample_count = 3
    name = "edgeRPaired"
    columns = ["log2FC", "p", "FDR"]
    supports_other_samples = False

    def __init__(self, ignore_if_max_count_less_than=None, manual_dispersion_value=0.4):
        self.ignore_if_max_count_less_than = ignore_if_max_count_less_than
        self.manual_dispersion_value = manual_dispersion_value

    def edgeR_comparison(
        self, df, columns_a, columns_b, library_sizes=None, manual_dispersion_value=0.4
    ):
        """Call edgeR exactTest comparing two groups.
        Resulting dataframe is in df order.
        """
        import mbf_r
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri

        if len(columns_a) != len(columns_b):
            raise ValueError("paired requires equal length groups")

        ro.r("library(edgeR)")
        input_df = df[columns_a + columns_b]
        input_df.columns = ["X_%i" % x for x in range(len(input_df.columns))]
        if library_sizes is not None:  # pragma: no cover
            samples = pd.DataFrame({"lib.size": library_sizes})
        else:
            samples = pd.DataFrame({"lib.size": input_df.sum(axis=0)})
        # remember, edgeR does b-a not a-b...
        samples.insert(0, "group", ["z"] * len(columns_b) + ["y"] * len(columns_a))
        samples.insert(
            1,
            "pairs",
            [str(x) for x in list(range(len(columns_a))) + list(range(len(columns_a)))],
        )

        r_counts = mbf_r.convert_dataframe_to_r(input_df)
        r_samples = mbf_r.convert_dataframe_to_r(samples)
        design = ro.r("model.matrix")(ro.r("~pairs+group"), data=r_samples)
        y = ro.r("DGEList")(
            counts=r_counts,
            samples=r_samples,
            **{
                "lib.size": ro.r("as.vector")(
                    numpy2ri.py2rpy(np.array(samples["lib.size"]))
                )
            },
        )
        # apply TMM normalization
        y = ro.r("calcNormFactors")(y)
        z = ro.r("estimateDisp")(y, design, robust=True)
        fit = ro.r("glmFit")(z, design)
        lrt = ro.r("glmLRT")(fit)
        res = ro.r("topTags")(lrt, n=len(input_df), **{"sort.by": "none"})
        result = mbf_r.convert_dataframe_from_r(res[0])
        return result


class DESeq2Unpaired:
    min_sample_count = 3
    name = "DESeq2unpaired"
    columns = ["log2FC", "p", "FDR"]
    supports_other_samples = True

    def deps(self):
        import rpy2.robjects as ro

        ro.r("library('DESeq2')")
        version = str(ro.r("packageVersion")("DESeq2"))
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name, (version,)
        )

    def compare(self, df, columns_a, columns_b, columns_other, _laplace_offset):
        # laplace_offset is ignored
        import rpy2.robjects as robjects

        robjects.r('library("DESeq2")')
        columns = []
        conditions = []
        samples = []
        name_cols = [
            ("c", columns_a),
            ("base", columns_b),
        ]
        for g in columns_other:
            name_cols.append(
                (
                    "other_" + g.replace("-", "m").replace("+", "p").replace("_", ""),
                    columns_other[g],
                )
            )
        for (name, cols) in name_cols:
            for col in cols:
                columns.append(col)
                conditions.append(name)
                samples.append(col)
        count_data = df[columns]
        df = self.call_DESeq2(count_data, samples, conditions)
        df = df.rename(
            columns={"log2FoldChange": "log2FC", "pvalue": "p", "padj": "FDR"}
        )
        return df[self.columns].reset_index(drop=True)

    def call_DESeq2(self, count_data, samples, conditions):
        """Call DESeq2.
        @count_data is a DataFrame with 'samples' as the column names.
        @samples is a list. @conditions as well. Condition is the one you're contrasting on.
        You can add additional_conditions (a DataFrame, index = samples) which DESeq2 will
        keep under consideration (changes the formula).
        """
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        import mbf_r

        count_data = count_data.values
        count_data = np.array(count_data)
        nr, nc = count_data.shape
        count_data = count_data.reshape(count_data.size)  # turn into 1d vector
        count_data = robjects.r.matrix(
            numpy2ri.py2rpy(count_data), nrow=nr, ncol=nc, byrow=True
        )
        col_data = pd.DataFrame({"sample": samples, "condition": conditions}).set_index(
            "sample"
        )
        formula = "~ condition"
        col_data = col_data.reset_index(drop=True)
        col_data = mbf_r.convert_dataframe_to_r(pd.DataFrame(col_data.to_dict("list")))
        deseq_experiment = robjects.r("DESeqDataSetFromMatrix")(
            countData=count_data, colData=col_data, design=robjects.Formula(formula)
        )
        deseq_experiment = robjects.r("DESeq")(deseq_experiment)
        res = robjects.r("results")(
            deseq_experiment, contrast=robjects.r("c")("condition", "c", "base")
        )
        df = mbf_r.convert_dataframe_from_r(robjects.r("as.data.frame")(res))
        return df


class DESeq2MultiFactor:
    def __init__(self):
        self.min_sample_count = 3
        self.name = "DESeq2unpairedMulti"
        self.supports_other_samples = True
        self.columns = ["log2FC", "p", "FDR", "mean", "lfcSE"]
        pattern0 = re.compile(
            "(?P<main>[^:]*):(?P<main_ref>[^:()]*)\((?P<main_factor>.*)\) effect \(controlling for .*"  # noqa
        )
        pattern1 = re.compile(
            "(?P<main>[^:]*):(?P<main_ref>[^:()]*)\((?P<main_factor>.*)\) effect for (?P<other1>[^:]*)\((?P<other_factor>.*)\)"  # noqa
        )
        pattern2 = re.compile(
            "(?P<main>[^:]*):(?P<main_ref>[^:()]*)\((?P<main_factor>.*)\) effect for (?P<other1>[^:]*):(?P<other2>[^:()]*)\((?P<other_factor>.*)\)"  # noqa
        )
        pattern3 = re.compile(
            "(?P<main>[^:]*):(?P<main_ref>[^:()]*)\((?P<main_factor>.*)\) effect difference for (?P<other1>[^:]*):(?P<other2>[^:()]*)\((?P<other_factor>.*)\)"  # noqa
        )
        self.patterns = [pattern0, pattern1, pattern2, pattern3]

    def select_contrast_c(self, factor: str, level: str, reference: str) -> Callable:
        """
        Returns a callable to select results from a multi-factor deseq
        experiment using the contrast keyword.

        Parameters
        ----------
        main_factor : str
            The effect factor to select for.
        level : str
            The level to be compared to the factor reference.
        reference : str
            The reference level.
        Returns
        -------
        Callable
            A selection function that takes an robject instance from a deseq
            experiment.
        """

        def __select(dds):
            return robjects.r("results")(
                dds, contrast=robjects.r("c")(factor, level, reference)
            )

        return __select

    def select_contrast_list(self, selection: str, interaction: str) -> Callable:
        """
        Returns a callable to select results from a multi-cator deseq
        experiment using the contrast keyword with to selection terms.

        Parameters
        ----------
        selection : str
            The effect selection term, e.g. "condition_B_vs_A".
        interaction : str
            The interaction term to select for, e.g. "genotypeIII.conditionB".

        Returns
        -------
        Callable
            A selection function that takes an robject instance from a deseq
            experiment.
        """

        def __select(dds):
            return robjects.r("results")(
                dds,
                contrast=robjects.r("list")(robjects.r("c")(selection, interaction)),
            )

        return __select

    def select_name(self, interaction: str):
        """
        Returns a callable to select results from a multi-cator deseq
        experiment using the name keyword.

        Parameters
        ----------
        interaction : str
            Term to select for.

        Returns
        -------
        Callable
            A selection function that takes an robject instance from a deseq
            experiment.
        """

        def __select(dds):
            return robjects.r("results")(dds, name=interaction)

        return __select

    def get_selector(self, prefix: str, factor_reference: Dict[str, str],) -> Callable:
        """
        Returns the appropriate select function for a given prefix.

        This matches the prefix to one of four possible re patterns, extracts
        relevant factors and levels and generates the appropriate parameters
        for one of the thre selection function generators defined in this class.

        Parameters
        ----------
        prefix : str
            The column prefix, specifying a specific aspect of the deseq analysis,
            e.g. selecting the main condition effect for a specific genotype.
        factor_reference : Dict[str, str]
            Dictionary of factor names (key) to base level (value), e.g.
            {"treatment": "DMSO"}.

        Returns
        -------
        Callable
            The appropriate selection function for a given prefix that takes an
            robject instance from a deseq experiment.

        Raises
        ------
        ValueError
            If a prefix matches to multiple patterns.
        ValueError
            If a prefix does not match to any pattern.
        """
        found = -1
        selector = None
        for i, pattern in enumerate(self.patterns):
            match = pattern.match(prefix)
            if match:
                if found != -1:
                    raise ValueError(
                        f"Prefix {prefix} matched to multiple patterns (pattern{found}, pattern{i})."
                    )
                main = match.group("main")
                main_ref = match.group("main_ref")
                main_factor = match.group("main_factor")
                if i == 0:
                    # not interactions
                    selector = self.select_contrast_c(
                        main_factor, f"z{main}", f"b{main_ref}"
                    )
                elif i == 1:
                    # main effect with interaction versus other reference
                    selector = self.select_contrast_c(
                        main_factor, f"z{main}", f"b{main_ref}"
                    )
                elif i == 2:
                    other_factor = match.group("other_factor")
                    other1 = match.group("other1")
                    selection = f"{main_factor}_z{main}_vs_b{main_ref}"
                    interaction = f"{other_factor}z{other1}.{main_factor}z{main}"
                    selector = self.select_contrast_list(selection, interaction)
                elif i == 3:
                    other_factor = match.group("other_factor")
                    other1 = match.group("other1")
                    other2 = match.group("other2")
                    interaction = f"{other_factor}z{other1}.{main_factor}z{main}"
                    if other2 == factor_reference[other_factor]:
                        selector = self.select_name(interaction)
                    else:
                        interaction2 = f"{other_factor}z{other2}.{main_factor}z{main}"
                        selector = self.select_contrast_list(interaction, interaction2)
                found = i
        if selector is None:
            raise ValueError(f"prefix {prefix} did not match to any pattern.")
        return selector

    def prefixes(
        self,
        main_factor: str,
        factor_reference: Dict[str, str],
        df_factors: DataFrame,
        interactions: List[Tuple[str, str]],
        test_difference: bool,
        compare_non_reference: bool,
    ) -> List[str]:
        """
        Generates prefixes for each aspect to be selected from the multi-factor
        deseq run.

        This generates list of prefixes that define a certain aspect of interest
        to be obtained from the deseq run. These are used to select different
        aspects from the DEseq result and as prefixes for columns that are
        reported for each such aspect.

        Parameters
        ----------
        main_factor : str
            The main factor, e.g. treatment/condition.
        factor_reference : Dict[str, str]
            Dictionary of factor names (key) to base level (value), e.g.
            {"treatment": "DMSO"}.
        df_factors : DataFrame
            A DataFrame containing all factors and levels of the experiment.
            It should contain a column group as well as a column for each factor
            which is the column name.
        interactions : List[Tuple[str, str]]
            List if interaction terms. If this is empty, the analysis will
            report the main factor effects controlling for the other factors.
        test_difference : bool, optional
            Test for differences in the main effects for different levels
            of other factors, by default True.
        compare_non_reference : bool, optional
            Test for difference of the main effects for different levels of other
            factors compared to non-reference levels, by default False.

        Returns
        -------
        List[str]
            List of column prefixes.
        """
        prefixes = []
        main_reference = factor_reference[main_factor]
        main_levels = [
            level for level in df_factors[main_factor] if level != main_reference
        ]
        other_factors = [factor for factor in factor_reference if factor != main_factor]
        if len(interactions) == 0:
            # without interaction, DEseq just controls for the other factors
            for main_level in main_levels:
                prefix = f"{main_level}:{main_reference}({main_factor}) effect (controlling for {other_factors})"
                # N3a:untr(treatment) effect (controlling for ['genotype'])
                prefixes.append(prefix)
        else:
            for main_level in main_levels:
                common = f"{main_level}:{main_reference}({main_factor})"
                for factor in other_factors:
                    reference = factor_reference[factor]
                    # test for the main factor effect (e.g. treatment effect B vs A) for
                    # the reference level of other factor, (e.g. genotype I).
                    # results(dds, contrast=c("condition","B","A"))
                    prefix = f"{common} effect for {reference}({factor})"
                    # N3a:untr(treatment) effect for LSL(genotype)
                    prefixes.append(prefix)
                    levels = [
                        level for level in df_factors[factor] if level != reference
                    ]
                    for i1, level in enumerate(levels):
                        # test for the main factor effect (e.g. treatment effect B vs A) for
                        # another level of other factor (e.g. genotype III). It is the
                        # main effect plus interaction
                        # results(dds, contrast=list( c("condition_B_vs_A","genotypeIII.conditionB") ))
                        prefix = f"{common} effect for {level}:{reference}({factor})"
                        # N3a:untr(treatment) effect for WT:LSL(genotype)
                        prefixes.append(prefix)
                        if test_difference:
                            # test if the main factor effect is different
                            # for the level of other factor compared to the reference level.
                            # results(dds, name="genotypeIII.conditionB")
                            prefix = f"{common} effect difference for {level}:{reference}({factor})"
                            # N3a:untr(treatment) effect difference for WT:LSL(genotype)
                            prefixes.append(prefix)
                            if compare_non_reference:
                                # test if the main factor effect is different
                                # for the level of other factor compared to a non-reference level of factor.
                                # results(dds, contrast=list("genotypeIII.conditionB", "genotypeII.conditionB"))
                                for i2, level2 in enumerate(levels):
                                    if i2 <= i1:
                                        continue
                                    prefix = f"{common} effect difference for {level}:{level2}({factor})"
                                    # N3a:untr(treatment) effect difference for WT:KO(genotype)
                                    prefixes.append(prefix)
        return prefixes

    def compare(
        self,
        df: DataFrame,
        main_factor: str,
        factor_reference: Dict[str, str],
        columns_by_group: Dict[str, List],
        df_factors: DataFrame,
        interactions: List[Tuple[str, str]],
        test_difference: bool,
        compare_non_reference: bool,
        laplace_offset: float,
        prefixes: List[str],
    ) -> DataFrame:
        """
        Returns a dataframe containing a multi-factor analysis.

        This implements the compare method which is called from the
        annotator. This method prepares the input for DEseq2 and calls the
        DEseq2 subsequently.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing the raw counts.
        main_factor : str
            The main effect variable.
        factor_reference : Dict[str, str]
            Dictionary of factor names (key) to base level (value), e.g.
            {"treatment": "DMSO"}.
        columns_by_group : Dict[str, List]
            A dictionary containg groups as key and the raw count column names
            as list as values.
        df_factors : DataFrame
            A DataFrame containing all factors and levels of the experiment.
            It should contain a column group as well as a column for each factor
            which is the column name.
        interactions : List[Tuple[str, str]]
            List if interaction terms. If this is empty, the analysis will
            report the main factor effects controlling for the other factors.
        test_difference : bool, optional
            Test for differences in the main effects for different levels
            of other factors, by default True.
        compare_non_reference : bool, optional
            Test for difference of the main effects for different levels of other
            factors compared to non-reference levels, by default False.
        laplace_offset : float, optional
            laplace offset for methods that cannot handle zeros, this is ignored.
        prefixes : List[str]
            List of column prefixes.

        Returns
        -------
        DataFrame
            A result DataFrame returned from DEseq2 call.
        """
        import rpy2.robjects as robjects

        robjects.r('library("DESeq2")')
        columns = []
        to_df: List[pd.Series] = []
        prefix_select = {}
        for prefix in prefixes:
            select = self.get_selector(prefix, factor_reference)
            prefix_select[prefix] = select

        # make sure the reference level is lexicographically first
        other_factors = []
        for factor in factor_reference:
            ref_level = factor_reference[factor]
            replace = [
                f"z{level}" if level != ref_level else f"b{level}"
                for level in df_factors[factor]
            ]
            df_factors.replace(df_factors[factor].values, replace, inplace=True)
            if factor != main_factor:
                other_factors.append(factor)
        for _, row in df_factors.iterrows():
            group = row["group"]
            # name = trim_group_name(row["group"])
            for col in columns_by_group[group]:
                columns.append(col)
                new_row = row.copy()
                new_row["sample"] = col
                to_df.append(new_row)
        count_data = df[columns]
        column_data = pd.DataFrame(to_df)
        column_data = column_data.set_index("sample")
        formula = "~ " + " + ".join(other_factors)
        formula += f" + {main_factor}"
        for f1, f2 in interactions:
            formula += f" + {f1}:{f2}"
        df = self.call_DESeq2(count_data, column_data, formula, prefix_select,)
        df = df.reset_index(drop=True)
        return df

    def call_DESeq2(
        self,
        count_data: DataFrame,
        column_data: DataFrame,
        formula: str,
        prefix_select: Dict[str, Callable],
    ) -> DataFrame:
        """
        Returns a dataframe containing a multi-factor analysis.

        This actually calls DEseq2 via robjects, prepares the formula and
        joins the result dataframes.

        Parameters
        ----------
        count_data : DataFrame
            DataFrame with raw counts.
        column_data : DataFrame
            DataFrame with factor data.
        formula : str
            The formula to use for the DEseq analysis.
        prefix_select : Dict[str, Callable]
            A dictionary prefix to appropriate selector function.

        Returns
        -------
        DataFrame
            A result DataFrame which is annotated to the DelayedDataFrame.
        """

        def res_to_df(res, prefix):
            rename = {
                "log2FoldChange": "log2FC",
                "pvalue": "p",
                "padj": "FDR",
                "baseMean": "mean",
            }
            df = mbf_r.convert_dataframe_from_r(robjects.r("as.data.frame")(res))
            df = df[["baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]]
            df = df.rename(columns=rename)
            df = df.rename(columns=dict([(col, f"{prefix} {col}") for col in df]))
            return df

        count_data = count_data.values
        count_data = np.array(count_data)
        nr, nc = count_data.shape
        count_data = count_data.reshape(count_data.size)  # turn into 1d vector
        count_data = robjects.r.matrix(
            numpy2ri.py2rpy(count_data), nrow=nr, ncol=nc, byrow=True
        )
        # col_data = col_data.reset_index(drop=True)
        col_data = mbf_r.convert_dataframe_to_r(
            pd.DataFrame(column_data.to_dict("list"))
        )
        deseq_experiment = robjects.r("DESeqDataSetFromMatrix")(
            countData=count_data, colData=col_data, design=robjects.Formula(formula)
        )
        dds = robjects.r("DESeq")(deseq_experiment)
        # all that is left is extracting the results.
        dfs_to_concat = []
        for prefix in prefix_select:
            select = prefix_select[prefix]
            res = select(dds)
            dfs_to_concat.append(res_to_df(res, prefix))
        df = dfs_to_concat[0]
        for df2 in dfs_to_concat[1:]:
            df = df.join(df2)
        return df

    def get_columns(self, prefixes: List[str],) -> List[str]:
        """
        Returns a list of all columns generated by the compare method.

        This is used in the ComparisonAnnotatorMulti class to declare the
        generated columns in advance.

        Parameters
        ----------
        prefixes : List[str]
            List of column prefixes as generated by self.prefixes.

        Returns
        -------
        List[str]
            List of columns.
        """
        columns = []
        for col in self.columns:
            for prefix in prefixes:
                columns.append(f"{prefix} {col}")
        return sorted(columns)


class NOISeq:
    """
    NoiseSeq comparison strategy that returns a probability measure for
    genes being differentially expressed. Use this if you have only single samples
    or technical replicates.

    This uses the R package NOISeq and calculates log2FC as well as difference
    in in mean expression values after performing an appropriate normalization step.
    Then it it simulates noise and estimates probabilities that the obtained
    values are due to conditions. This can be used as a ranking score for
    DEG, not as a p-value.
    This can only do pairwise comparisons on a single factor.

    Parameters
    ----------
    norm : str, optional
        The normalization to be applied, by default "tmm".
    nss : int, optional
        Number of simulated noise samples, by default 5.
    lc : int, optional
        Whether length normalization is performed (rpkm), by default 0 (ie. off).
    v : float, optional
        The variance for the simulated noise, by default 0.02.
    pnr : float, optional
        Percentage of the total reads used to simulated each sample when no replicates
        are available, by default 0.2.
    replicates: 'technical', 'biological' or 'no', default 'technical'

    Raises
    ------
    ValueError
        If an unknown normalization method is given.
    """

    min_sample_count = 1
    name = "NOIseq"
    supports_other_samples = True
    columns = ["log2FC", "Prob", "Rank", "D"]

    def __init__(
        self,
        norm: str = "tmm",
        nss: int = 5,
        lc: int = 0,
        v: float = 0.02,
        pnr: float = 0.2,
        interval_strategy: IntervalStrategyGene = None,
        genome: GenomeBase = None,
        replicates: str = "technical",
    ):
        """Constructor"""
        self.norm = norm
        accepted = ["tmm", "rpkm", "uqua", "n"]
        if self.norm not in accepted:
            raise ValueError(
                f"Only {accepted} are accepted as values for norm, given was {norm}"
            )
        if self.norm == "rpkm":
            if interval_strategy is None or genome is None:
                raise ValueError(
                    "If you choose 'rpkm' as norm, you need to supply an IntervalStrategy and a genome."
                )
        self.columns = ["log2FC", "Prob", "Rank", "D"]
        self.interval_strategy = interval_strategy
        self.lc = lc
        self.v = v
        self.nss = nss
        self.pnr = pnr
        self.genome = genome
        self.replicates = replicates

    def compare(
        self,
        df: DataFrame,
        columns_a: List[str],
        columns_b: List[str],
        columns_other: Dict[str, List[str]],
        _laplace_offset: float = 0.5,
    ) -> DataFrame:
        """
        Performas the comparison a vs b by preparing the NOISeq inputs and
        calling NOISeq.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be annotated.
        columns_a : List[str]
            Column names for condition a.
        columns_b : List[str]
            Column names for condition b.
        columns_other : Dict[str, str]
            Dictionary group to list of column names for other groups.
        _laplace_offset : float, optional
            offset to be added to prevent zero division (k for NOIseq), by default 0.5.

        Returns
        -------
        DataFrame
            DataFrame with Log2FC, prob and rank columns.
        """
        robjects.r('library("NOISeq")')
        columns = []
        condition = []
        name_cols = [
            ("a", columns_a),
            ("base", columns_b),
        ]
        for i, g in enumerate(columns_other):
            name_cols.append((f"other_{i}", columns_other[g],))
        factor = "condition"
        rename = {}
        for (name, cols) in name_cols:
            for i, col in enumerate(cols):
                rename[col] = f"{name}_{i}"
                columns.append(col)
                condition.append(name)
        count_data = df[columns + ["gene_stable_id"]].rename(columns=rename)
        count_data = count_data.set_index("gene_stable_id")
        factors = pd.DataFrame({factor: condition}, index=count_data.columns)
        df_chrom = df[["chr", "start", "stop"]]
        df_chrom.index = df["gene_stable_id"]
        biotypes = df["biotype"].values
        if "length" not in df.columns:
            if self.interval_strategy is not None:
                lengths = self.interval_strategy.get_interval_lengths_by_gene(
                    self.genome
                )
            else:
                lengths = df["stop"] - df["start"]
        else:
            lengths = df["length"].values
        df = self.call_noiseq(
            count_data, factors, biotypes, lengths, df_chrom, _laplace_offset,
        )
        df = df.rename(columns={"M": "log2FC", "prob": "Prob", "ranking": "Rank"})
        return df[self.columns].reset_index(drop=True)

    def deps(self) -> List[Job]:
        """Returns a list of job dependencies."""

        robjects.r("library('NOISeq')")
        version = str(robjects.r("packageVersion")("NOISeq"))
        return [
            ppg.ParameterInvariant(
                self.__class__.__name__ + "_" + self.name,
                (version, self.lc, self.nss, self.norm, self.v, self.columns),
            ),
            ppg.FunctionInvariant(f"FI_{self.name}_compare", self.compare),
            ppg.FunctionInvariant(f"FI_{self.name}_call_noiseq", self.call_noiseq),
        ]

    def call_noiseq(
        self,
        count_data: DataFrame,
        factors: DataFrame,
        biotypes: List[str],
        lengths: List[int],
        df_chrom: DataFrame,
        _laplace_offset: float,
    ) -> DataFrame:
        """
        Calls NOISeq via r2py.

        Prior to calling NOISeq, all input data is converted to something R
        can understand.

        Parameters
        ----------
        count_data : DataFrame
            DataFrame with count data.
        factors : DataFrame
            DataFrame with factor data.
        biotypes : List[str]
            List of biotypes ordered as in cout_data.
        lengths : List[int]
            List of gene lengths ordered as in cout_data.
        df_chrom : DataFrame
            DataFrame with 'chr', 'start', 'stop'
        _laplace_offset : float
            Offset to add to avoid zero division.

        Returns
        -------
        DataFrame
            Result DataFrame from NOISeq.
        """
        data = mbf_r.convert_dataframe_to_r(count_data)
        factors = mbf_r.convert_dataframe_to_r(factors)
        df_chrom = df_chrom.astype({"start": "int32", "stop": "int32"})
        chromosome = mbf_r.convert_dataframe_to_r(df_chrom)
        biotype = robjects.vectors.StrVector(biotypes)
        stable_ids = robjects.vectors.StrVector(list(df_chrom.index.values))
        biotype.names = stable_ids
        length = robjects.vectors.IntVector(lengths)
        length.names = stable_ids
        conditions = robjects.vectors.StrVector(["a", "base"])
        noisedata = robjects.r("readData")(
            data=data,
            factors=factors,
            biotype=biotype,
            length=length,
            chromosome=chromosome,
        )
        noiseq = robjects.r("noiseq")(
            noisedata,
            k=_laplace_offset,
            norm=self.norm,
            factor="condition",
            replicates=self.replicates,
            conditions=conditions,
            lc=self.lc,
            pnr=self.pnr,
            nss=self.nss,
            v=self.v,
        )
        results = robjects.r("function(mynoiseq){mynoiseq@results}")(noiseq)
        df = mbf_r.convert_dataframe_from_r(robjects.r("as.data.frame")(results))
        return df


class DESeq2UnpairedOld(DESeq2Unpaired):
    # this is the original deseq2unpaired, i keep it to reproduce the erroneus results, please delete this
    name = "DESeq2UnpairedOld"

    def call_DESeq2(self, count_data, samples, conditions):
        """Call DESeq2.
        @count_data is a DataFrame with 'samples' as the column names.
        @samples is a list. @conditions as well. Condition is the one you're contrasting on.
        You can add additional_conditions (a DataFrame, index = samples) which DESeq2 will
        keep under consideration (changes the formula).
        """
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        import mbf_r

        count_data = count_data.values
        count_data = np.array(count_data)
        nr, nc = count_data.shape
        count_data = count_data.reshape(count_data.size)  # turn into 1d vector
        count_data = robjects.r.matrix(
            numpy2ri.py2rpy(count_data), nrow=nr, ncol=nc, byrow=True
        )
        col_data = pd.DataFrame({"sample": samples, "condition": conditions}).set_index(
            "sample"
        )
        formula = "~ condition"
        col_data = col_data.reset_index(drop=True)
        col_data = mbf_r.convert_dataframe_to_r(pd.DataFrame(col_data.to_dict("list")))
        deseq_experiment = robjects.r("DESeqDataSetFromMatrix")(
            countData=count_data, colData=col_data, design=robjects.Formula(formula)
        )
        deseq_experiment = robjects.r("DESeq")(deseq_experiment)
        res = robjects.r("results")(
            deseq_experiment, contrast=robjects.r("c")("condition", "c", "base")
        )
        df = mbf_r.convert_dataframe_from_r(robjects.r("as.data.frame")(res))
        return df

    def compare(self, df, columns_a, columns_b, columns_other, _laplace_offset):
        # laplace_offset is ignored
        import rpy2.robjects as robjects

        robjects.r('library("DESeq2")')
        columns = []
        conditions = []
        samples = []
        for (name, cols) in [
            ("c", columns_a),
            ("other", columns_other),
            ("base", columns_b),
        ]:
            for col in cols:
                columns.append(col)
                conditions.append(name)
                samples.append(col)
        for col in df.columns:
            print(col)
        print("-------------")
        for c in df.columns:
            print(c, c in columns)
        count_data = df[columns]
        df = self.call_DESeq2(count_data, samples, conditions)
        df = df.rename(
            columns={"log2FoldChange": "log2FC", "pvalue": "p", "padj": "FDR"}
        )
        return df[self.columns].reset_index(drop=True)
