import pypipegraph as ppg
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
import numpy as np


class Log2FC:
    min_sample_count = 0

    def __init__(self):
        self.columns = ["log2FC"]
        self.name = "simple"

    def compare(self, df, columns_a, columns_b, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        return pd.DataFrame({"log2FC": logFC})


class TTest:
    """Standard students t-test, independent on log2FC + benjamini hochberg"""

    min_sample_count = 3

    def __init__(self, equal_variance=False):
        self.equal_var = equal_variance
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest"

    def compare(self, df, columns_a, columns_b, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_ind(a, b, axis=1, equal_var=self.equal_var, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})


class TTestPaired:
    """Standard students t-test, paired, on log2FC + benjamini hochberg"""

    min_sample_count = 3

    def __init__(self):
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest_paired"

    def compare(self, df, columns_a, columns_b, laplace_offset):
        a = np.log2(df[columns_a] + laplace_offset)
        b = np.log2(df[columns_b] + laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_rel(a, b, axis=1, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})


class EdgeRUnpaired:

    min_sample_count = 3
    name = "edgeRunpaired"
    columns = ["log2FC", "p", "FDR"]

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
        samples.insert(0, "group", ["b"] * len(columns_b) + ["a"] * len(columns_a))
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

    def compare(self, df, columns_a, columns_b, _laplace_offset):
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


class DESeq2Unpaired:
    min_sample_count = 3
    name = "DESeq2unpaired"
    columns = ["log2FC", "p", "FDR"]

    def deps(self):
        import rpy2.robjects as ro

        ro.r("library('DESeq2')")
        version = str(ro.r("packageVersion")("DESeq2"))
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name, (version,)
        )

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

        count_data = count_data.as_matrix()
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
        res = robjects.r("results")(deseq_experiment)
        df = mbf_r.convert_dataframe_from_r(robjects.r("as.data.frame")(res))
        return df

    def compare(self, df, columns_a, columns_b, _laplace_offset):
        # laplace_offset is ignored
        import rpy2.robjects as robjects

        robjects.r('library("DESeq2")')
        columns = []
        conditions = []
        samples = []
        for (name, cols) in [
            ("c", columns_a),  # this must be the second value...
            # this must be first in alphabetical sorting
            ("base", columns_b),
        ]:
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
