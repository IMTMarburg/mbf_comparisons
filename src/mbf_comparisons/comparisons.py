import numpy as np
import pypipegraph as ppg
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests


class Log2FC:
    min_sample_count = 0

    def __init__(self, laplace_offset=1 / 1e6):
        self.laplace_offset = laplace_offset
        self.columns = ["log2FC"]
        self.name = "simple"

    def compare(self, df, columns_a, columns_b):
        a = np.log2(df[columns_a] + self.laplace_offset)
        b = np.log2(df[columns_b] + self.laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        return pd.DataFrame({"log2FC": logFC})

    def get_dependencies(self):
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name, (self.laplace_offset)
        )


class TTest:
    """Standard students t-test, independent on log2FC + benjamini hochberg"""

    min_sample_count = 3

    def __init__(self, equal_variance=False, laplace_offset=1 / 1e6):
        self.laplace_offset = laplace_offset
        self.equal_var = equal_variance
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest"

    def compare(self, df, columns_a, columns_b):
        a = np.log2(df[columns_a] + self.laplace_offset)
        b = np.log2(df[columns_b] + self.laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_ind(a, b, axis=1, equal_var=self.equal_var, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})

    def get_dependencies(self):
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name, (self.laplace_offset)
        )


class TTestPaired:
    """Standard students t-test, paired, on log2FC + benjamini hochberg"""

    min_sample_count = 3

    def __init__(self, laplace_offset=1 / 1e6):
        self.laplace_offset = laplace_offset
        self.columns = ["log2FC", "p", "FDR"]
        self.name = "ttest_paired"

    def compare(self, df, columns_a, columns_b):
        a = np.log2(df[columns_a] + self.laplace_offset)
        b = np.log2(df[columns_b] + self.laplace_offset)
        logFC = a.mean(axis=1, skipna=True) - b.mean(axis=1, skipna=True)
        p = ss.ttest_rel(a, b, axis=1, nan_policy="omit")[1]
        fdr = multipletests(p, method="fdr_bh")[1]
        return pd.DataFrame({"log2FC": logFC, "p": p, "FDR": fdr})

    def get_dependencies(self):
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name, (self.laplace_offset)
        )


class EdgeRUnpaired:

    min_sample_count = 3
    name = "edgeR"

    def __init__(self, ignore_if_max_count_less_than=None, manual_dispersion_value=0.4):
        self.ignore_if_max_count_less_than = ignore_if_max_count_less_than
        self.manual_dispersion_value = manual_dispersion_value
        self.columns = ["log2FC", "p", "FDR"]

    def get_dependencies(self):
        import rpy2.robjects as ro

        ro.r("library('edgeR')")
        version = str(ro.r("packageVersion")("edgeR"))
        return ppg.ParameterInvariant(
            self.__class__.__name__ + "_" + self.name,
            (self.version, self.ignore_if_max_count_less_than),
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
        if library_sizes is not None:
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
            }
        )
        #apply TMM normalization
        y = ro.r('calcNormFactors')(y)
        if len(columns_a) == 1 and len(columns_b) == 1:
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

    def compare(self, df, columns_a, columns_b):
        value_columns = columns_a + columns_b
        # we need to go by key, since filter out nan rows.
        idx = ["G%i" % ii for ii in range(len(df))]
        input_df = df[value_columns]
        input_df = input_df.assign(idx=idx)
        input_df = input_df.set_index("idx")
        if pd.isnull(input_df).any().any():
            raise ValueError("Nans before filtering in edgeR input")

        if self.ignore_if_max_count_less_than is not None:
            max_raw_count_per_gene = input_df.max(axis=1)
            input_df.ix[
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


# TODO: Quantile normalization
# DESEq
# DESEq2
# edger
# genomics - remove *_Biotypes - filtering first then applying should be enough.
# refactor FDR
# test laplaco offfset change makes recalc
