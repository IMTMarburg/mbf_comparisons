import pytest
import pypipegraph as ppg
import itertools
from pytest import approx
import pandas as pd
import numpy
from mbf_genomics import DelayedDataFrame
from mbf_genomics.annotator import Constant
from mbf_comparisons import (
    Comparisons,
    Log2FC,
    TTest,
    TTestPaired,
    EdgeRUnpaired,
    EdgeRPaired,
    DESeq2Unpaired,
    NOISeq,
    DESeq2MultiFactor,
)
from mbf_qualitycontrol import prune_qc, get_qc_jobs
from mbf_qualitycontrol.testing import assert_image_equal
from mbf_sampledata import get_pasilla_data_subset

from pypipegraph.testing import (
    # RaisesDirectOrInsidePipegraph,
    run_pipegraph,
    force_load,
)  # noqa: F401
from dppd import dppd

dp, X = dppd()


@pytest.mark.usefixtures("both_ppg_and_no_ppg_no_qc")
class TestComparisons:
    def test_simple(self):
        d = DelayedDataFrame("ex1", pd.DataFrame({"a": [1, 2, 3], "b": [2, 8, 16 * 3]}))
        c = Comparisons(d, {"a": ["a"], "b": ["b"]})
        a = c.a_vs_b("a", "b", Log2FC, laplace_offset=0)
        assert d.has_annotator(a)
        force_load(d.add_annotator(a), "fl1")
        run_pipegraph()
        assert (d.df[a["log2FC"]] == [-1.0, -2.0, -4.0]).all()

    def test_simple_from_anno(self):
        d = DelayedDataFrame("ex1", pd.DataFrame({"a": [1, 2, 3], "b": [2, 8, 16 * 3]}))
        a = Constant("five", 5)
        b = Constant("ten", 10)
        c = Comparisons(d, {"a": [a], "b": [b]})
        a = c.a_vs_b("a", "b", Log2FC(), laplace_offset=0)
        force_load(d.add_annotator(a), "fl1")
        run_pipegraph()
        assert (d.df[a["log2FC"]] == [-1, -1, -1]).all()

    def test_simple_from_anno_plus_column_name(self):
        d = DelayedDataFrame("ex1", pd.DataFrame({"a": [1, 2, 3], "b": [2, 8, 16 * 3]}))
        a = Constant("five", 5)
        b = Constant("ten", 10)
        c = Comparisons(d, {"a": [(a, "five")], "b": [(b, "ten")]})
        a = c.a_vs_b("a", "b", Log2FC(), laplace_offset=0)
        force_load(d.add_annotator(a), "fl1")
        run_pipegraph()
        assert (d.df[a["log2FC"]] == [-1, -1, -1]).all()

    def test_simple_from_anno_plus_column_pos(self):
        d = DelayedDataFrame("ex1", pd.DataFrame({"a": [1, 2, 3], "b": [2, 8, 16 * 3]}))
        a = Constant("five", 5)
        b = Constant("ten", 10)
        c = Comparisons(d, {"a": [(a, 0)], "b": [(b, 0)]})
        a = c.a_vs_b("a", "b", Log2FC(), laplace_offset=0)
        force_load(d.add_annotator(a), "fl1")
        run_pipegraph()
        assert (d.df[a["log2FC"]] == [-1, -1, -1]).all()

    def test_input_checking(self):
        d = DelayedDataFrame("ex1", pd.DataFrame({"a": [1, 2, 3], "b": [2, 8, 16 * 3]}))
        with pytest.raises(ValueError):
            Comparisons(None, {})
        with pytest.raises(ValueError):
            Comparisons(d, {55: {"a"}, "b": ["b"]})

    def test_multi_plus_filter(self, clear_annotators):
        d = DelayedDataFrame(
            "ex1",
            pd.DataFrame(
                {
                    "a1": [1 / 0.99, 2 / 0.99, 3 / 0.99],
                    "a2": [1 * 0.99, 2 * 0.99, 3 * 0.99],
                    "b1": [2 * 0.99, 8 * 0.99, (16 * 3) * 0.99],
                    "b2": [2 / 0.99, 8 / 0.99, (16 * 3) / 0.99],
                    "delta": [10, 20, 30],
                }
            ),
        )
        c = Comparisons(d, {"a": ["a1", "a2"], "b": ["b1", "b2"]})
        a = c.a_vs_b("a", "b", Log2FC(), laplace_offset=0)
        anno1 = Constant("shu1", 5)
        anno2 = Constant("shu2", 5)  # noqa: F841
        anno3 = Constant("shu3", 5)  # noqa: F841
        to_test = [
            (("log2FC", "==", -1.0), [-1.0]),
            (("log2FC", ">", -2.0), [-1.0]),
            (("log2FC", "<", -2.0), [-4.0]),
            (("log2FC", ">=", -2.0), [-1.0, -2.0]),
            (("log2FC", "<=", -2.0), [-2.0, -4.0]),
            (("log2FC", "|>", 2.0), [-4.0]),
            (("log2FC", "|<", 2.0), [-1.0]),
            (("log2FC", "|>=", 2.0), [-2.0, -4.0]),
            (("log2FC", "|<=", 2.0), [-1.0, -2.0]),
            ((a["log2FC"], "<", -2.0), [-4.0]),
            (("log2FC", "|", -2.0), ValueError),
            ([("log2FC", "|>=", 2.0), ("log2FC", "<=", 0)], [-2.0, -4.0]),
            ((anno1, ">=", 5), [-1, -2.0, -4.0]),
            (((anno1, 0), ">=", 5), [-1, -2.0, -4.0]),
            (("shu2", ">=", 5), [-1, -2.0, -4.0]),
            (("delta", ">", 10), [-2.0, -4.0]),
        ]
        if not ppg.inside_ppg():  # can't test for missing columns in ppg.
            to_test.extend([(("log2FC_no_such_column", "<", -2.0), KeyError)])
        filtered = {}
        for ii, (f, r) in enumerate(to_test):
            if r in (ValueError, KeyError):
                with pytest.raises(r):
                    a.filter([f], "new%i" % ii)
            else:
                filtered[tuple(f)] = a.filter(
                    [f] if isinstance(f, tuple) else f, "new%i" % ii
                )
                assert filtered[tuple(f)].name == "new%i" % ii
                force_load(filtered[tuple(f)].annotate(), filtered[tuple(f)].name)

        force_load(d.add_annotator(a), "somethingsomethingjob")
        run_pipegraph()
        c = a["log2FC"]
        assert (d.df[c] == [-1.0, -2.0, -4.0]).all()
        for f, r in to_test:
            if r not in (ValueError, KeyError):
                try:
                    assert filtered[tuple(f)].df[c].values == approx(r)
                except AssertionError:
                    print(f)
                    raise

    def test_ttest(self):
        data = pd.DataFrame(
            {
                "A.R1": [0, 0, 0, 0],
                "A.R2": [0, 0, 0, 0],
                "A.R3": [0, 0.001, 0.001, 0.001],
                "B.R1": [0.95, 0, 0.56, 0],
                "B.R2": [0.99, 0, 0.56, 0],
                "B.R3": [0.98, 0, 0.57, 0.5],
                "C.R1": [0.02, 0.73, 0.59, 0],
                "C.R2": [0.03, 0.75, 0.57, 0],
                "C.R3": [0.05, 0.7, 0.58, 1],
            }
        )
        ddf = DelayedDataFrame("ex1", data)
        gts = {
            k: list(v)
            for (k, v) in itertools.groupby(sorted(data.columns), lambda x: x[0])
        }

        c = Comparisons(ddf, gts)
        a = c.a_vs_b("A", "B", TTest)
        b = a.filter([("log2FC", ">", 2.38), ("p", "<", 0.05)])
        assert b.name == "Filtered_A-B_log2FC_＞_2.38__p_＜_0.05"
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        # value calculated with R to double check.
        assert ddf.df[a["p"]].iloc[0] == pytest.approx(8.096e-07, abs=1e-4)
        # value calculated with scipy to double check.
        assert ddf.df[a["p"]].iloc[1] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[a["p"]].iloc[2] == pytest.approx(0.04157730613277929, abs=1e-4)
        assert ddf.df[a["p"]].iloc[3] == pytest.approx(0.703158104919873, abs=1e-4)
        assert ddf.df[a["FDR"]].values == pytest.approx(
            [3.238535e-06, 5.635329e-01, 8.315462e-02, 7.031581e-01], abs=1e-4
        )

    def test_ttest_min_sample_count(self):
        df = pd.DataFrame(
            {"A.R1": [0, 0, 0, 0], "A.R2": [0, 0, 0, 0], "B.R1": [0.95, 0, 0.56, 0]}
        )
        ddf = DelayedDataFrame("x", df)
        gts = {
            k: list(v)
            for (k, v) in itertools.groupby(sorted(df.columns), lambda x: x[0])
        }

        c = Comparisons(ddf, gts)
        with pytest.raises(ValueError):
            c.a_vs_b("A", "B", TTest())

    def test_ttest_paired(self):
        data = pd.DataFrame(
            {
                "A.R1": [0, 0, 0, 0],
                "A.R2": [0, 0, 0, 0],
                "A.R3": [0, 0.001, 0.001, 0.001],
                "B.R1": [0.95, 0, 0.56, 0],
                "B.R2": [0.99, 0, 0.56, 0],
                "B.R3": [0.98, 0, 0.57, 0.5],
                "C.R1": [0.02, 0.73, 0.59, 0],
                "C.R2": [0.03, 0.75, 0.57, 0],
                "C.R3": [0.05, 0.7, 0.58, 1],
            }
        )
        ddf = DelayedDataFrame("ex1", data)
        gts = {
            k: list(v)
            for (k, v) in itertools.groupby(sorted(data.columns), lambda x: x[0])
        }

        c = Comparisons(ddf, gts)
        a = c.a_vs_b("A", "B", TTestPaired())
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        assert ddf.df[a["p"]].iloc[0] == pytest.approx(8.096338300746213e-07, abs=1e-4)
        assert ddf.df[a["p"]].iloc[1] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[a["p"]].iloc[2] == pytest.approx(0.041378369826042816, abs=1e-4)
        assert ddf.df[a["p"]].iloc[3] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[a["FDR"]].values == pytest.approx(
            [3.238535e-06, 4.226497e-01, 8.275674e-02, 4.226497e-01], abs=1e-4
        )

    def test_double_comparison_with_different_strategies(self):
        data = pd.DataFrame(
            {
                "A.R1": [0, 0, 0, 0],
                "A.R2": [0, 0, 0, 0],
                "A.R3": [0, 0.001, 0.001, 0.001],
                "B.R1": [0.95, 0, 0.56, 0],
                "B.R2": [0.99, 0, 0.56, 0],
                "B.R3": [0.98, 0, 0.57, 0.5],
                "C.R1": [0.02, 0.73, 0.59, 0],
                "C.R2": [0.03, 0.75, 0.57, 0],
                "C.R3": [0.05, 0.7, 0.58, 1],
            }
        )
        ddf = DelayedDataFrame("ex1", data)
        gts = {
            k: list(v)
            for (k, v) in itertools.groupby(sorted(data.columns), lambda x: x[0])
        }

        c = Comparisons(ddf, gts)
        a = c.a_vs_b("A", "B", TTestPaired())
        force_load(ddf.add_annotator(a))
        b = c.a_vs_b("A", "B", TTest())
        force_load(ddf.add_annotator(b))
        run_pipegraph()
        assert ddf.df[a["p"]].iloc[0] == pytest.approx(8.096338300746213e-07, abs=1e-4)
        assert ddf.df[a["p"]].iloc[1] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[a["p"]].iloc[2] == pytest.approx(0.041378369826042816, abs=1e-4)
        assert ddf.df[a["p"]].iloc[3] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[a["FDR"]].values == pytest.approx(
            [3.238535e-06, 4.226497e-01, 8.275674e-02, 4.226497e-01], abs=1e-4
        )
        assert ddf.df[b["p"]].iloc[0] == pytest.approx(8.096e-07, abs=1e-4)
        # value calculated with scipy to double check.
        assert ddf.df[b["p"]].iloc[1] == pytest.approx(0.42264973081037427, abs=1e-4)
        assert ddf.df[b["p"]].iloc[2] == pytest.approx(0.04157730613277929, abs=1e-4)
        assert ddf.df[b["p"]].iloc[3] == pytest.approx(0.703158104919873, abs=1e-4)
        assert ddf.df[b["FDR"]].values == pytest.approx(
            [3.238535e-06, 5.635329e-01, 8.315462e-02, 7.031581e-01], abs=1e-4
        )

    def _get_tuch_data(self):
        import mbf_sampledata
        import mbf_r
        import rpy2.robjects as ro

        path = mbf_sampledata.get_sample_path("mbf_comparisons/TuchEtAlS1.csv")
        # directly from the manual.
        # plus minus """To make
        # this file, we downloaded Table S1 from Tuch et al. [39], deleted some unnecessary columns
        # and edited the column headings slightly:"""
        ro.r(
            """load_data = function(path) {
                rawdata <- read.delim(path, check.names=FALSE, stringsAsFactors=FALSE)
                library(edgeR)
                y <- DGEList(counts=rawdata[,3:8], genes=rawdata[,1:2])
                library(org.Hs.eg.db)
                idfound <- y$genes$idRefSeq %in% mappedRkeys(org.Hs.egREFSEQ)
                y <- y[idfound,]
                egREFSEQ <- toTable(org.Hs.egREFSEQ)
                m <- match(y$genes$idRefSeq, egREFSEQ$accession)
                y$genes$EntrezGene <- egREFSEQ$gene_id[m]
                egSYMBOL <- toTable(org.Hs.egSYMBOL)
                m <- match(y$genes$EntrezGene, egSYMBOL$gene_id)
                y$genes$Symbol <- egSYMBOL$symbol[m]

                o <- order(rowSums(y$counts), decreasing=TRUE)
                y <- y[o,]
                d <- duplicated(y$genes$Symbol)
                y <- y[!d,]

                cbind(y$genes, y$counts)
            }
"""
        )
        df = mbf_r.convert_dataframe_from_r(ro.r("load_data")(str(path)))
        df.columns = [
            "idRefSeq",
            "nameOfGene",
            "EntrezGene",
            "Symbol",
            "8.N",
            "8.T",
            "33.N",
            "33.T",
            "51.N",
            "51.T",
        ]
        assert len(df) == 10519
        return df

    def test_edgeR(self):
        df = self._get_tuch_data()

        ddf = DelayedDataFrame("ex1", df)
        gts = {
            "T": [x for x in df.columns if ".T" in x],
            "N": [x for x in df.columns if ".N" in x],
        }

        c = Comparisons(ddf, gts)
        a = c.a_vs_b("T", "N", EdgeRUnpaired())
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        # these are from the last run - the manual has no simple a vs b comparison...
        # at least we'l notice if this changes
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["log2FC"]].values == approx(
            [4.003122]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["FDR"]].values == approx(
            [1.332336e-11]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["p"]].values == approx(
            [5.066397e-15]
        )
        df = ddf.df.set_index("nameOfGene")
        t_columns = [x[1] for x in gts["T"]]
        n_columns = [x[1] for x in gts["N"]]
        assert df.loc["PTHLH"][t_columns].sum() > df.loc["PTHLH"][n_columns].sum()

        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["log2FC"]].values == approx(
            [-5.127508]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["FDR"]].values == approx(
            [6.470885e-10]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["p"]].values == approx(
            [3.690970e-13]
        )
        assert df.loc["PTGFR"][t_columns].sum() < df.loc["PTGFR"][n_columns].sum()

    def test_edgeR_paired(self):
        df = self._get_tuch_data()

        ddf = DelayedDataFrame("ex1", df)
        gts = {
            "T": [x for x in sorted(df.columns) if ".T" in x],
            "N": [x for x in sorted(df.columns) if ".N" in x],
        }

        c = Comparisons(ddf, gts)
        a = c.a_vs_b("T", "N", EdgeRPaired())
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        # these are from the last run - the manual has no simple a vs b comparison...
        # at least we'l notice if this changes
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["log2FC"]].values == approx(
            [3.97], abs=1e-3
        )
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["FDR"]].values == approx(
            [4.27e-18]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTHLH"][a["p"]].values == approx([8.13e-22])
        df = ddf.df.set_index("nameOfGene")
        t_columns = [x[1] for x in gts["T"]]
        n_columns = [x[1] for x in gts["N"]]
        assert df.loc["PTHLH"][t_columns].sum() > df.loc["PTHLH"][n_columns].sum()

        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["log2FC"]].values == approx(
            [-5.18], abs=1e-2
        )
        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["FDR"]].values == approx(
            [3.17e-19]
        )
        assert ddf.df[ddf.df.nameOfGene == "PTGFR"][a["p"]].values == approx([3.01e-23])
        assert df.loc["PTGFR"][t_columns].sum() < df.loc["PTGFR"][n_columns].sum()

    def test_edgeR_filter_on_max_count(self):
        ddf, a, b = get_pasilla_data_subset()
        gts = {"T": a, "N": b}
        c = Comparisons(ddf, gts)
        a = c.a_vs_b("T", "N", EdgeRUnpaired(ignore_if_max_count_less_than=100))
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        assert pd.isnull(ddf.df[a["log2FC"]]).any()
        assert (pd.isnull(ddf.df[a["log2FC"]]) == pd.isnull(ddf.df[a["p"]])).all()
        assert (pd.isnull(ddf.df[a["FDR"]]) == pd.isnull(ddf.df[a["p"]])).all()

    def test_deseq2(self):
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        gts = {
            "treated": [x for x in pasilla_data.columns if x.startswith("treated")],
            "untreated": [x for x in pasilla_data.columns if x.startswith("untreated")],
        }
        ddf = DelayedDataFrame("ex", pasilla_data)
        c = Comparisons(ddf, gts)
        a = c.a_vs_b("treated", "untreated", DESeq2Unpaired())
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        check = """# This is deseq2 version specific data- probably needs fixing if upgrading deseq2
## baseMean log2FoldChange lfcSE stat pvalue padj
## <numeric> <numeric> <numeric> <numeric> <numeric> <numeric>
## FBgn0039155 453 -3.72 0.160 -23.2 1.63e-119 1.35e-115
## FBgn0029167 2165 -2.08 0.103 -20.3 1.43e-91 5.91e-88
## FBgn0035085 367 -2.23 0.137 -16.3 6.38e-60 1.75e-56
## FBgn0029896 258 -2.21 0.159 -13.9 5.40e-44 1.11e-40
## FBgn0034736 118 -2.56 0.185 -13.9 7.66e-44 1.26e-40
"""
        df = ddf.df.sort_values(a["FDR"])
        df = df.set_index("Gene")
        for row in check.split("\n"):
            row = row.strip()
            if row and not row[0] == "#":
                row = row.split()
                self.assertAlmostEqual(
                    df.ix[row[0]][a["log2FC"]], float(row[2]), places=2
                )
                self.assertAlmostEqual(df.ix[row[0]][a["p"]], float(row[5]), places=2)
                self.assertAlmostEqual(df.ix[row[0]][a["FDR"]], float(row[6]), places=2)

    def _get_pasilla_3(self):
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        pasilla_data = pasilla_data.set_index("Gene")
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        seed = 12345
        numpy.random.seed(seed)
        for i in range(3):
            pasilla_data[f"other{i}fb"] = (
                pasilla_data["untreated4fb"].values
                + numpy.abs(numpy.random.randn(len(pasilla_data)) * 10)
            ).astype(int)
        for i in range(3):
            pasilla_data[f"otherse{i}fb"] = (
                pasilla_data["untreated4fb"].values
                + numpy.abs(numpy.random.randn(len(pasilla_data)) * 15)
            ).astype(int)
        return pasilla_data

    def test_deseq2_3groups(self):
        import mbf_r
        import rpy2.robjects as robjects

        robjects.r("library(DESeq2)")
        pasilla_data = self._get_pasilla_3()
        condition_data = pd.DataFrame(
            {
                "condition": [x[:-3] for x in pasilla_data.columns],
                "type": [
                    "se"
                    if x
                    in ["treated1fb", "untreated1fb", "untreated2fb"]
                    + [f"otherse{i}fb" for i in range(3)]
                    else "pe"
                    for x in pasilla_data.columns
                ],
            },
            index=pasilla_data.columns,
        )
        gts = {}
        for cond, sub in condition_data.groupby("condition"):
            gts[cond] = list(sub.index.values)
        cts = mbf_r.convert_dataframe_to_r(pasilla_data)
        col = mbf_r.convert_dataframe_to_r(condition_data)
        rresults = robjects.r(
            """
            function (cts, col){
                dds = DESeqDataSetFromMatrix(countData=cts, colData=col, design = ~ condition)
                dds = DESeq(dds)
                print(resultsNames(dds))
                res = results(dds, contrast=c("condition", "treated", "untreated"))
                res = as.data.frame(res)
                res
            }
        """
        )(cts=cts, col=col)
        ddf = DelayedDataFrame("ex", pasilla_data)
        c = Comparisons(ddf, gts)
        a = c.a_vs_b(
            "treated",
            "untreated",
            DESeq2Unpaired(),
            include_other_samples_for_variance=True,
        )
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        rresults = mbf_r.convert_dataframe_from_r(rresults)
        numpy.testing.assert_almost_equal(
            rresults["log2FoldChange"].values,
            ddf.df[
                "Comp. treated - untreated log2FC (DESeq2unpaired,Other=True)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults["pvalue"].values,
            ddf.df["Comp. treated - untreated p (DESeq2unpaired,Other=True)"].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults["padj"].values,
            ddf.df["Comp. treated - untreated FDR (DESeq2unpaired,Other=True)"].values,
            decimal=4,
        )

    def test_deseq2_multi(self):
        import mbf_r
        import rpy2.robjects as robjects

        robjects.r("library(DESeq2)")
        pasilla_data = self._get_pasilla_3()
        condition_data = pd.DataFrame(
            {
                "condition": [
                    "treated"
                    if (x.startswith("treated") | x.startswith("otherse"))
                    else "base.untreated"
                    for x in pasilla_data.columns
                ],
                "type": [
                    "se" if x.startswith("other") else "pe"
                    for x in pasilla_data.columns
                ],
            },
            index=pasilla_data.columns,
        )
        gts = {}
        groups = []
        df_factors = pd.DataFrame(
            {
                "group": [
                    "base.untreated_pe",
                    "base.untreated_se",
                    "treated_pe",
                    "treated_se",
                ],
                "condition": ["base.untreated", "base.untreated", "treated", "treated"],
                "type": ["pe", "se", "pe", "se"],
            }
        )
        for cond1, sub in condition_data.groupby("condition"):
            for cond2, sub2 in sub.groupby("type"):
                group = f"{cond1}_{cond2}"
                groups.extend([group for i in sub2.index])
                gts[group] = list(sub2.index.values)
        cts = mbf_r.convert_dataframe_to_r(pasilla_data)
        col = mbf_r.convert_dataframe_to_r(condition_data)
        rresults_pe = robjects.r(
            """
            function (cts, col){
                dds = DESeqDataSetFromMatrix(countData=cts, colData=col, design = ~ type + condition + type:condition)
                dds = DESeq(dds)
                res = results(dds, contrast=c("condition", "treated", "base.untreated"))
                res = as.data.frame(res)
                res
            }
        """
        )(cts=cts, col=col)
        rresults_se = robjects.r(
            """
            function (cts, col){
                dds = DESeqDataSetFromMatrix(countData=cts, colData=col, design = ~ type + condition + type:condition)
                dds = DESeq(dds)
                res = results(dds, list( c("condition_treated_vs_base.untreated", "typese.conditiontreated") ))
                res = as.data.frame(res)
                res
            }
        """
        )(cts=cts, col=col)
        ddf = DelayedDataFrame("ex", pasilla_data)
        c = Comparisons(ddf, gts)
        factor_reference = {"condition": "base.untreated", "type": "pe"}
        condition_data["group"] = groups
        a = c.multi(
            name="multi",
            main_factor="condition",
            factor_reference=factor_reference,
            df_factors=df_factors,
            interactions=[("type", "condition")],
            method=DESeq2MultiFactor(),
            test_difference=True,
            compare_non_reference=False,
        )
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        rresults_pe = mbf_r.convert_dataframe_from_r(rresults_pe)
        rresults_se = mbf_r.convert_dataframe_from_r(rresults_se)
        numpy.testing.assert_almost_equal(
            rresults_pe["log2FoldChange"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for pe(type) log2FC (Comp. multi)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults_pe["padj"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for pe(type) FDR (Comp. multi)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults_pe["pvalue"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for pe(type) p (Comp. multi)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults_se["log2FoldChange"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for se:pe(type) log2FC (Comp. multi)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults_se["padj"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for se:pe(type) FDR (Comp. multi)"
            ].values,
            decimal=4,
        )
        numpy.testing.assert_almost_equal(
            rresults_se["pvalue"].values,
            ddf.df[
                "treated:base.untreated(condition) effect for se:pe(type) p (Comp. multi)"
            ].values,
            decimal=4,
        )

    def test_other_sample_dependencies(self):
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]

        gts = {
            "treated": [x for x in pasilla_data.columns if x.startswith("treated")],
            "untreated": [x for x in pasilla_data.columns if x.startswith("untreated")],
        }
        ddf = DelayedDataFrame("ex", pasilla_data)
        c = Comparisons(ddf, gts)
        a = c.a_vs_b("treated", "untreated", DESeq2Unpaired())
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        check = """# This is deseq2 version specific data- probably needs fixing if upgrading deseq2
## baseMean log2FoldChange lfcSE stat pvalue padj
## <numeric> <numeric> <numeric> <numeric> <numeric> <numeric>
## FBgn0039155 453 -3.72 0.160 -23.2 1.63e-119 1.35e-115
## FBgn0029167 2165 -2.08 0.103 -20.3 1.43e-91 5.91e-88
## FBgn0035085 367 -2.23 0.137 -16.3 6.38e-60 1.75e-56
## FBgn0029896 258 -2.21 0.159 -13.9 5.40e-44 1.11e-40
## FBgn0034736 118 -2.56 0.185 -13.9 7.66e-44 1.26e-40
"""
        df = ddf.df.sort_values(a["FDR"])
        df = df.set_index("Gene")
        for row in check.split("\n"):
            row = row.strip()
            if row and not row[0] == "#":
                row = row.split()
                self.assertAlmostEqual(
                    df.ix[row[0]][a["log2FC"]], float(row[2]), places=2
                )
                self.assertAlmostEqual(df.ix[row[0]][a["p"]], float(row[5]), places=2)
                self.assertAlmostEqual(df.ix[row[0]][a["FDR"]], float(row[6]), places=2)

    def _get_marioni_data(self):
        import mbf_r
        import rpy2.robjects as robjects

        robjects.r("library(NOISeq)")
        robjects.r("data(Marioni)")
        counts = mbf_r.convert_dataframe_from_r(robjects.r("mycounts"))
        counts["gene_stable_id"] = counts.index.values
        factors = mbf_r.convert_dataframe_from_r(robjects.r("myfactors"))
        chroms = mbf_r.convert_dataframe_from_r(robjects.r("mychroms"))
        counts["chr"] = chroms["Chr"]
        counts["start"] = chroms["GeneStart"]
        counts["stop"] = chroms["GeneEnd"]
        biotypes = robjects.r("mybiotypes")
        counts["biotype"] = biotypes
        mynoiseq = robjects.r(
            """
            mydata = readData(data=mycounts, length = mylength, biotype = mybiotypes, chromosome=mychroms, factors=myfactors)
            mynoiseq = noiseq(mydata, k = 0.5, norm = "tmm", factor = "Tissue", pnr = 0.2, nss = 5, v = 0.02, lc = 0, replicates = "technical")
            """
        )
        results = robjects.r("function(mynoiseq){as.data.frame(mynoiseq@results)}")(
            mynoiseq
        )
        results = mbf_r.convert_dataframe_from_r(results)
        up = robjects.r(
            "function(mynoiseseq){as.data.frame(degenes(mynoiseq, q = 0.8, M = 'up'))}"
        )(mynoiseq)
        up = mbf_r.convert_dataframe_from_r(up)
        return counts, factors, results, up

    def test_noiseq(self):
        df_counts, df_factors, results, up = self._get_marioni_data()
        ddf = DelayedDataFrame("ex1", df_counts)
        gts = {}
        for tissue, sub in df_factors.groupby("Tissue"):
            gts[tissue] = list(sub.index.values)
        c = Comparisons(ddf, gts)
        noise = NOISeq()
        assert noise.norm == "tmm"
        assert noise.lc == 0
        assert noise.v == 0.02
        assert noise.nss == 5
        a = c.a_vs_b("Kidney", "Liver", noise, laplace_offset=0.5)
        force_load(ddf.add_annotator(a))
        run_pipegraph()
        numpy.testing.assert_array_equal(
            results["ranking"], ddf.df["Comp. Kidney - Liver Rank (NOIseq,Other=False)"]
        )
        numpy.testing.assert_array_equal(
            results["prob"], ddf.df["Comp. Kidney - Liver Prob (NOIseq,Other=False)"]
        )
        numpy.testing.assert_array_equal(
            results["M"], ddf.df["Comp. Kidney - Liver log2FC (NOIseq,Other=False)"]
        )
        numpy.testing.assert_array_equal(
            results["D"], ddf.df["Comp. Kidney - Liver D (NOIseq,Other=False)"]
        )
        upregulated = ddf.df[
            (ddf.df["Comp. Kidney - Liver Prob (NOIseq,Other=False)"] >= 0.8)
            & (ddf.df["Comp. Kidney - Liver log2FC (NOIseq,Other=False)"] > 0)
        ]
        genes_up = set(upregulated["gene_stable_id"])
        genes_should_up = set(up.index.values)
        assert (
            len(genes_up.intersection(genes_should_up))
            == len(genes_up)
            == len(genes_should_up)
        )


@pytest.mark.usefixtures("no_pipegraph")
class TestComparisonsNoPPG:
    def test_deseq2_with_and_without_additional_columns(self):
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        print(pasilla_data.columns)
        pasilla_data = pasilla_data.assign(
            treated_fake=pasilla_data.treated2fb,
            untreated_fake=pasilla_data.untreated2fb,
        )

        gts = {
            "treated": [
                x
                for x in pasilla_data.columns
                if x.startswith("treated") and "3" not in x
            ],
            "untreated": [
                x
                for x in pasilla_data.columns
                if x.startswith("untreated") and "3" not in x
            ],
            "other": [x for x in pasilla_data.columns if "3" in x],
        }
        assert len(gts["other"]) == 2
        assert sum((len(x) for x in gts.values())) + 1 == len(
            pasilla_data.columns
        )  # GeneId
        ddf = DelayedDataFrame("ex", pasilla_data)
        c = Comparisons(ddf, gts)
        with_other = c.a_vs_b(
            "treated",
            "untreated",
            DESeq2Unpaired(),
            include_other_samples_for_variance=True,
        )
        without_other = c.a_vs_b(
            "treated",
            "untreated",
            DESeq2Unpaired(),
            include_other_samples_for_variance=False,
        )
        force_load(ddf.add_annotator(with_other))
        force_load(ddf.add_annotator(without_other))
        # run_pipegraph()
        df = ddf.df
        print(df.head())
        df.to_csv("test.csv")
        # this is a fairly weak test, but it shows that it at least does *something*
        assert (df[with_other["p"]] != pytest.approx(df[without_other["p"]])).all()
        assert (
            df[with_other["log2FC"]] != pytest.approx(df[without_other["log2FC"]])
        ).all()


@pytest.mark.usefixtures("new_pipegraph")
class TestQC:
    def test_distribution(self):
        ppg.util.global_pipegraph.quiet = False
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        treated = [x for x in pasilla_data.columns if x.startswith("treated")]
        untreated = [x for x in pasilla_data.columns if x.startswith("untreated")]
        pasilla_data = DelayedDataFrame("pasilla", pasilla_data)
        Comparisons(pasilla_data, {"treated": treated, "untreated": untreated})
        prune_qc(lambda job: "distribution" in job.job_id)
        run_pipegraph()
        qc_jobs = list(get_qc_jobs())
        qc_jobs = [x for x in qc_jobs if not x._pruned]
        print(qc_jobs)
        assert len(qc_jobs) == 1
        assert_image_equal(qc_jobs[0].filenames[0])

    def test_pca(self):
        ppg.util.global_pipegraph.quiet = False
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        treated = [x for x in pasilla_data.columns if x.startswith("treated")]
        untreated = [x for x in pasilla_data.columns if x.startswith("untreated")]
        pasilla_data = DelayedDataFrame("pasilla", pasilla_data)
        Comparisons(pasilla_data, {"treated": treated, "untreated": untreated})
        prune_qc(lambda job: "pca" in job.job_id)
        run_pipegraph()
        qc_jobs = list(get_qc_jobs())
        qc_jobs = [x for x in qc_jobs if not x._pruned]
        print(qc_jobs)
        assert len(qc_jobs) == 1
        assert_image_equal(qc_jobs[0].filenames[0])

    def test_correlation(self):
        ppg.util.global_pipegraph.quiet = False
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        treated = [x for x in pasilla_data.columns if x.startswith("treated")]
        untreated = [x for x in pasilla_data.columns if x.startswith("untreated")]
        pasilla_data = DelayedDataFrame("pasilla", pasilla_data)
        Comparisons(pasilla_data, {"treated": treated, "untreated": untreated})
        prune_qc(lambda job: "correlation" in job.job_id)
        run_pipegraph()
        qc_jobs = list(get_qc_jobs())
        qc_jobs = [x for x in qc_jobs if not x._pruned]
        print(qc_jobs)
        assert len(qc_jobs) == 1
        assert_image_equal(qc_jobs[0].filenames[0])

    def test_volcano_plot(self):
        ppg.util.global_pipegraph.quiet = False
        import mbf_sampledata

        pasilla_data = pd.read_csv(
            mbf_sampledata.get_sample_path(
                "mbf_comparisons/pasillaCount_deseq2.tsv.gz"
            ),
            sep=" ",
        )
        # pasilla_data = pasilla_data.set_index('Gene')
        pasilla_data.columns = [str(x) for x in pasilla_data.columns]
        treated = [x for x in pasilla_data.columns if x.startswith("treated")]
        untreated = [x for x in pasilla_data.columns if x.startswith("untreated")]
        pasilla_data = DelayedDataFrame("pasilla", pasilla_data)
        comp = Comparisons(
            pasilla_data, {"treated": treated, "untreated": untreated}
        ).a_vs_b("treated", "untreated", TTest())
        comp.filter([("log2FC", "|>=", 2.0), ("FDR", "<=", 0.05)])
        prune_qc(lambda job: "volcano" in job.job_id)
        run_pipegraph()
        qc_jobs = list(get_qc_jobs())
        qc_jobs = [x for x in qc_jobs if not x._pruned]
        print(qc_jobs)
        assert len(qc_jobs) == 1
        assert_image_equal(qc_jobs[0].filenames[0])

    @pytest.mark.skip  # no ma plots right now, they're way to slow for general usage :(
    def test_ma_plot(self):
        ppg.util.global_pipegraph.quiet = False
        pasilla_data, treated, untreated = get_pasilla_data_subset()
        import numpy

        numpy.random.seed(500)

        comp = Comparisons(
            pasilla_data, {"treated": treated, "untreated": untreated}
        ).a_vs_b("treated", "untreated", TTest(), laplace_offset=1)

        comp.filter(
            [
                ("log2FC", "|>=", 2.0),
                # ('FDR', '<=', 0.05),
            ]
        )
        prune_qc(lambda job: "ma_plot" in job.job_id)
        run_pipegraph()
        qc_jobs = list(get_qc_jobs())
        qc_jobs = [x for x in qc_jobs if not x._pruned]
        assert len(qc_jobs) == 1
        assert_image_equal(qc_jobs[0].filenames[0])
