import pytest
import pypipegraph as ppg
import itertools
from pytest import approx
import pandas as pd
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
        assert b.name == "Filtered_A-B_log2FC_>_2.38__p_<_0.05"
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
