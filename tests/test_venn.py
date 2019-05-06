import pytest
import pypipegraph as ppg
import pandas as pd
from mbf_genomics import DelayedDataFrame
from mbf_comparisons import Comparisons, venn, Log2FC
from mbf_qualitycontrol.testing import assert_image_equal


@pytest.mark.usefixtures("new_pipegraph_no_qc")
class TestVenn:
    def test_venn_from_logfcs(self):
        ppg.util.global_pipegraph.quiet = False
        d = DelayedDataFrame(
            "ex1",
            pd.DataFrame(
                {
                    "stable_id": ["A", "B", "C", "D", "E"],
                    "a": [1, 1, 1, 1, 1],
                    "b": [1, 2, 3, 4, 5],
                    "c": [1, 1, 3, 0.5, 0.75],
                }
            ),
        )
        comp = Comparisons(d, {"a": ["a"], "b": ["b"], "c": ["c"]})
        a = comp.all_vs_b("a", Log2FC())
        selected = {name: x.filter([("log2FC", "|>=", 1)]) for name, x in a.items()}
        plot_job = venn.plot_venn("test", selected)
        ppg.run_pipegraph()
        assert_image_equal(plot_job.filenames[0], "_down")
        assert_image_equal(plot_job.filenames[1], "_up")
