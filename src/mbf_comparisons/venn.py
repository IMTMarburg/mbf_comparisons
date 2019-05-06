from matplotlib import pyplot as plt
import pypipegraph as ppg
import venn


def plot_venn(output_prefix, a_dict):
    if hasattr(next(iter(a_dict.values())), "venn_annotator"):
        return plot_venn_from_genes_with_comparisons(output_prefix, a_dict)
    else:
        raise NotImplementedError("Expand!")


def plot_venn_from_genes_with_comparisons(
    output_prefix, a_dict, id_column="gene_stable_id"
):
    if len(a_dict) not in (2, 3):
        raise ValueError("Max support 3 sets currently")

    def plot():
        up = {}
        down = {}
        for name, genes_ddf in sorted(a_dict.items()):
            df = genes_ddf.df
            stable_ids = df[id_column]
            column = genes_ddf.venn_annotator["log2FC"]
            up[name] = set(stable_ids[df[column] > 0])
            down[name] = set(stable_ids[df[column] < 0])
        plt.figure(figsize=(4, 4))
        venn.venn(up)
        plt.savefig(str(output_prefix) + ".up.png", dpi=72)
        plt.figure(figsize=(4, 4))
        venn.venn(down)
        plt.savefig(str(output_prefix) + ".down.png", dpi=72)

    return (
        ppg.MultiFileGeneratingJob(
            [str(output_prefix) + ".up.png", str(output_prefix) + ".down.png"], plot
        )
        .depends_on([x.add_annotator(x.venn_annotator) for x in a_dict.values()])
        .depends_on(ppg.ParameterInvariant(output_prefix, id_column))
    )
