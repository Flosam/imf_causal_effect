## functions for plotting
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# histogram of the distribution of the treatment effect
def plot_hte_distribution(df:pd.DataFrame, result:str, path:str) -> None:
    # compute ate
    ate = df[result].mean()
    # create plot
    plt.figure()
    plt.hist(df[result], bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(ate, color='red', linestyle='--')
    plt.title("Distribution of Estimated CATE")
    plt.xlabel("Estimated Treatment Effect (CATE)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(path)

# scatter plot of treatment effect against another variable
def plot_scatter_hte(df:pd.DataFrame, path:str, result:str, var:str) -> None:
    plt.figure()
    sns.regplot(
        x=df[var],
        y=df[result],
        lowess=True,
        scatter_kws={'alpha':0.3, 's':10},
        line_kws={'color': 'red'}
    )
    plt.xlabel(var)
    plt.ylabel("Estimated CATE")
    plt.grid(alpha=0.3)
    plt.savefig(path)

# line plot to summarize final dataset
def plot_final_summary(df:pd.DataFrame, path:str) -> None:
    print()
    print("========== Summary ==========")

    counts = df.groupby('year').agg({
        'ccode_cow' : 'count',
        'imf_prog' : 'sum'
    }).reset_index()

    # total number of data points
    num_cu = counts['ccode_cow'].sum()
    print("Data points: " + str(num_cu))

    # number of treated units
    num_prog = counts['imf_prog'].sum()
    print("Number of treated units: " + str(num_prog))

    # plot number of countries under program per year
    plt.figure()
    plt.plot(counts['year'], counts['ccode_cow'])
    plt.plot(counts['year'], counts['imf_prog'])
    plt.legend(['Total number of units', 'Number of treated units'])
    plt.savefig(path)
    print("=============================")
    print()


def plot_causal_tree(ct, filename=None, feature_names=None, figsize=(10, 6)):
    """
    Render a simple tree plot using Matplotlib. This is a lightweight fallback
    when Graphviz is not available. The layout places leaves left-to-right and
    parents centered above their children.
    Parameters
    - ct: CausalTree instance
    - filename: if provided, saves the figure to this path (png recommended)
    - feature_names: optional list of feature names
    """
    import matplotlib.pyplot as plt

    if ct.root is None:
        raise ValueError("Tree is empty")

    pos = {}
    # counter for assigning x coordinates to leaves
    leaf_counter = {'x': 0}

    def _layout(node, depth=0):
        if node is None:
            return None, None
        if node.is_leaf:
            x = leaf_counter['x']
            pos[node] = (x, -depth)
            leaf_counter['x'] += 1
            return x, x
        left_min, left_max = _layout(node.left, depth+1)
        right_min, right_max = _layout(node.right, depth+1)
        # handle degenerate cases
        if left_min is None and right_min is None:
            x = leaf_counter['x']
            pos[node] = (x, -depth)
            leaf_counter['x'] += 1
            return x, x
        if left_min is None:
            x = right_min
        elif right_min is None:
            x = left_max
        else:
            x = 0.5 * (left_min + right_max)
        pos[node] = (x, -depth)
        min_x = left_min if left_min is not None else x
        max_x = right_max if right_max is not None else x
        return min_x, max_x

    _layout(ct.root)

    fig, ax = plt.subplots(figsize=figsize)

    # draw edges and nodes
    for node, (x, y) in pos.items():
        if not node.is_leaf:
            for child in (node.left, node.right):
                if child is None:
                    continue
                cx, cy = pos[child]
                ax.plot([x, cx], [y, cy], color='k', linewidth=1)

    for node, (x, y) in pos.items():
        if node.is_leaf:
            label = f"Leaf\nn={node.n_samples}\ntau={node.tau if node.tau is not None else 'nan'}"
            bbox = dict(boxstyle="round,pad=0.3", fc="#f8cecc", ec="k")
        else:
            # safe feature name lookup (guard against short/missing feature_names)
            if feature_names is not None and node.feature is not None:
                try:
                    fname = feature_names[node.feature]
                except Exception:
                    fname = f"X[{node.feature}]"
            else:
                fname = f"X[{node.feature}]"
            thr = f"{node.threshold:.3f}" if node.threshold is not None else "nan"
            label = f"{fname} <= {thr}\nn={node.n_samples}"
            bbox = dict(boxstyle="round,pad=0.3", fc="#c6dbef", ec="k")
        ax.text(x, y, label, ha='center', va='center', bbox=bbox, fontsize=9)

    ax.set_axis_off()
    # set x limits with small margin
    if leaf_counter['x'] > 0:
        ax.set_xlim(-0.5, leaf_counter['x'] - 0.5)

    plt.tight_layout()
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        plt.close(fig)
        return filename
    return fig


def render_causal_tree(ct, filename_prefix='Plots/causal_tree', feature_names=None):
    """
    Render and save the causal tree using Matplotlib fallback only (Graphviz support removed).

    Returns the path to the saved PNG file.
    """
    Path('Plots').mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    try:
        fig = plot_causal_tree(ct, filename=None, feature_names=feature_names)
        # display inline if possible
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()

        png_path = f"{filename_prefix}_matplotlib.png"
        fig.savefig(png_path)
        plt.close(fig)
        print(f"Causal tree rendered and saved to '{png_path}' using Matplotlib")
        return png_path
    except Exception as e:
        print(f"Failed to render causal tree with Matplotlib: {e}")
        raise

