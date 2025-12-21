## functions for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# histogram of the distribution of the treatment effect
def plot_hte_distribution(df:pd.DataFrame, path:str) -> None:
    # compute ate
    ate = df['te_w'].mean()
    # create plot
    plt.figure()
    plt.hist(df['te_w'], bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(ate, color='red', linestyle='--')
    plt.title("Distribution of Estimated CATE")
    plt.xlabel("Estimated Treatment Effect (CATE)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(path)

# scatter plot of treatment effect against another variable
def plot_scatter_hte(df:pd.DataFrame, path:str, var:str) -> None:
    plt.figure()
    sns.regplot(
        x=df[var],
        y=df["te_w"],
        lowess=True,
        scatter_kws={'alpha':0.3, 's':10},
        line_kws={'color': 'red'}
    )
    plt.xlabel(var)
    plt.ylabel("Estimated CATE")
    plt.grid(alpha=0.3)
    plt.savefig(path)
