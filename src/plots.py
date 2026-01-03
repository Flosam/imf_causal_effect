## functions for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# histogram of the distribution of the treatment effect
def plot_hte_distribution(df:pd.DataFrame, path:str) -> None:
    # compute ate
    ate = df['hte'].mean()
    # create plot
    plt.figure()
    plt.hist(df['hte'], bins=40, edgecolor='black', alpha=0.7)
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
        y=df["hte"],
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
