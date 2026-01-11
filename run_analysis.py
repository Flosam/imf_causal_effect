import logging
logging.getLogger("shelved_cache").setLevel(logging.ERROR)
from pathlib import Path
import pandas as pd
from src.data_loader import *
from src.cleaning import *
from src.features import *
from src.config import INT_DATA,OUTCOME,TREATMENT,CONTROLS
from src.doubleML import dr_learner
from src.plots import *
from src.causalTree import CausalTree
from src.dataset import create_dataset

results_path='test1'
dataset_path='final_dataset.csv'

# Create dataset
if dataset_path:
    if Path(INT_DATA + dataset_path).is_file():
        main = pd.read_csv(INT_DATA + dataset_path)
    else:
        main = create_dataset()
        # Save cleaned data
        main.to_csv(INT_DATA + dataset_path, index=False)
else:
    main = create_dataset()

# Clean final dataset
data, controls = clean_main(main, CONTROLS, TREATMENT, OUTCOME)
print("Final dataset created")
plot_final_summary(data, 'Plots/final_summary.png')

# Run doubleml model
results_dr = dr_learner(
    X=data[controls].to_numpy(),
    y=data[OUTCOME].to_numpy(),
    w=data[TREATMENT].to_numpy()
)

# Run causal tree model
ct = CausalTree(max_depth=20, min_sample_leaf=20)
ct.fit(
    X=data[controls].to_numpy(),
    y=data[OUTCOME].to_numpy(),
    w=data[TREATMENT].to_numpy()
)
results_ct = ct.predict(data[controls].to_numpy())

# rejoin the results to clean data
data_results = data.copy()
data_results['dr_hte'] = results_dr
data_results['ct_hte'] = results_ct

# save data & results
data_results.to_csv('Data/results/'+ results_path + '.csv', index=False)
print("Results have been computed")

# Render causal tree (tries Graphviz, then Matplotlib fallback)
render_causal_tree(ct, filename_prefix='Plots/causal_tree', feature_names=controls)

# plot distribution
plot_hte_distribution(data_results, 'dr_hte', 'Plots/hte_distribution.png')
plot_hte_distribution(data_results, 'ct_hte', 'Plots/hte_distribution_ct.png')

# plot distribution against lagged v2x_polyarchy
plot_scatter_hte(data_results, 'Plots/hte_polyarchy.png', 'dr_hte', 'v2x_polyarchy')
plot_scatter_hte(data_results, 'Plots/hte_polyarchy_ct.png', 'ct_hte', 'v2x_polyarchy')