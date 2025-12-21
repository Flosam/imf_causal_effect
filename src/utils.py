## utility functions
import pandas as pd
from src.config import OPEC_MEMBERSHIP, WEO_OIL_EXPORTERS

# count the number of transitions from democracy to autocracy
def count_transitions(subdf:pd.DataFrame) -> pd.DataFrame:
    subdf = subdf.sort_values("year").copy()
    
    # Lag autocracy state
    subdf["is_autocracy_lag"] = subdf["is_autocracy"].shift(1).fillna(0)
    
    # Transition occurs if previous year was NOT autocracy, current year IS autocracy
    subdf["autocracy_transition"] = (
        (subdf["is_autocracy"] == 1) &
        (subdf["is_autocracy_lag"] == 0)
    ).astype(int)
    
    # Cumulative count since 1946
    subdf["num_aut_trans"] = subdf["autocracy_transition"].cumsum()

    # drop transition and lag
    subdf.drop(columns=['is_autocracy_lag','autocracy_transition'], inplace=True)
    
    return subdf


# create dummy for opec members
def opec_dummy(row):
    iso = row["country_text_id"]
    yr = row["year"]
    if iso in OPEC_MEMBERSHIP:
        start, end = OPEC_MEMBERSHIP[iso]
        return int(start <= yr <= end)
    return 0

def weo_dummy(iso):
    return 1 if iso in WEO_OIL_EXPORTERS else 0