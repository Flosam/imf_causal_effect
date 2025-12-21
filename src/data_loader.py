## data loader
import pandas as pd
import wbdata
import time
from pathlib import Path

# import IMF program data
def load_imf(path:str) -> pd.DataFrame:
    return pd.read_stata(path)

# import vdem data
def load_vdem(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

# import gwf autocracy data
def load_gwf(path:str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name='TSCS data')

# import world bank data
def load_wb(indicators: dict[str:str], cache_path=None) -> pd.DataFrame:
    # check if data has been retrieved before
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)
    
    max_retries = 5
    # import data
    for attempt in range(max_retries):
        try:
            wdi_raw = wbdata.get_dataframe(indicators)
            if wdi_raw.empty:
                raise ValueError("Empty WB response")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2*(attempt+1))

    wdi_raw = wdi_raw.reset_index().rename(columns={"country":"country_name"})

    # import country code and region to match on country name
    wdi_codes = pd.DataFrame(wbdata.get_countries())
    wdi_codes = wdi_codes.rename(columns={"name":"country_name"})[["id","country_name","region"]]
    
    # merge data and country code together to assign country_code to each country
    wdi = wdi_raw.merge(wdi_codes,
                        on="country_name",
                        how='inner')
    
    # store to cache_path
    if cache_path:
        wdi.to_csv(cache_path, index=False)
    
    return wdi


# import penn world table variables
def load_pwt(path:str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name='Data')

# import IMF exchange rate data
def load_imf_xr(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

# import political violence index data
def load_mepv(path:str) -> pd.DataFrame:
    return pd.read_excel(path)

