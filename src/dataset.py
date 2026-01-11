import logging
logging.getLogger("shelved_cache").setLevel(logging.ERROR)
import pandas as pd
from src.data_loader import *
from src.cleaning import *
from src.features import *
from src.config import RAW_DATA,INT_DATA,WB_INDICATORS,FEATURES
from src.merge import merge_all

def create_dataset():
    # Load data
        imf_raw = load_imf(RAW_DATA + "imf_agreements/master_merge.dta")
        vdem_raw = load_vdem(RAW_DATA + "vdem/V-Dem-CY-Full+Others-v15.csv")
        gwf_raw = load_gwf(RAW_DATA + "GWF Autocratic Regimes 1.2/GWF Autocratic Regimes.xlsx")
        wb_raw = load_wb(WB_INDICATORS, cache_path=RAW_DATA+"world_bank_data.csv")
        pwt_raw = load_pwt(RAW_DATA + "pwt110.xlsx")
        imfxr_raw = load_imf_xr(RAW_DATA + "imf data/imf_xr.csv")
        mepv_raw = load_mepv(RAW_DATA + "mepv/MEPV2012ex.xls")
        print("Data has been loaded")

        # Clean data
        imf = clean_imf(imf_raw)
        vdem = clean_vdem(vdem_raw)
        gwf = clean_gwf(gwf_raw)
        wb = clean_wb(wb_raw)
        pwt = clean_pwt(pwt_raw)
        imfxr = clean_imf_xr(imfxr_raw)
        mepv = clean_mepv(mepv_raw)
        print("Data has been cleaned")

        # Add features
        imf = add_imf_prog(imf)
        vdem = add_vdem_lags(vdem)
        vdem = add_num_aut_trans(vdem)
        wb = add_wb_vars(wb)
        wb = add_wb_region(wb)
        pwt = add_pwt_vars(pwt)
        imfxr = add_curr_crash_dummy(imfxr)
        print("Data-specific features have been added")

        # Merge all datasets
        main = merge_all(imf,vdem,gwf,wb,pwt,imfxr,mepv)
        print('Data has been merged')

        # Add additional cross-datasets features
        main = add_oil_export_dummy(main)
        main = add_year_dummies(main)
        print('Cross-data features have been added')

        return main