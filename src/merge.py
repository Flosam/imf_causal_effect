## function to merge all datasets into one
import pandas as pd

## Function to build dataset
def merge_all(
        imf:pd.DataFrame,
        vdem:pd.DataFrame,
        gwf:pd.DataFrame,
        wb:pd.DataFrame,
        pwt:pd.DataFrame,
        imfxr:pd.DataFrame,
        mepv:pd.DataFrame
) -> pd.DataFrame:
    
    # merge IMF and vdem (treatment and outcome) to create main dataset
    main = imf.merge(vdem, on=['year','ccode_cow'], how='inner')

    # merge controls onto the main dataset
    main = merge_gwf(main, gwf)
    main = merge_wb(main, wb)
    main = merge_pwt(main, pwt)
    main = merge_imfxr(main, imfxr)
    main = merge_mepv(main, mepv)
    
    return main

## Functions to merge small datasets to main
def merge_gwf(main:pd.DataFrame, gwf:pd.DataFrame) -> pd.DataFrame:
    main = main.merge(gwf, on=['year','ccode_cow'], how='left')
    # fill empty rows with 0 for binary coding
    main['gwf_military'] = main['gwf_military'].fillna(value=0)
    main['gwf_monarch'] = main['gwf_monarch'].fillna(value=0)
    return main

def merge_wb(main:pd.DataFrame, wb:pd.DataFrame) -> pd.DataFrame:
    main = main.merge(wb, on=['year', 'country_text_id'], how='left')
    return main

def merge_pwt(main:pd.DataFrame, pwt:pd.DataFrame) -> pd.DataFrame:
    main = main.merge(pwt, on=['year','country_text_id'], how='left')
    return main

def merge_imfxr(main:pd.DataFrame, imfxr:pd.DataFrame) -> pd.DataFrame:
    main = main.merge(imfxr, on=['year','country_text_id'], how='left')
    main = main.drop(columns=['COUNTRY'])
    return main

def merge_mepv(main:pd.DataFrame, mepv:pd.DataFrame) -> pd.DataFrame:
    main = main.merge(mepv, on=['year', 'ccode_cow'], how='left')
    # fill empty values with 0 for binary coding (I checked manually the validity of this)
    main['ACTOTAL'] = main['ACTOTAL'].fillna(value=0)
    main = main.drop(columns=['SCODE','COUNTRY'])
    return main
