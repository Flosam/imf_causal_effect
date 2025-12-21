## Create additional features in datasets
import pandas as pd
import numpy as np
from src.config import ECON_LAG, WB_INDICATORS
from src.utils import count_transitions, opec_dummy, weo_dummy

############################################################################
## feature functions

# Add binary variable representing if a country-year is under an imf program
def add_imf_prog(df:pd.DataFrame) -> pd.DataFrame:
    df['imf_prog'] = (df['agree_count'] >= 1).astype(int)
    return df


# add 1 lag and next period value for vdem v2x_polyarchy
def add_vdem_lags(df:pd.DataFrame) -> pd.DataFrame:
    df['v2x_polyarchy_l1'] = df.groupby('country_id')['v2x_polyarchy'].shift(1)
    df['v2x_polyarchy_n1'] = df.groupby('country_id')['v2x_polyarchy'].shift(-1)
    # add growth rate
    df['v2x_polyarchy_gr'] = np.log(df['v2x_polyarchy_n1']) - np.log(df['v2x_polyarchy'])
    return df


# add a count of previous transisitions into autocracy for each country-year
def add_num_aut_trans(df:pd.DataFrame) -> pd.DataFrame:
    # create dummy to differentiate autocracries from democracies
    df['is_autocracy'] = (df['v2x_regime_amb'] < 4).astype(int)

    # groupby and count the number of transitions from 0 to 1 in the dummy
    df = df.groupby("COWcode", group_keys=False).apply(count_transitions)

    return df


# compute additional variable from world bank data
def add_wb_vars(df:pd.DataFrame) -> pd.DataFrame:
    # total reserves to gross national income
    df['wbi_total_reserves2gni'] = df['wbi_total_reserves']/df['wbi_gni']
    # lof of gdp per capita
    df['l_wbi_gdp_pc'] = np.log(df['wbi_gdp_pc'])
    # total debt to gdp
    df['wbi_total_debt2gdp'] = df['wbi_total_debt']/df['wbi_gdp']
    # exchange rate depreciation
    df['wbi_xr_dep'] = df.groupby('wbi_id')['wbi_xr'].pct_change()

    # add lags
    econ_var = list(WB_INDICATORS.values()) + ['wbi_total_reserves2gni','l_wbi_gdp_pc','wbi_total_debt2gdp','wbi_xr_dep']
    for v in econ_var:
        for i in range(1,ECON_LAG+1):
            df[v+"_l"+str(i)] = df.groupby('country')[v].shift(i)

    return df


# add region dummy variables from wdi
def add_wb_region(df:pd.DataFrame) -> pd.DataFrame:
    df['wbi_region_name'] = df['region'].apply(lambda x: x['value'])
    df = df.drop(columns=['region'])

    # assign region to regionless countries
    df.loc[df['country_name']=='Republic of Vietnam','wbi_region_name'] = "East Asia & Pacific"
    df.loc[df['country_name']=='Taiwan','wbi_region_name'] = "East Asia & Pacific"
    df.loc[df['country_name']=="Yemen People's Republic",'wbi_region_name'] = "Middle East, North Africa, Afghanistan & Pakistan"
    df.loc[df['country_name']=="Zanzibar",'wbi_region_name'] = "Sub-Saharan Africa "
    df.loc[df['country_name']=="Zimbabwe",'wbi_region_name'] = "Sub-Saharan Africa "

    # assign Post-Communist as region to post communist countries
    post_communist_countries = ['Albania','Armenia','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria','Croatia',
                            'Czech Republic','Czechoslovakia','Estonia','Georgia','German Democratic Republic',
                            'Hungary','Kazakhstan','Kosovo','Kyrgyz Republic','Latvia','Lithuania','Macedonia','Moldova',
                            'Montenegro','Poland','Romania','Russia','Serbia','Slovakia','Slovenia','Tajikistan',
                            'Turkmenistan','Ukraine','Uzbekistan','Yugoslavia','German Democratic Republic']
    df.loc[df['country_name'].isin(post_communist_countries),'wbi_region_name'] = 'Post Communist'

    # create region dummies
    region_dummies = pd.get_dummies(df['wbi_region_name'], prefix='region').astype(int)
    
    # drop 1 dummy for base case
    region_dummies.drop(columns=['region_Sub-Saharan Africa'], inplace=True)
   
    df = pd.concat([df, region_dummies], axis=1)

    return df


# compute additional variables from Penn World Table
def add_pwt_vars(df:pd.DataFrame) -> pd.DataFrame:
    df['rgdpe_pc'] = df['rgdpe']/df['pop']
    df['l_rgdpe_pc'] = np.log(df['rgdpe_pc'])
    df['l_pl_c'] = np.log(df['pl_c'])
    df['l_pl_m'] = np.log(df['pl_m'])
    df['emp_rate'] = df['emp']/df['pop']

    df['ld_rgdpe_pc'] = df.groupby('countrycode')['l_rgdpe_pc'].diff()
    df['ld_pl_c'] = df.groupby('countrycode')['l_pl_c'].diff()
    df['ld_pl_m'] = df.groupby('countrycode')['l_pl_m'].diff()

    df.drop(columns=['l_pl_c','l_pl_m'], inplace=True)

    # add lags of the econ variables
    for c in df.columns[2:]:
        for i in range(1,ECON_LAG+1):
            df[c+"_l"+str(i)] = df.groupby('countrycode')[c].shift(i)

    return df


# add oil exporter dummy
def add_oil_export_dummy(df:pd.DataFrame) -> pd.DataFrame:
    # WDI rule: Fuel exports >= 33%
    df["oil_from_wbi"] = (df["wbi_fuel_export_share"] >= 33).astype(int)

    # OPEC fallback
    df["opec_oil"] = df.apply(opec_dummy, axis=1)

    # WEO fallback
    df["weo_oil"] = df["country_text_id"].apply(weo_dummy)

    # Final oil-exporter dummy
    df["oil_exporter"] = df[["oil_from_wbi","opec_oil","weo_oil"]].max(axis=1)

    # drop the three rules
    df.drop(columns=['oil_from_wbi','opec_oil','weo_oil'], inplace=True)

    return df


# add currency crash dummy from imf exchange rate data
def add_curr_crash_dummy(df:pd.DataFrame) -> pd.DataFrame:
    # compute depreciation rate and change in depreciation rate
    df = df.sort_values(['COUNTRY','year'])
    df['imf_xr_dep'] = df.groupby('COUNTRY')['imf_xr'].pct_change()
    df['imf_xr_dep_acc'] = df.groupby('COUNTRY')['imf_xr_dep'].diff()
    
    # create dummy 1 if country in crash
    df['imf_curr_crash'] = ((df['imf_xr_dep'] >= 0.3) & (df['imf_xr_dep_acc'] >= 0.1)).astype(int)
    
    return df


# add year dummies
def add_year_dummies(df:pd.DataFrame) -> pd.DataFrame:
    # create dummies
    year_dummies = pd.get_dummies(df['year']).astype(int)
    # drop 1 dummy for base case
    year_dummies = year_dummies.iloc[:,1:]
    df = pd.concat([df,year_dummies], axis=1)
    return df