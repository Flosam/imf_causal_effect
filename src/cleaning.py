## clean all data sources
import pandas as pd
from src.config import START_DATE, END_DATE

# clean imf program data
def clean_imf(df:pd.DataFrame) -> pd.DataFrame:
    # keep only relevant columns
    cols = ['year','ccode_cow','cname_imf','country_syear','country_eyear','agree_count']
    df = df[cols]

    # remove years after end_date and before start_date
    df = df[(df['year'] <= END_DATE) & (df['year'] >= START_DATE)]

    # remove row when year is before country start or after country end
    df = df[(df['year'] >= df['country_syear']) & (df['year'] <= df['country_eyear'])]

    # remove small countries not in data
    countries = ['Abkhazia', 'Democratic Republic of Vietnam', 'South Ossetia']
    df = df[~df['cname_imf'].isin(countries)]

    # modify some country codes to match with other datasets
    df.loc[df['cname_imf']=='Serbia', 'ccode_cow'] = 345

    # causal unit is country-year-agreement. Change to country-year
    df = df.drop_duplicates()

    return df


# clean vdem data
def clean_vdem(df:pd.DataFrame) -> pd.DataFrame:
    # keep only relevant columns
    cols = ['year','country_text_id','country_id','COWcode','country_name','codingstart','codingend','v2x_polyarchy','v2x_regime_amb']
    df = df[cols]

    # keep country-years between their start and end dates
    df = df[(df['year'] >= df['codingstart']) & (df['year'] <= df['codingend'])]

    # add cow code to missing countries
    df.loc[df['country_name']=='Zanzibar', 'COWcode'] = 511

    # remove specific countries
    countries = ['Palestine/West Bank', 'Palestine/Gaza', 'Somaliland', 'Hong Kong']
    df = df[~df['country_name'].isin(countries)]

    # change specific country codes
    df.loc[df['country_name'].str.contains('Yugoslavia',case='False',na=False),'country_text_id'] = 'YUG'

    # remove NaNs in polyarchy
    df = df[~df['v2x_polyarchy'].isna()]

    return df


# clean gwf data
def clean_gwf(df:pd.DataFrame) -> pd.DataFrame:
    # keep years between their start and end dates
    df['gwf_startdate'] = pd.to_datetime(df['gwf_startdate'])
    df['gwf_enddate'] = pd.to_datetime(df['gwf_enddate'])
    df = df[(df['year'] >= df['gwf_startdate'].dt.year)
            & (df['year'] <= df['gwf_enddate'].dt.year)]
    
    # keep only relevant columns
    cols = ['year','cowcode','gwf_military','gwf_monarch']
    df = df[cols]

    # rename COW code column
    df = df.rename(columns={'cowcode':'ccode_cow'})

    return df


# clean world bank data
def clean_wb(df:pd.DataFrame) -> pd.DataFrame:
    # rename columns
    df = df.rename(columns={'date':'year','id':'country_code'})

    # make year as int
    df['year'] = df['year'].astype(int)

    return df


# clean penn world table
def clean_pwt(df:pd.DataFrame) -> pd.DataFrame:
    # keep only relevant columns
    cols = ['year','countrycode','rgdpe','pop','emp','pl_m','pl_c','xr']
    df = df[cols]

    return df


# clean imf exchange rate data
def clean_imf_xr(df:pd.DataFrame) -> pd.DataFrame:
    # rename columns
    df = df.rename(columns={'TIME_PERIOD':'year','OBS_VALUE':'imf_xr','COUNTRY.ID':'country_text_id'})

    # keep only relevant columns
    df = df.drop(columns=['INDICATOR.ID','INDICATOR'])

    # change some country codes to match dataset
    df.loc[(df['COUNTRY']=='Czechoslovakia'),'country_text_id'] = "CZE"
    df.loc[(df['COUNTRY']=='Yemen Arab Republic'),'country_text_id'] = "YEM"
    df.loc[(df['COUNTRY']=='Yugoslavia'),'country_text_id'] = "YUG"

    return df


# Clean Major Episodes of Political Violence (mepv) data
def clean_mepv(df:pd.DataFrame) -> pd.DataFrame:
    # keep only relevant columns
    cols = ['SCODE','CCODE','COUNTRY','YEAR','ACTOTAL']
    df = df[cols]

    # rename columns
    df = df.rename(columns={'CCODE':'ccode_cow','YEAR':'year'})

    # change some country codes to match with dataset
    df.loc[df['SCODE']=='SER','ccode_cow'] = 345
    df.loc[df['SCODE']=='VIE','ccode_cow'] = 816

    return df


