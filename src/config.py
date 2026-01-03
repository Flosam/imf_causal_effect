# Config file
START_DATE = 1971
END_DATE = 2007
ECON_LAG = 1

# features wanted for modelling
OUTCOME = 'v2x_polyarchy'
TREATMENT = 'imf_prog'
CONTROLS = ['num_aut_trans',
            'gwf_military',
            'gwf_monarch',
            'wbi_total_reserves2gni_l1',
            'l_rgdpe_pc_l1',
            'ld_rgdpe_pc_l1',
            'oil_exporter',
            'imf_curr_crash',
            'ACTOTAL']
FEATURES = [OUTCOME, TREATMENT] + CONTROLS



# data path
RAW_DATA = 'Data/raw/'
INT_DATA = 'Data/intermediate'

# World bank data indicators
WB_INDICATORS = {
    "TX.VAL.FUEL.ZS.UN": "wbi_fuel_export_share",
    "FI.RES.TOTL.CD": "wbi_total_reserves",
    "NY.GNP.MKTP.CD": "wbi_gni",
    "NY.GDP.MKTP.KD": "wbi_gdp",
    "NY.GDP.PCAP.KD": "wbi_gdp_pc",
    "PA.NUS.FCRF": "wbi_xr",
    "DT.DOD.DECT.CD":"wbi_total_debt",
    "DT.DOD.DECT.GN.ZS":"wbi_total_debt2gni"
}

# OPEC membership
OPEC_MEMBERSHIP = {
    "ALG": (1969, 2024),
    "ANG": (2007, 2023),
    "ECU": (1973, 1992), 
    "IRN": (1960, 2024),
    "IRQ": (1960, 2024),
    "KWT": (1960, 2024),
    "LBY": (1962, 2024),
    "NGA": (1971, 2023),
    "QAT": (1961, 2019),
    "SAU": (1960, 2024),
    "ARE": (1967, 2024),
    "VEN": (1960, 2024),
}

WEO_OIL_EXPORTERS = [
    "SAU","IRN","IRQ","KWT","ARE","QAT",
    "RUS","KAZ","AZE",
    "NGA","AGO","GAB",
    "VEN","ECU","MEX"
]

