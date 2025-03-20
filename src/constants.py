# constants for lymedata.py

CHRONIC = 'chronic'
ACUTE = 'acute'

NEURO = 'neuro'
NON_NEURO = 'non_neuro'
MUSCULO = 'musculo'
NON_MUSCULO = 'non_musculo'
BOTH = 'both'
NEITHER = 'neither'

# Order of Columns in Dataframe: diag_cir, addl_cir, symptoms
DIAG_CIR = 'diag_cir'
DIAG_CIR_COLS = tick_related_dict = {
    '*B270_TKBITE_P2': 'recall a tick bite',
    'B280_TKATTACH_P2': 'length of time noticed tick bite',
    '*B291_TKTX_P2': 'treated with antibiotics',
    'B300_TKTXDUR_P2': 'length of time treated for tick bite',
    'B461_DXTM_P2': 'period of time for diagnosis',
    '*B480_DXMISDX_P2': 'misdiagnosis',
    '*B580_DXCOIN_P2': 'tick born coinfection',
    'B590_1_DXCOINSP_P2': 'Babesia',
    'B590_2_DXCOINSP_P2': 'Bartonella',
    'B590_3_DXCOINSP_P2': 'Ehrlichia/ Anaplasma',
    'B590_4_DXCOINSP_P2': 'Mycoplasma',
    'B590_5_DXCOINSP_P2': 'Rickettsia'
}

ADDL_CIR = "addl_cir"
ADDL_CIR_COLS = {
    '*R200_SEX_P2': "Bio Sex",
    '*U1001_CTX_P2': "Antibiotics",
    'B250_DXTIMESINF_P2': "Times Infected",
    "U3011_TXGROC_P2": "GROC",
    "*U3191_BEDDAY_P2": "Bed Days",
    "U3170_MENTHLTH_P2": "Mental Health Days",
    "U3180_PHYSHLTH_P2": "Physical Health Days",
    "*U3040_EMPSTAT_P2": "Disability"
}

SYMPTOMS = 'symptoms'
SYMPTOMS_COLS = {
    "U80_1_SXSVR_P2": "Fatigue",
    "U80_2_SXSVR_P2": "Headache",
    "U80_3_SXSVR_P2": "Joint Pain",
    "U80_4_SXSVR_P2": "Muscle aches",
    "U80_5_SXSVR_P2": "Neuropathy",
    "U80_6_SXSVR_P2": "Twitching",
    "U80_7_SXSVR_P2": "Memory Loss",
    "U80_8_SXSVR_P2": "Cognitive Impairment",
    "U80_9_SXSVR_P2": "Sleep Impairment",
    "U80_10_SXSVR_P2": "Psychiatric",
    "U80_11_SXSVR_P2": "Heart related",
    "U80_12_SXSVR_P2": "Gastrointestinal",
} #"U80_13_SXSVR_P2": "Other"

CATG = "categorical"
CATG_COLS = {"B570_1_DXHCP_P2": "who diagnosed"}  

# Constants for definitions.py
DEF_OWD = "OWD" # original working definition
DEF_PNS1 = "PNS1"
DEF_PNS2 = "PNS2"
DEF_PNS3 = "PNS3"

DEF_CNS1 = "CNS1"
DEF_CNS2 = "CNS2"
DEF_CNS3 = "CNS3"

DEFNS = {DEF_OWD, DEF_PNS1, DEF_PNS2, DEF_PNS3, DEF_CNS1, DEF_CNS2, DEF_CNS3}
