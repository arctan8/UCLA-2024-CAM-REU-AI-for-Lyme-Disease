Symptoms:
    Fatigue: U80_1_SXSVR_P2
    Headache: U80_2_SXSVR_P2
    Joint Pain: U80_3_SXSVR_P2
    Muscle aches: U80_4_SXSVR_P2
    Neuropathy: U80_5_SXSVR_P2
    Twitching: U80_6_SXSVR_P2
    Memory Loss: U80_7_SXSVR_P2
    Cognitive Impairment: U80_8_SXSVR_P2
    Sleep Impairment: U80_9_SXSVR_P2
    Psychiatric: U80_10_SXSVR_P2
    Heart related: U80_11_SXSVR_P2
    Gastrointestinal: U80_12_SXSVR_P2
    Other: U80_13_SXSVR_P2
    
Symptoms Values: 1-7
Symptom NaN Handling: Any row with NaN values in the columns determing Neuro/Musculo status is dropped. Patients answering less than 8 questions can be filtered and NaN filled with 0 depending on flag drop_skipped_8 (default True).

Diagnostic Circumstances:
    recall a tick bite: *B270_TKBITE_P2
    length of time noticed tick bite: B280_TKATTACH_P2
    treated with antibiotics: *B291_TKTX_P2
    length of time treated for tick bite: B300_TKTXDUR_P2
    period of time for diagnosis: B461_DXTM_P2
    misdiagnosis: *B480_DXMISDX_P2
    tick born coinfection: *B580_DXCOIN_P2
    Babesia: B590_1_DXCOINSP_P2
    Bartonella: B590_2_DXCOINSP_P2
    Ehrlichia/ Anaplasma: B590_3_DXCOINSP_P2
    Mycoplasma: B590_4_DXCOINSP_P2
    Rickettsia: B590_5_DXCOINSP_P2

Diagnostic Circumstances Values:
    All Binary except time variables: 1 yes, 2 no, 99 unknown
    
    Time variables, binned to binary:  
    length of time noticed tick bite: B280_TKATTACH_P2
        1: < 12hr, 2: < 24hr, 3: < 36hr, 4: < 48hr, 5: > 48hr, 99: unknown
        Binning: 5 -> 1, else -> 0
        
    length of time treated for tick bite: B300_TKTXDUR_P2
        1: 1 or 2 days, 2: <= 2wks, 3 : <= 3wks, 4: <= 4wks, 5: >= 4wks, 99: unknown
        Binning: 5 -> 1, else -> 0
    
    period of time for diagnosis: 
        1: < 1mo, 2: 1-3mo, 3: 4-12mo, 4: 13-24mo, 5: 25-36mo, 6: >3yr, 99:unknown
        Binning: < 3 -> 0, else -> 1

Diagnostic Circumstances Handling: NaN values are filled with 0. 99 values may be dropped depending on flag drop_99.
        
Additional Circumstances:
    Male/Female:
        bio sex: *R200_SEX_P2
        
    Chronic/Acute:
        symptoms at diagnosis: *B561_1_DXSTAGE_P2
        
    Antibiotics/No Antibiotics:
        treatment approach: *U1001_CTX_P2
        
        Antibiotics-Only Q:
        antibiotic results: U20_1_DXABXRESP_P2
        other textbox: U20_2_DXABXRESP_P2

        follow up antibiotics: U30_1_DXABXADDL_P2
        other textbox: U30_2_DXABXADDL_P2 
        
        No Antibiotics-Only Q:
        if not antibiotics why not (U1011):
        never effective: U1011_1_NOABX_P2
        no longer effective: U1011_2_NOABX_P2
        alternative treatment: U1011_3_NOABX_P2
        no access to doctor who treats: U1011_4_NOABX_P2
        other non-insurance cost considerations: U1011_5_NOABX_P2
        insurance company constraints: U1011_6_NOABX_P2
        treatment break/dr protocol: U1011_7_NOABX_P2
        side effects: U1011_8_NOABX_P2
        personal choice: U1011_9_NOABX_P2
        well/remission: U1011_10_NOABX_P2
        other: U1011_11_NOABX_P2
            

    who diagnosed (B570): B570_1_DXHCP_P2
    times infected: B250_DXTIMESINF_P2
    
    Other Medications:
    currently used other meds (U2151):
        IVIG: U2151_1_MEDO_P2
        Sleep medications: U2151_2_MEDO_P2
        Psychiatric medications: U2151_3_MEDO_P2
        Antidepressants: U2151_4_MEDO_P2
        Thyroid medications: U2151_5_MEDO_P2
        Steroids: U2151_6_MEDO_P2
        Anti-inflammatories: U2151_7_MEDO_P2
        Prescription pain relievers: U2151_8_MEDO_P2
        Over the counter pain relievers: U2151_9_MEDO_P2
        Headache medications: U2151_10_MEDO_P2
        Stomach medications: U2151_11_MEDO_P2
        Seizure medication: U2151_12_MEDO_P2
        Other: U2151_13_MEDO_P2
        Other TEXT BOX: U2151_14_MEDO_P2
        Claritin (Loratadine): U2151_15_MEDO_P2
        Low dose Naltrexone: U2151_16_MEDO_P2
        Nerve pain/anti-convulsant: U2151_17_MEDO_P2
        Tagamet (Cimetidine): U2151_18_MEDO_P2
        None of these: U2151_19_MEDO-P2
        
    Quality of Life Measures:
    disability status (U3050):
        stopped work: U3050_1_EMPCHG_P2 
        reduced work: U3050_2_EMPCHG_P2 
        changed work: U3050_3_EMPCHG_P2 
        no change: U3050_4_EMPCHG_P2 
        
    bed days: *U3191_BEDDAY_P2    
    mental health days: U3170_MENTHLTH_P2
    physical health days: U3180_PHYSHLTH_P2
    
    special equipment (pre-req to U3140): U3130_EQ_P2
    
    Global rating of change from treatment start(GROC, U3011):
        symptoms change: U3011_TXGROC_P2 
        1: worsen, 2: unchanged, 3: better
        Binning: 1,2 -> 0, 3-> 1
        
    
Additional Circumstances Values:
    bio sex: R200_SEX_P2
        1: male, 2: female, 3: other
        
    times infected: B250_DXTIMESINF_P2
        1-4: # of times infected w/ Lyme, 99: unknown
        Binning: 1 -> 0, else -> 1
    
    symptoms at diagnosis: B561_1_DXSTAGE_P2
        1: asymptomatic w/ bite, 2: EM rash, 3: early, 4: late untreated, 5: other, 6: chronic
        Preprocessing: 0: acute, 2: chronic
    
    treatment approach: *U1001_CTX_P2
        1: only antibiotics (skip to U-1020)
        2: antibiotics and alternatives (skip to U-1020)
        3: only alternatives (skip to U-1011)
        4: none (skip to U-1011)
        
    who treated: B570_1_DXHCP_P2
        type of healthcare provider that first diagnosed Lyme
        1: general practitioner, 2: internist, 3: infectious disease specialist, 4: rheumatologist, 5: lyme specialist, 6: pediatrician, 7: other, 99: unknown
        Categorical circumstance
    
    bed days: U3191_BEDDAY_P2
        in bed for 1/2 day or more in last 30 days
        1: 0dy, 2-31: 1-30dy
        Binning: >=15 -> 1, else -> 0
        
    mental health days: U3170_MENTHLTH_P2
        poor mental health due to Lyme in last 30 days
        1: 0dy, 2-31: 1-30dy
        Binning: >=15 -> 1, else -> 0

    physical health days: U3180_PHYSHLTH_P2
        poor physical health due to Lyme in last 30 days
        1: 0dy, 2-31: 1-30dy
        Binning: >=15 -> 1, else -> 0
    
    antibiotics results: U20_1_DXABXRESP_P2
        1: asymptomatic before treatment
        2: resolved fully
        3: resolved then recurred
        4: resolved partially then improved over time w/o further treatment
        5: persisted
        6: other
        99: unknown
        
    follow up antibiotics: U30_1_DXABXADDL_P2
        1: additional antibiotics
        2: no additional antibiotics
        3: other
        99: unknown
        
    GROC:
        symptoms change: U3011_TXGROC_P2 
        1: worse (skip to U-3020)
        2: unchanged (skip to U-3040)
        3: better (skip to U-3030)
        
        health decline: U3020_GROCWRS_P2
        1: almost, 2:hardly, 3: little, 4: somewhat, 5: moderate, 6: good deal, 7: great deal, 8: very great deal 
        Not included in analysis
        
        health improvement: U3030_GROCBTR_P2
        1: almost, 2:hardly, 3: little, 4: somewhat, 5: moderate, 6: good deal, 7: great deal, 8: very great deal 
        Not included in analysis