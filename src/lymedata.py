import pandas as pd
import zipfile
from io import BytesIO
import io
import msoffcrypto
import numpy as np
import json

from functools import partial

import constants
from constants import *

import definitions
from definitions import *

# Bug-free as of 3-8-25

class LymeData:    
    def __init__(self, select_rows: set, select_cols: set, labels: set, *args, **kwargs):
        '''
        Initialize LymeData class with specifed rows and columns. 
        Specifying one member of pairs (chronic/acute or neuro/musculo) selects appropriate data.
        Specifying both members of a pair selects data that satisfies either criteria.
        Labels must be a subset of select_rows: can be chronic/acute, neuro/musculo, neuro/non-neuro, musculo/non-musculo

        Definition must be one of OWD, PNS 1-3, CNS 1-3.
        
        Parameters:
        select_rows (set): select only chronic/acute/neuro/musculo groups or some combination
        select_cols (set): select diagnostic circumstances/symptoms or both
        labels (set): select chronic/acute or neuro/musculo or neuro/non-neuro or musculo/non-musculo
        individual_cols (set, optional): names of individual columns to be included for customization.
        defn (const, optional): definition of neuro/musculo. default=original
        
        Returns: None
        
        '''
        self.df = self.load_data()
        # self.config = self.load_config("config.json")
        self.config = self.load_config('/home/reu24lyme/code_lib/src/config.json')
        
        assert len(labels) == 2 or len(labels) == 4, "Please specify 2 or 4 labels!"
        assert NEURO in select_rows or MUSCULO in select_rows, 'Both Neuro and Musculo patients omitted from dataset!'
        assert CHRONIC in select_rows or ACUTE in select_rows, 'Both Chronic and Acute omitted from dataset!'
        
        if NEURO in select_rows and MUSCULO in select_rows:
            assert labels == {NEURO, MUSCULO} or labels == {NEURO, MUSCULO, BOTH, NEITHER} or labels == {CHRONIC, ACUTE}
        elif NEURO in select_rows:
            assert labels == {NEURO, NON_NEURO} or labels == {CHRONIC, ACUTE}
        else: # musculo only
            assert labels == {MUSCULO, NON_MUSCULO} or labels == {CHRONIC, ACUTE}

        if 'defn' in kwargs:
            assert kwargs['defn'] in DEFNS
            self.defn = kwargs['defn']
        else:
            self.defn = DEF_OWD
            
        self.labels = labels
        self.select_data(select_rows, select_cols, **kwargs)
        
    def load_data(self):
        '''
        Imports preproccessed data.
        '''
        password = b'password'
        with open('/home/reu23/DATA/Preprocessing.zip', 'rb') as file:
            zip_buffer = BytesIO(file.read())
        with zipfile.ZipFile(zip_buffer, 'r', zipfile.ZIP_LZMA) as zip_file:
            zip_file.setpassword(password)
            with zip_file.open('preprocessing.csv', 'r') as csv_file:
                # Read the CSV data into a DataFrame
                df = pd.read_csv(csv_file)
        df = df.copy()
        return df
    
    def load_config(self, file_path):
        '''
        Loads JSON config file containing settings on how to bin and clean data.
        It returns a dictionary whose keys are the type of data (eg diag_cir, symptoms, etc) and values are dictionaries specifying how data should be binned/cleaned.
        
        Parameters:
        file_path (string): JSON config file
        Returns:
        config (dict): contains key=data group (eg diag_cir), val=data_cleaning_settings (dict)
        '''
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
            return config
        
    def select_data(self, select_rows, select_cols, **kwargs):
        '''
        Filter DataFrame to only select certain groups, eg Chronic + Diag Cir or Acute&Chronic+Symptoms
        Any row with NaN values in the columns determing Neuro/Musculo status is dropped.
        
        For diagnostic circumstances, NaN values are filled with 0, 99 values may be dropped depending on flag.
        
        For symptoms, NaN values are filled with 99 by default. Patients answering less than 8 questions can be filtered and NaN filled with 0 depending on flag.
        
        Parameters:
        select_rows (list): select only chronic/acute/neuro/musculo groups or some combination
        select_cols (list): select diagnostic circumstances/symptoms or both        
        individual_cols (set, optional): names of individual columns to be included for customization.
        
        All Data:
        drop_99 (bool, optional): Drop any patients with 99 values for all questions except categorical, default=True
        
        Symptoms:
        drop_skipped_8 (bool, optional): Drop any patients that answered less than 8 questions (NaN values) for symptoms and fill NaN with 0. default=False
        
        
        Return: None
        '''
        self.drop_99 = kwargs.get('drop_99', self.config['all_data']['drop_99'])
        self.drop_skipped_8 = kwargs.get('drop_skipped_8', self.config['symptoms']['drop_skipped_8'])
        
        # Filter unwell patients
        self.df = self.df[self.df['*B641_STATWELLSK_P2']== 0]
        self.df = self.df.drop(['*B641_STATWELLSK_P2'],axis=1) # remove column indicating well/unwell

        # Row handling, Acute or Chronic Filtering
        if ACUTE not in select_rows:
            self.df = self.df[self.df['B650_1_STATSTAGE_P2']== 2] # filter chronic
        elif CHRONIC not in select_rows:
            self.df = self.df[self.df['B650_1_STATSTAGE_P2']== 0] # filter acute
                    
        # Neuro or Musculo Filtering
        self.add_neuro_musculo_labels()
        self.get_counts()
        
        if MUSCULO not in select_rows and self.labels != {NEURO, NON_NEURO}: # Neuro only
            print('Filter neuro patients only!')
            self.df = self.df[self.df[NEURO] == 1]
            
        elif NEURO not in select_rows and self.labels != {MUSCULO, NON_MUSCULO} : # Musculo only
            print('Filter musculo patients only!')
            self.df = self.df[self.df[MUSCULO] == 0]
        
        # Label handling
        if self.labels == {NEURO, MUSCULO}:
            pass
        
        elif self.labels == {CHRONIC, ACUTE}:
            # add chronic/acute labels
            self.df[CHRONIC] = (self.df['B650_1_STATSTAGE_P2'] == 2).astype(int)
            self.df[ACUTE] = (self.df['B650_1_STATSTAGE_P2'] == 0).astype(int)
            self.df.drop([NEURO, MUSCULO], axis=1)
                
        elif self.labels == {NEURO, NON_NEURO}:
            # Add non-neuro label:
            assert not {BOTH, NEITHER}.intersection(self.df.columns), "Dataframe contains BOTH or NEITHER columns! Incorrect for NEURO vs NON_NEURO!"
            self.df.drop([MUSCULO], axis=1, inplace=True)
            self.df[NON_NEURO] = (self.df[NEURO] == 0).astype(int)
            
                        
        elif self.labels == {MUSCULO, NON_MUSCULO}:
            # Add non-musculo label:
            assert not {BOTH, NEITHER}.intersection(self.df.columns), "Dataframe contains BOTH or NEITHER columns! Incorrect for MUSCULO vs NON_MUSCULO!"            
            self.df.drop([NEURO], axis=1, inplace=True)
            self.df[NON_MUSCULO] = (self.df[MUSCULO] == 0).astype(int)
            
        
        # Column handling (diag_cr, symptoms)
        dataframes = [] # list of dataframes to be concatenated
        
        if DIAG_CIR in select_cols:
            # Select diag_cir columns only:
            diag_cir = list(DIAG_CIR_COLS.keys())
                
            diag_cir_df = self.df[diag_cir].fillna(0) # No -1 values, safe to replace NA with 0
             
            # Replace 2 with 0 in the specified columns using the dictionary
            diag_cir_df = diag_cir_df.replace(self.get_replace_dict('diag_cir'))           
             
            # Binning for Diagnostic Circumstances Time Variables
            diag_cir_df = self.binning(diag_cir_df, 'diag_cir')
    
            coin_df = self.df.filter(like="B590").fillna(0).drop(['B590_6_DXCOINSP_P2'], axis=1)
            
            common_columns = list(set(diag_cir_df.columns).intersection(set(coin_df.columns)))
            coin_df.drop(columns=common_columns, inplace=True)

            combined_df = pd.concat([diag_cir_df, coin_df], axis=1)
            
            combined_df.rename(columns=DIAG_CIR_COLS, inplace=True) # Rename columns to english
            dataframes.append(combined_df)
         
        
        if ADDL_CIR in select_cols:
            addl_cir = list(ADDL_CIR_COLS.keys())
                
            addl_cir_df = self.df[addl_cir].dropna()
            addl_cir_df = addl_cir_df.replace(self.get_replace_dict('addl_cir'))
            addl_cir_df = self.binning(addl_cir_df, 'addl_cir')
            
            addl_cir_df.rename(columns=ADDL_CIR_COLS, inplace=True)
            dataframes.append(addl_cir_df)
                              
        if SYMPTOMS in select_cols:
            data = self.df.copy()
            symptoms_df = data.filter(like="U80") # symptom columns
            symptoms_df = symptoms_df.drop('U80_13_SXSVR_P2', axis=1) # drop 'Other' column
            
            if self.drop_skipped_8:
                num_skipped_q = symptoms_df.isnull().sum(axis=1)
                symptoms_df = symptoms_df[num_skipped_q < 8]
                symptoms_df = symptoms_df.fillna(0)
            else:
                print('Caution! Some symptoms may have 99 values, affecting Neuro/Musculo label')
                symptoms_df = symptoms_df.fillna(99)
                
            symptoms_df.rename(columns=SYMPTOMS_COLS, inplace=True)
            dataframes.append(symptoms_df)
        
        if CATG in select_cols:
            for cir in self.config['categorical']['cols']:
                cir_df = self.df[cir]
                
                if self.config['categorical']['drop_99']:
                    cir_df = cir_df[cir_df != 99]
                else:
                    print('Caution! 99 values are not dropped from categorical questions: Zero rows in one-hot-encoded columns may occur')
                    
                cir_dict = self.config['categorical']['cols'][cir]
                cir_df = cir_df.map({v: k for k,v in cir_dict.items()})
                
                one_hot_df = pd.get_dummies(cir_df).astype(int)
                dataframes.append(one_hot_df)
                
        for frame in dataframes:
            if(frame.isnull().values.any()):
                print('++++++++++++')
                rows_with_nan_any = frame[frame.isna().any(axis=1)]
                print(rows_with_nan_any)
                raise Exception('DataFrame has NaN value! Line 250')
                
        dataframes = pd.concat(dataframes, axis=1)
        # patients (rows) that are present in one dataframe but not others are dropped
        dataframes.dropna(inplace=True)
        
        # Drop 99 values
        if self.drop_99:
            print('Dropping 99')
            dataframes = dataframes[~dataframes.isin([99]).any(axis=1)]
        else:
            print('Columns with 99')
            print(dataframes.columns[(dataframes == 99).any()].tolist())
            
        # Include only columns specified by individual_cols optional kwarg
        indiv_cols = kwargs.get('individual_cols', None)
        
        if indiv_cols is not None: 
            df_cols = set(dataframes.columns)
            if indiv_cols.issubset(df_cols):
                dataframes = dataframes[list(indiv_cols)] 
            else:
                raise AssertionError('Specified Columns Not Found in DataFrame: ' + str(indiv_cols - df_cols))

        # Add Labels
        for label in self.labels:
            dataframes[label] = self.df[label]
            
        if dataframes.isnull().values.any():
            raise TypeError('Null values detected in the following columns: ', dataframes.columns[dataframes.isna().any()].tolist())
        
        if (dataframes == -1).any().any():
            raise TypeError('-1 values detected in the following columns: ', dataframes.columns[(dataframes == -1).any()].tolist())

        # Ensure that column order is correct
        columns = list(self.df.columns)
        for label in self.labels: # remove and then add on label in correct order 
            columns.remove(label)
        for label in self.labels:
            columns.append(label)
        
        self.df = dataframes
    
    def binning(self, dataframe, data_category):
        '''
        Bin values in the columns of the dataframe as specified in config.json
        for this data category
        Parameters:
        data_category (string): question type
        Returns:
        dataframe (DataFrame): modified dataframe
        '''
        bin_dict =  self.config[data_category]["binning"]
        
        def replace(data_val, bin_val):
            if data_val == 99: return data_val
            elif data_val < bin_val: return 0
            else: return 1 # data_val >= bin_val
            
        for col, v in bin_dict.items():
            repl = partial(replace, bin_val=v)
            dataframe[col] = dataframe[col].apply(repl)
        return dataframe
    
    def get_replace_dict(self, data_category):
        '''
        Find the replacement dictionary in config.json for this data category
        Parameters:
        data_category (string): question type (diag_cir, adl_cir)
        Returns:
        repl_dict (dict): key=colname, val={old_val:new_val}
        '''
        repl_dict = self.config[data_category]["replacement_dict"]
        # convert json dict to python dict
        repl_dict = {col: {int(k): v for k,v in d.items()} for col, d in repl_dict.items()}
        return repl_dict
    
    def add_neuro_musculo_labels(self):
        if self.defn == DEF_OWD:
            definition = OriginalWD(self.df)
        elif self.defn == DEF_PNS1:
            definition = PNS1(self.df)
        elif self.defn == DEF_PNS2:
            definition = PNS2(self.df)
        elif self.defn == DEF_PNS3:
            definition = PNS3(self.df)
        elif self.defn == DEF_CNS1:
            definition = CNS1(self.df)
        elif self.defn == DEF_CNS2:
            definition = CNS2(self.df)
        elif self.defn == DEF_CNS3:
            definition = CNS3(self.df)
            
        definition.label()
        
        # Add Both, Neither columns
        if self.labels == {NEURO, MUSCULO, BOTH, NEITHER}:
            self.df[BOTH] = self.df.apply(lambda row: 1 if row[NEURO] == 1 and row[MUSCULO] == 1 else 0, axis=1)
            self.df[NEITHER] = self.df.apply(lambda row: 1 if row[NEURO] == 0 and row[MUSCULO] == 0 else 0, axis=1)
                                             
            # Replace double 1's and double 0's in neuro and musculo cols with 1's in both/neither
            self.df.loc[self.df[BOTH]==1, [NEURO, MUSCULO]] = 0
            self.df = self.df.dropna(subset=[NEURO, MUSCULO, BOTH, NEITHER])
            return
            
        # Drop any row which has NaN values in NEURO, MUSCULO columns
        self.df = self.df.dropna(subset=[NEURO, MUSCULO])
        
    def get_data_and_labels(self):
        '''
        Separate the dataframe into matrices of patients vs features and patients vs labels.
        If 99 values are not dropped, also returns matrix of missing data.
        
        Returns:
        data (np.array): patients vs features
        labels (np.array): patients vs labels
        missing_data (np.array, optional): patients vs features, binary: 1 if data present, 0 if missing.
        '''
        dataframe = self.df.copy()
        labels = dataframe[list(self.labels)].to_numpy()
        data = dataframe.drop(columns=self.labels).to_numpy()
        # Line 354 PROBLEM SO WHERES THE PROBLEM???
        
        if not self.drop_99:
            missing_data = (data != 99).astype(int)
            return data, labels, missing_data
        
        else:
            return data, labels
        
    def drop_one_label(self, label_name):
        '''
        Drop specified label from dataframe.
        
        Parameters:
        col (string): column name
        '''
        try:
            self.df = self.df.drop(label_name, axis=1)
            self.labels.remove(label_name)
        except KeyError:
            print('Label ', label_name,' is not in self.df.')

    def get_counts(self):
        '''
        Print out all counts of Neuro/Musculo/both/neither
        '''
        neuro = ((self.df[NEURO] == 1) & (self.df[MUSCULO] == 0)).sum()
        musculo = ((self.df[NEURO] == 0) & (self.df[MUSCULO] == 1)).sum()
        
        if self.labels == {NEURO, MUSCULO, BOTH, NEITHER}:
            both = (self.df[BOTH] == 1).sum()
            neither = (self.df[NEITHER] == 1).sum()
        else:       
            both = ((self.df[NEURO] == 1) & (self.df[MUSCULO] == 1)).sum()
            neither = ((self.df[NEURO] == 0) & (self.df[MUSCULO] == 0)).sum()
            
        print(f'Both Neuro and Mus: {both}')
        print(f'Only Neuro: {neuro}')
        print(f'Only Mus: {musculo}')
        print(f'Neither Neuro nor Mus: {neither}')
    