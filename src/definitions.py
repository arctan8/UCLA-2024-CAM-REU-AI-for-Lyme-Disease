import pandas as pd
from abc import ABC, abstractmethod

import constants
from constants import *

# Classes for Neuro Classification
class ComputeB(ABC):
    @abstractmethod
    def computeB(self):
        pass
        
class PNS(ComputeB):
    ''' Peripheral Nervous System (Neuropathy/Twitching)'''
    
    def computeB(self):
        '''
        Find B (1d array) indicating moderate/severe neuropathy or twitching
        Returns:
        B (pd.Series): B = 2 if mod/sev neuropathy/twitching, else B = 0, length = number of patients
        '''
        # U80_5_SXSVR_P2 Neuropathy
        # U80_6_SXSVR_P2 Twitching
        neuro_col_1 = ['U80_5_SXSVR_P2', 'U80_6_SXSVR_P2']
        # 1 if sum of cols exceeds 3 (mod-severe Neuropathy/Twitching)
        neuropathy_twitching = self.df[neuro_col_1].gt(3).any(axis=1).astype(int)

        B = 2 * neuropathy_twitching
        return B

class CNS(ComputeB):
    ''' Central Nervous System (Memory/Cognitive)'''

    def computeB(self):
        '''
        Find B (1d array) indicating moderate/severe neuropathy or twitching
        Returns:
        B (pd.Series): B = 2 if mod/sev memory/cognitive, else B = 0, length = number of patients
        '''
        # U80_7_SXSVR_P2 Memory Loss
        # U80_8_SXSVR_P2 Cognitive Impairment
        neuro_col_2 = ['U80_7_SXSVR_P2', 'U80_8_SXSVR_P2']
        # 1 if sum of cols exceeds 3 (mod-severe Neuropathy/Twitching)
        memory_cognitive = self.df[neuro_col_2].gt(3).any(axis=1).astype(int)
        
        B = 2 * memory_cognitive
        return B

class PNSCNS(ComputeB):
    ''' Combined Peripheral/Central Nervous System '''

    def computeB(self):
        '''
        Find B (1d array) indicating moderate/severe neuropathy/twitching or memory/cognitive
        Returns:
        B (pd.Series): B = twitching/neuropathy + memory/cognitive, length = number of patients
        '''
        # 7-15-24 Working Defn

        # U80_5_SXSVR_P2 Neuropathy
        # U80_6_SXSVR_P2 Twitching
        neuro_col_1 = ['U80_5_SXSVR_P2', 'U80_6_SXSVR_P2']
        # 1 if sum of cols exceeds 3 (mod-severe Neuropathy/Twitching)
        neuropathy_twitching = self.df[neuro_col_1].gt(3).any(axis=1).astype(int)

        # U80_7_SXSVR_P2 Memory Loss
        # U80_8_SXSVR_P2 Cognitive Impairment
        neuro_col_2 = ['U80_7_SXSVR_P2', 'U80_8_SXSVR_P2']
        # 1 if sum of cols exceeds 3 (mod-severe Neuropathy/Twitching)
        memory_cognitive = self.df[neuro_col_2].gt(3).any(axis=1).astype(int)

        B = neuropathy_twitching + memory_cognitive
        return B

# Classes For Mus Classification
class ComputeG(ABC):
    @abstractmethod
    def computeG(self):
        pass
        
class CombinedJM(ComputeG):
    
    def computeG(self):
        '''
        Find G (1d array) indicating moderate/severe joint pain or muscle aches
        Returns:
        G (pd.Series): G = 2 if mod/sev joint/muscle, else G = 0, length = number of patients
        '''
        # U80_4_SXSVR_P2 Muscle aches
        # U80_3_SXSVR_P2 Joint Pain
        
        mus_cols = ['U80_4_SXSVR_P2', 'U80_3_SXSVR_P2']
        combined = self.df[mus_cols].gt(3).any(axis=1).astype(int)
        G = combined * 2
        return G

class SeparateJM(ComputeG):
    
    def computeG(self):
        '''
        Find G (1d array) indicating moderate/severe joint pain or muscle aches
        Returns:
        G (pd.Series): G = mod/sev joint + muscle, length = number of patients
        '''
        # U80_4_SXSVR_P2 Muscle aches
        # U80_3_SXSVR_P2 Joint Pain
        muscle = self.df['U80_4_SXSVR_P2'].gt(3).astype(int)
        joint = self.df['U80_3_SXSVR_P2'].gt(3).astype(int)

        G = muscle + joint
        return G

''' Abstract Base Class for all ways to compute B & G ''' 
class ComputeBG(ComputeB, ComputeG):
    def __init__(self, data):
        self.df = data
    
    def computeBG(self):
        ''' Compute B and G using classification scheme '''
        return self.computeB(), self.computeG()

# Classes for various ways to label Neuro vs Musculo
class Comparison(ABC):
    def __init__(self, data, computebg):
        self.df = data
        self.computebg = computebg

    @abstractmethod
    def label(self):
        ''' Use ComputeBG instance to create neuro/mus labels '''
        
class InequalityComparison(Comparison):
    ''' Neuro: B > G, Mus: B < G, Both: B = G = 2, Neither: else'''
    def __init__(self, data, computebg):
        super.__init__(data, computebg)

    def label(self):
        '''
        Create Neuro, Musculo columns
        '''
        B, G = self.computebg.computeBG()
        
        # Create columns Neuro, Musculo
        self.df[NEURO] = ((B > G) | ((B == 2) & (G == 2))).astype(int)
        self.df[MUSCULO] = ((B < G) | ((B == 2) & (G == 2))).astype(int)
        
class EqualityComparison(Comparison):
    ''' Neuro: B = 2, Mus: G = 2, Both: B = G = 2, Neither: else'''
    def __init__(self, data, computebg):
        super.__init__(data, computebg)

    def label(self):
        B, G = self.computebg.computeBG()
        # Create columns Neuro, Musculo
        self.df[NEURO] = (B == 2).astype(int)
        self.df[MUSCULO] = (G == 2).astype(int)

class Definition(ComputeBG, Comparison): 
    def __init__(self, data):
        ComputeBG.__init__(self, data)        
        Comparison.__init__(self, data, self)
  
class OriginalWD(Definition, PNSCNS, SeparateJM, InequalityComparison):
    '''Original Working Definition'''        
    pass
    
class PNS1(Definition, PNS, CombinedJM, EqualityComparison):
    '''PNS Definition 1'''        
    pass
    
class PNS2(Definition, PNS, SeparateJM, InequalityComparison):
    '''PNS Definition 2'''        
    pass

class PNS3(Definition, PNS, SeparateJM, EqualityComparison):
    '''PNS Definition 3'''
    pass
    
class CNS1(Definition, CNS, CombinedJM, EqualityComparison):
    ''' CNS Definition 1'''
    pass
class CNS2(Definition, CNS, SeparateJM, InequalityComparison):
    ''' CNS Definition 2'''
    pass
class CNS3(Definition, CNS, SeparateJM, EqualityComparison):
    ''' CNS Definition 3'''
    pass
    

