# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:34:12 2019

@author: Lluc Puig Codina
"""

import numpy as np

class EStest:
    """
    Implements the direct test of Expected Shortfall of Acerbi and Szekely
    (2014). The test does not require the assumption of independence for the 
    realizations of the stochastic process X_t which describes the portfolio
    return.
    """
    
    def __init__(self, X_obs, X, VaRLevel, VaR, ES, nSim):
        """        
        X_obs (np.array): Numpy array of size 1xT containing the actual 
                          realization of the portfolio return.
                          
        X (function): Function that simulates (outputs) a realization of the 
                      portfolio return under H0: 
                      X^s = (X_1^s, X_2^s,..., X_T^s), with X_t^s ~ P_t for all
                      t = 1 to T.
                      Output should be a numpy array of 1xT.
                      
        VaRLevel (float): Number describing the level of the VaR, say 0.05 for 
                          95% or 0.01 for 99%.
        
        VaR (np.array): Numpy array of size 1xT containing the projected 
                        Value-at-Risk estimates for t = 1 to T at VarLevel.
                        
        ES (np.array): Numpy array of size 1xT containing the projected
                       Expected Shortfall estimates for t = 1 to T at VaRLevel.

        nSim (int): Number of Monte Carlo simulations, scenarios, to obtain the
                    distribution of the statistic under the null hypothesis of 
                    P_t^[VaRLevel] = F_t^[VaRLevel].
        """
        self.X_obs = X_obs
        self.T = X_obs.size
        self.X = X
        self.VaR = -VaR
        self.ES = -ES
        self.VaRLevel = VaRLevel
        self.nSim = nSim        
        
        self.Z_obs, self.VaR_breaches, self.p_value = self.significance()
        
        print('--------------------------------------------------')
        print('   Expected Shortfall Direct Test by Simulation   ')
        print('--------------------------------------------------')
        print('Number of VaR breaches: ' + str(self.VaR_breaches))
        print('Expected number of VaR breaches: ' + str(self.T*self.VaRLevel))
        print('ES Statistic: ' + str(self.Z_obs))
        print('Expected ES Statistic: ' + str(0))
        print('P-value: ' + str(self.p_value))
        print('--------------------------------------------------')
        
        
    def statistic(self, X, I):
        """
        The statistic Z is defined as:
            Z = (1/(T*VaRLevel)) * \sum_{t=1}^T X_t*I_t/(ES_{VarLevel,t});    
            where I_t = 1(X_t + VaR_{VaRLevel,t}<0) is an indicator function 
            for whether VaR has been breached.     
        """
        return (1/(self.VaRLevel*self.T)*sum((X*I)/self.ES) + 1)
        
        
    def significance(self):
        """
        Obtains the p-value defined as the fraction of scenarios for which the
        simulated test statistic is smaller than the test statistic evaluated
        at the true PnL realizations:
            
            p-value = (1/M) * \sum_{s=1}^M 1(Z^s < Z^realized)
        """
        I_obs = (self.X_obs + self.VaR > 0) 
        Z_obs = self.statistic(self.X_obs, I_obs)
        
        statistic_breaches = []

        for k in range(self.nSim):
            X_i = self.X
            I_i = (X_i + self.VaR > 0) 
            Z_i = self.statistic(X_i, I_i)
            statistic_breach = all(Z_i < Z_obs)
            statistic_breaches.append(statistic_breach)
        
        VaR_breaches = sum(I_obs) 
        p_value = np.mean(statistic_breaches)
        
        return Z_obs, VaR_breaches, p_value    