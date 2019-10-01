# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:34:12 2019

@author: Lluc Puig Codina
"""

import numpy as np

class EStest:
    """
    Implements the direct/unconditional Expected Shortfall test of Acerbi and 
    Szekely (2014). This test does not require the assumption of independence 
    for the realizations of the stochastic process describing portfolio returns
    
    Acerbi, C., and B. Szekely, "Backtesting Expected Shortfall", MSCI Inc., 
    December, 2014.
    """
    
    def __init__(self, X_obs, X, VaRLevel, VaR, ES, nSim, alpha = 0.05, 
                 print_results = True):
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
                        VaR must not be reported in its positive values, but
                        rather in its actual values, usually negative.
                        
        ES (np.array): Numpy array of size 1xT containing the projected
                       Expected Shortfall estimates for t = 1 to T at VaRLevel.
                       ES must not be reported in its positive values, but 
                       rather in its actual values, usually negative.

        nSim (int): Number of Monte Carlo simulations, scenarios, to obtain the
                    distribution of the statistic under the null hypothesis of 
                    P_t^[VaRLevel] = F_t^[VaRLevel].
                    
        alpha (float): Number in [0, 1] denoting the Type I error, the 
                       significance level threshold used to compute the 
                       critical value. Default set at 5%
                       
        print_results (boolean): Boolean for whether results should be printed. 
        """
        self.X_obs = X_obs
        self.T = X_obs.size
        self.X = X
        self.VaR = -VaR
        self.ES = -ES
        self.VaRLevel = VaRLevel
        self.nSim = nSim      
        self.alpha = alpha
        
        self.simulation()
        
        if print_results == True:
            print('----------------------------------------------------------------')
            print('   Direct/Unconditional Expected Shortfall Test by Simulation   ')
            print('----------------------------------------------------------------')
            print('Number of observations: ' + str(self.T))
            print('Number of VaR breaches: ' + str(self.VaR_breaches))
            print('Expected number of VaR breaches: ' + str(self.T*self.VaRLevel))
            print('ES Statistic: ' + str(self.Z_obs))
            print('Expected ES Statistic: ' + str(0))
            print('Critical Value at α = ' + str(self.alpha) + ': ' + str(self.critical_value))
            print('p-value: ' + str(self.p_value))
            print('Number of Monte Carlo simulations: ' + str(self.nSim))
            print('----------------------------------------------------------------')
        
        
    def statistic(self, X, I):
        """
        The statistic Z is defined as:
            Z = (1/(T*VaRLevel)) * \sum_{t=1}^T X_t*I_t/(ES_{VarLevel,t});    
            where I_t = 1(X_t + VaR_{VaRLevel,t}<0) is an indicator function 
            for whether VaR has been breached.     
        """
        return sum((X*I)/self.ES)/(self.VaRLevel*self.T) + 1
        
        
    def simulation(self):
        """
        Obtains the critical value and the p-value through Monte Carlo
        simulation. The p-value is defined as the fraction of scenarios for
        which the simulated test statistic is smaller than the test statistic 
        evaluated at the true portfolio return realizations:
            
            p-value = (1/M) * \sum_{s=1}^M 1(Z^s < Z^realized)
            
        The critical value is defined as the value of the statistic such that
        the p-value would be equal to alpha.
        """
        I_obs = (self.X_obs + self.VaR < 0) 
        self.VaR_breaches = I_obs.sum()
        self.Z_obs = self.statistic(self.X_obs, I_obs)       
        
        statistics = []
        statistic_breaches = []

        for i in range(self.nSim):
            X_i = self.X()
            I_i = (X_i + self.VaR < 0) 
            Z_i = self.statistic(X_i, I_i)
            statistics.append(Z_i)
            statistic_breach = Z_i < self.Z_obs
            statistic_breaches.append(statistic_breach)
        
        self.critical_value = np.quantile(statistics, self.alpha)
        self.p_value = np.mean(statistic_breaches)