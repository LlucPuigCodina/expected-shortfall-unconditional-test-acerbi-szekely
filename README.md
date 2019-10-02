
# expected-shortfall-unconditional-test-acerbi-szekely

`Python` implementation of the Direct Expected Shortfall Test of [Acerbi and Szekely (2014)](https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b) by Lluc Puig Codina


```python
import numpy as np
import scipy.stats as stats
from EStest import EStest
```

## Input


```python
print(EStest.__init__.__doc__)
```

            
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
            
    

## Examples

We run the test under two different scenarios. In the first one portfolio returns are generated from a T-student distribution with degrees of freedom equal to ν = 5 but returns are assumed to follow a standard normal.
We can observe that the Value at Risk and Expected Shortfall estimates at 95% are rejected.


```python
np.random.seed(0) #Fix the seed for reproducible results

T = 250 #Sample size
r = 0.05 #VaRLevel
nu = 5 #degrees of freedom for the standard t-Student
x = np.random.standard_t(df = nu, size = T) #Realized values
mu  = 0 #mean
sigma = 1 #standard deviation

#Simulation of returns from the assumed normal distribution
def sim_returns(): return np.random.normal(loc = mu, scale = sigma, size = T) 

#Value-at-Risk estimates
y = np.repeat(stats.norm.ppf(r, loc = mu, scale = sigma), T)

#Expected Shortfall estimates
z = stats.norm.ppf(r, loc = mu, scale = sigma) #Estimated VaR
z = (z-mu)/sigma #Normalized VaR
z = mu - sigma*(stats.norm.pdf(z)/stats.norm.cdf(z)) #Estimated ES
z = np.repeat(z ,T) #ES projections
#The procedure to compute the ES under a normal distribution can be obtained
#here: https://stats.stackexchange.com/questions/166273/expected-value-of-x-in-a-normal-distribution-given-that-it-is-below-a-certain-v

#Test Standard T-Student
test_student = EStest(X_obs = x, X = sim_returns, VaRLevel = r, VaR = y,
                    ES = z, nSim = 100000, print_results = True) 
#one-hundred-thousand Monte-Carlo simulations
```

    ----------------------------------------------------------------
       Direct/Unconditional Expected Shortfall Test by Simulation   
    ----------------------------------------------------------------
    Number of observations: 250
    Number of VaR breaches: 27
    Expected number of VaR breaches: 12.5
    ES Statistic: -1.5893348791528705
    Expected ES Statistic: 0
    Critical Value at α = 0.05: -0.48128966742960516
    p-value: 0.0
    Number of Monte Carlo simulations: 100000
    ----------------------------------------------------------------
    

In the second case both portfolio returns and the estimated return distribution follow a standard normal. 
We can observe that the Value at Risk and Expected Shortfall estimates at 95% are not rejected.


```python
#Standard Normal simulation
p = np.random.normal(size = T)

#Test Standard Normal
test_normal = EStest(X_obs = p, X = sim_returns, VaRLevel = r, VaR = y,
                    ES = z, nSim = 100000, print_results = True) 
```

    ----------------------------------------------------------------
       Direct/Unconditional Expected Shortfall Test by Simulation   
    ----------------------------------------------------------------
    Number of observations: 250
    Number of VaR breaches: 7
    Expected number of VaR breaches: 12.5
    ES Statistic: 0.47382594246166465
    Expected ES Statistic: 0
    Critical Value at α = 0.05: -0.4825457500990175
    p-value: 0.96274
    Number of Monte Carlo simulations: 100000
    ----------------------------------------------------------------
    

## References

Acerbi, Carlo, and Balazs Szekely. "Back-testing expected shortfall." *Risk* 27.11 (2014): 76-81.
