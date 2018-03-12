import numpy as np

from scipy.special import erf

from scipy.special import erfc



def betaConstant(Stg, Stl, mu = 1., nu = 1., beta = 0.01):

    
    """
    Solves trascendental equation for beta constant

    Arguments
    Stg: Stefan number for gas region
    Stl: Stefan number for liquid region
    nu: square root of thermal diffusivities ratio, i.e., $\sqrt{\frac{\alpha_g}{\alpha_l}}$
    mu: density ratio. i.e., \frac{\rho_g}{\rho_l}

    Returns beta
    """


    def f(Stg, Stl, mu, nu, beta):

        a = beta * np.exp(beta**2) * erf(beta) 
        
        b = (mu*nu*beta) * np.exp((mu*nu*beta)**2) * erfc(mu*nu*beta)

        return Stg / a - Stl / b - np.sqrt(np.pi)
    


    
    # Iteration using secant method    

    er = 1.0

    beta_2 = beta

    beta_1 = 1.1*beta_2
    
    
    while( abs(er) > 1e-10 ):

        beta_0 = beta_1 - ( f(Stg, Stl, mu, nu, beta_1) * (beta_1 - beta_2) )  /  ( f(Stg, Stl, mu, nu, beta_1) - f(Stg, Stl, mu, nu, beta_2) )

        er = beta_0 - beta_1

        beta_2 = beta_1

        beta_1 = beta_0



    return beta_1
    
