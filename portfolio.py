import numpy as np
import pandas as pd
from scipy.optimize import minimize

def annualize_rets(ret, periods_per_year=12):
    compounded_growth = (1 + ret).prod()
    n_periods = ret.shape[0]
    return compounded_growth ** (periods_per_year / max(n_periods, 1)) - 1

def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights) ** 0.5

def msr(riskfree_rate, ret, covmat, bound=(0.0, 1.0)):
    n = ret.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = (bound,) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    def neg_sharpe(w, r_f, er, cov):
        r = portfolio_return(w, er)
        v = portfolio_vol(w, cov)
        return -(r - r_f) / max(v, 1e-6)

    res = minimize(neg_sharpe, init_guess, args=(riskfree_rate, ret, covmat),
                   method='SLSQP', bounds=bounds, constraints=(weights_sum_to_1,))
    return res.x

def min_volatility(riskfree_rate, ret, covmat, bound=(0.0, 1.0)):
    n = ret.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = (bound,) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(portfolio_vol, init_guess, args=(covmat,),
                   method='SLSQP', bounds=bounds, constraints=(weights_sum_to_1,))
    return res.x

def compute_portfolio_allocations(monthly_returns: pd.DataFrame, risk_free_rate: float = 0.04):
    cov_matrix = monthly_returns.cov()
    expected_returns = annualize_rets(monthly_returns)
    bound = (0.01, 0.30)
    
    w_msr = msr(risk_free_rate, expected_returns, cov_matrix, bound)
    w_vol = min_volatility(risk_free_rate, expected_returns, cov_matrix, bound)
    
    msr_ret = portfolio_return(w_msr, expected_returns)
    msr_vol = portfolio_vol(w_msr, cov_matrix)
    
    vol_ret = portfolio_return(w_vol, expected_returns)
    vol_vol = portfolio_vol(w_vol, cov_matrix)
    
    return {
        "msr_weights": pd.Series(w_msr, index=monthly_returns.columns),
        "msr_return": msr_ret, "msr_vol": msr_vol,
        "vol_weights": pd.Series(w_vol, index=monthly_returns.columns),
        "vol_return": vol_ret, "vol_vol": vol_vol
    }