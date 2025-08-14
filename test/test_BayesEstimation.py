import pytest
import numpy as np 
import random
import os
from quanestimation.BayesianBound.BayesEstimation import (
    Bayes,
    MLE,
)
from quanestimation.Parameterization.GeneralDynamics import Lindblad


def test_Bayes() -> None:
    # initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # free Hamiltonian
    B, omega0 = np.pi / 2.0, 1.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    H0_func = lambda x: 0.5 * B * omega0 * (sx * np.cos(x) + sz * np.sin(x))
    
    # derivative of the free Hamiltonian on x
    dH_func = lambda x: [0.5 * B * omega0 * (-sx * np.sin(x) + sz * np.cos(x))]
    
    # measurement
    M1 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    M2 = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    M = [M1, M2]
    
    # prior distribution
    x = np.linspace(0.0, 0.5 * np.pi, 1000)
    p = (1.0 / (x[-1] - x[0])) * np.ones(len(x))
    
    # time length for the evolution
    tspan = np.linspace(0.0, 1.0, 10)
    
    # dynamics
    rho = [
        np.zeros((len(rho0), len(rho0)), dtype=np.complex128
    ) for i in range(len(x))
    ]
    
    for i in range(len(x)):
        H0 = H0_func(x[i])
        dH = dH_func(x[i])
        dynamics = Lindblad(tspan, rho0, H0, dH)
        rho_tp, _ = dynamics.expm()
        rho[i] = rho_tp[-1]

    random.seed(1234)
    y = [0 for _ in range(500)]
    res_rand = random.sample(range(len(y)), 125)
    
    for i in res_rand:
        y[i] = 1

    pout_MAP, xout_MAP = Bayes(
        [x], p, rho, y, M=M, estimator="MAP", savefile=False
    )

    assert np.allclose(xout_MAP, 0.7861843477451934)
    assert np.allclose(max(pout_MAP), 0.15124761081089924)

    _, xout_MLE = MLE([x], rho, y, M=M, savefile=False)
    assert np.allclose(xout_MLE, 0.7861843477451934)

    pout_mean, xout_mean = Bayes(
        [x], p, rho, y, M=M, estimator="mean", savefile=False
    )

    assert np.allclose(xout_mean, 0.01158475411409417)
    assert np.allclose(max(pout_mean), 0.15124761081089924)
    
    # Clean up generated files
    for filename in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(filename):
            os.remove(filename)
            
    with pytest.raises(TypeError):
        Bayes([x], p, rho, y, M=1., estimator="mean", savefile=False)    

    with pytest.raises(ValueError):
        Bayes([x], p, rho, y, M=M, estimator="invalid_estimator", savefile=False) 

    with pytest.raises(TypeError):        
        _, xout_MLE = MLE([x], rho, y, M=1., savefile=False)
