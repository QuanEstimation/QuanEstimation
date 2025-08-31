import pytest
import numpy as np 
import random
import os
from scipy.integrate import simpson
from quanestimation.BayesianBound.BayesEstimation import (
    Bayes,
    MLE,
    BCB,
    BayesCost,
)
from quanestimation.Parameterization.GeneralDynamics import Lindblad

def test_Bayes_singleparameter() -> None:
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
    rho = []
    for xi in x:
        H0 = H0_func(xi)
        dH = dH_func(xi)
        dynamics = Lindblad(tspan, rho0, H0, dH)
        rho_tp, _ = dynamics.expm()
        rho.append(rho_tp[-1])

    random.seed(1234)
    y = [0 for _ in range(500)]
    res_rand = random.sample(range(len(y)), 125)
    
    for i in res_rand:
        y[i] = 1

    pout_MAP, xout_MAP = Bayes(
        [x], p, rho, y, M = M, estimator = "MAP", savefile = False
    )

    pout_MAP_max = max(pout_MAP)
    expected_xout_MAP = 0.7861843477451934
    expected_pout_MAP_max = 0.15124761081089924
    assert np.allclose(xout_MAP, expected_xout_MAP, atol = 1e-3)
    assert np.allclose(pout_MAP_max, expected_pout_MAP_max, atol = 1e-3)

    expected_xout_MLE = 0.7861843477451934
    _, xout_MLE = MLE([x], rho, y, M = M, savefile=False)
    assert np.allclose(xout_MLE, expected_xout_MLE, atol = 1e-3)

    pout_mean, xout_mean = Bayes(
        [x], p, rho, y, M=M, estimator="mean", savefile=False
    )

    pout_mean_max = max(pout_mean)
    expected_xout_mean = 0.01158475411409417
    expected_pout_mean_max = 0.15124761081089924
    assert np.allclose(xout_mean, expected_xout_mean, atol = 1e-3)
    assert np.allclose(pout_mean_max, expected_pout_mean_max, atol = 1e-3)
    
    # Clean up generated files
    for filename in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(filename):
            os.remove(filename)

    # Test saving functionality
    pout_mean, xout_mean = Bayes(
        [x], p, rho, y, M=M, estimator="mean", savefile=True
    )
    assert os.path.exists("pout.npy")
    assert os.path.exists("xout.npy")

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

def test_Bayes_multiparameter() -> None:
    # Initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # Free Hamiltonian parameters
    b_val = 0.5 * np.pi
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    # Hamiltonian function
    hamiltonian_func = lambda omega0, x: 0.5 * b_val * omega0 * (
        sigma_x * np.cos(x) + sigma_z * np.sin(x)
    )

    # Derivative of Hamiltonian
    d_hamiltonian_func = lambda omega0, x: [
        0.5 * b_val * (sigma_x * np.cos(x) + sigma_z * np.sin(x)),
        0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))
    ]
    
    # Prior distribution parameters
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)
    omega0_values = np.linspace(1, 2, 20)
    all_parameter_values = [omega0_values, x_values]
    
    # Joint probability density function (Gaussian for both parameters)
    mu_omega0, mu_x = 1.5, 0.0
    eta_omega0, eta_x = 0.2, 0.2
    prob_density = lambda omega0, x: (
        np.exp(-(omega0 - mu_omega0)**2 / (2 * eta_omega0**2)) / (eta_omega0 * np.sqrt(2 * np.pi))
        * np.exp(-(x - mu_x)**2 / (2 * eta_x**2)) / (eta_x * np.sqrt(2 * np.pi))
    )

    # Generate probability values
    prob_values_unnormalized = np.zeros((len(omega0_values), len(x_values)))
    for i, omega0_i in enumerate(omega0_values):
        for j, x_values_j in enumerate(x_values):
            prob_values_unnormalized[i, j] = prob_density(omega0_i, x_values_j)

    # Normalize the distribution
    integral_x = np.zeros(len(omega0_values))
    for i in range(len(omega0_values)):
        integral_x[i] = simpson(prob_values_unnormalized[i, :], x_values)
    norm_factor = simpson(integral_x, omega0_values)
    prob_normalized = prob_values_unnormalized / norm_factor

    random.seed(1234)
    y = [0 for _ in range(500)]
    res_rand = random.sample(range(len(y)), 125)
    
    for i in res_rand:
        y[i] = 1

    # Time evolution parameters
    time_span = np.linspace(0.0, 1.0, 50)

    # Prepare arrays for the state
    final_states = []

    # Evolve the system for each parameter combination
    for omega0_i in omega0_values:
        row_rho = []

        for x_values_j in x_values:
            hamiltonian = hamiltonian_func(omega0_i, x_values_j)
            d_hamiltonian = d_hamiltonian_func(omega0_i, x_values_j)

            dynamics = Lindblad(time_span, rho0, hamiltonian, d_hamiltonian)
            states, _ = dynamics.expm()
            
            row_rho.append(states[-1])

        final_states.append(row_rho) 

    pout_MAP, xout_MAP = Bayes(
        all_parameter_values, prob_normalized, final_states, y, M = None, estimator = "MAP", savefile = False
    )

    expected_xout_MAP = [2.0, 0.5787144361875933]
    pout_MAP_max = np.max(pout_MAP)
    expected_pout_MAP_max = 0.9977126619164614
    assert np.allclose(xout_MAP, expected_xout_MAP, atol = 1e-3)
    assert np.allclose(pout_MAP_max, expected_pout_MAP_max, atol = 1e-3)

    expected_xout_MLE = [2.0, 0.5787144361875933]
    _, xout_MLE = MLE(all_parameter_values, final_states, y, M = [], savefile = False)
    assert np.allclose(xout_MLE, expected_xout_MLE, atol = 1e-3)

    pout_mean, xout_mean = Bayes(
        all_parameter_values, prob_normalized, final_states, y, M = None, estimator="mean", savefile=False
    )

    pout_mean_max = np.max(pout_mean)
    expected_xout_mean = [0.01124410514670476, 0.0032666400994622027]
    expected_pout_mean_max = 0.9977126619164614
    assert np.allclose(xout_mean, expected_xout_mean, atol = 1e-3)
    assert np.allclose(pout_mean_max, expected_pout_mean_max, atol = 1e-3)

    # Clean up generated files
    for filename in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(filename):
            os.remove(filename)

    with pytest.raises(TypeError):
        Bayes(all_parameter_values, prob_normalized, final_states, y, M = 1., estimator = "mean", savefile = False)    

    with pytest.raises(ValueError):
        Bayes(all_parameter_values, prob_normalized, final_states, y, M = None, estimator = "invalid_estimator", savefile = False) 

    with pytest.raises(TypeError):        
        _, xout_MLE = MLE(all_parameter_values, final_states, y, M = 1., savefile = False)        

def test_BCB_singleparameter() -> None:
    # initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # free Hamiltonian
    B, omega0 = np.pi / 2.0, 1.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    H0_func = lambda x: 0.5 * B * omega0 * (sx * np.cos(x) + sz * np.sin(x))
    
    # derivative of the free Hamiltonian on x
    dH_func = lambda x: [0.5 * B * omega0 * (-sx * np.sin(x) + sz * np.cos(x))]
    
    # prior distribution
    x = np.linspace(0.0, 0.5 * np.pi, 1000)
    p = (1.0 / (x[-1] - x[0])) * np.ones(len(x))
    
    # time length for the evolution
    tspan = np.linspace(0.0, 1.0, 10)
    
    # dynamics
    rho = []   
    for xi in x:
        H0 = H0_func(xi)
        dH = dH_func(xi)
        dynamics = Lindblad(tspan, rho0, H0, dH)
        rho_tp, _ = dynamics.expm()
        rho.append(rho_tp[-1]) 

    result = BCB([x], p, rho, W = [], eps = 1e-8)
    expected_result = 0.16139667479361308
    assert np.allclose(result, expected_result, atol = 1e-3)

def test_BayesCost_singleparameter() -> None:
     # initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # free Hamiltonian
    B, omega0 = np.pi / 2.0, 1.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    H0_func = lambda x: 0.5 * B * omega0 * (sx * np.cos(x) + sz * np.sin(x))
    
    # derivative of the free Hamiltonian on x
    dH_func = lambda x: [0.5 * B * omega0 * (-sx * np.sin(x) + sz * np.cos(x))]
    
    # prior distribution
    x = np.linspace(0.0, 0.5 * np.pi, 1000)
    p = (1.0 / (x[-1] - x[0])) * np.ones(len(x))
    
    # time length for the evolution
    tspan = np.linspace(0.0, 1.0, 10)
    
    # dynamics
    rho = []   
    for xi in x:
        H0 = H0_func(xi)
        dH = dH_func(xi)
        dynamics = Lindblad(tspan, rho0, H0, dH)
        rho_tp, _ = dynamics.expm()
        rho.append(rho_tp[-1]) 

    random.seed(1234)
    y = [0 for _ in range(500)]
    res_rand = random.sample(range(len(y)), 125)
    
    for i in res_rand:
        y[i] = 1 

    pout_mean, xout_mean = Bayes(
        [x], p, rho, y, M = [], estimator = "mean", savefile = False
    )    

    # Clean up generated files
    for filename in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(filename):
            os.remove(filename)   

    result = BayesCost([x], pout_mean, xout_mean, rho, M = [])
    expected_result = 0.0029817568167127637
    assert np.allclose(result, expected_result, atol = 1e-3)

    with pytest.raises(TypeError):
        BayesCost([x], pout_mean, xout_mean, rho, M = 1.)

