import pytest
import numpy as np
from quanestimation.Common.Common import (
    basis,
    gramschmidt,
    suN_generator,
    BayesInput,
    extract_ele,
    mat_vec_convert,
    SIC,
    annihilation,
    brgd,
    fidelity
)


def test_basis():
    """Test basis function for generating quantum state basis vectors."""
    result = basis(2, 0)
    assert np.allclose(result, np.array([[1], [0]]))


def test_gramschmidt():
    """Test Gram-Schmidt process for orthonormalizing vectors."""
    A = np.array([[1., 0.], [1., 1.]], dtype=np.complex128)
    result = gramschmidt(A)
    expected = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])
    assert np.allclose(result, expected)


def test_suN_generator():
    """Test generation of SU(N) generators."""
    # Test SU(2) generators (Pauli matrices)
    result = suN_generator(2)
    sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    sy = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
    sz = np.array([[1., 0.j], [0.j, -1.]], dtype=np.complex128)
    sall = [sx, sy, sz]
    assert all(np.allclose(result[i], sall[i]) for i in range(3))

    # Test SU(3) generators (Gell-Mann matrices)
    su3 = suN_generator(3)
    expected_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    expected_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    expected_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    expected_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    expected_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    expected_6 = np.array([[0, 0, 0.], [0, 0, 1.], [0., 1., 0]])
    expected_7 = np.array([[0, 0, 0.], [0, 0, -1j], [0., 1j, 0]])
    expected_8 = (1/np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    expect = [
        expected_1, expected_2, expected_3, expected_4,
        expected_5, expected_6, expected_7, expected_8
    ]
    assert all(np.allclose(su3[i], expect[i]) for i in range(8))


def test_mat_vec_convert():
    """Test matrix to vector conversion and vice versa."""
    A = np.array([[1., 2.], [3., 4.]])
    result_A = mat_vec_convert(A)
    expected_A = np.array([[1.], [2.], [3.], [4.]])
    result_inv = mat_vec_convert(result_A)
    assert np.allclose(result_A, expected_A)
    assert np.allclose(result_inv, A)


def test_SIC():
    """Test generation of SIC-POVM."""
    result = SIC(2)
    expected = [
        np.array([[0.39433757+0.j, 0.14433757+0.14433757j],
                  [0.14433757-0.14433757j, 0.10566243+0.j]]),
        np.array([[0.39433757+0.j, -0.14433757-0.14433757j],
                  [-0.14433757+0.14433757j, 0.10566243+0.j]]),
        np.array([[0.10566243+0.j, 0.14433757-0.14433757j],
                  [0.14433757+0.14433757j, 0.39433757+0.j]]),
        np.array([[0.10566243+0.j, -0.14433757+0.14433757j],
                  [-0.14433757-0.14433757j, 0.39433757+0.j]])
    ]
    assert all(np.allclose(result[i], expected[i]) for i in range(4))

    with pytest.raises(ValueError):
        SIC(200)


def test_annihilation():
    """Test generation of annihilation operator."""
    result = annihilation(2)
    expected = np.array([[0., 1.], [0., 0.]])
    assert np.allclose(result, expected)


def test_brgd():
    """Test binary reflected Gray code generation."""
    result1 = brgd(1)
    expected1 = ["0", "1"]
    result2 = brgd(2)
    expected2 = ['00', '01', '11', '10']
    assert result1 == expected1
    assert result2 == expected2


def test_BayesInput():
    """Test BayesInput function for quantum state generation."""
    # Test with Hamiltonian
    H = lambda x: [x * np.array([[1, 0], [0, -1]])]
    dH = lambda x: [np.array([[1, 0], [0, -1]])]
    xspan = [np.linspace(0, 1, 2)]
    result_H, result_dH = BayesInput(xspan, H, dH, channel="dynamics")
    expected_H = [
        np.array([[0., 0.], [0., 0.]]),
        np.array([[1., 0.], [0., -1.]])
    ]
    expected_dH = [
        np.array([[1, 0], [0, -1]]),
        np.array([[1, 0], [0, -1]])
    ]
    assert all(np.allclose(result_H[i], expected_H[i]) for i in range(2))
    assert all(np.allclose(result_dH[i], expected_dH[i]) for i in range(2))

    # Test with invalid channel type
    with pytest.raises(ValueError):
        BayesInput(xspan, H, dH, channel="invalid_channel")

    # Test with Kraus operators
    K = lambda x: [
        np.array([[1, 0], [0, np.sqrt(1-x)]]),
        np.array([[0, np.sqrt(x)], [0, 0]])
    ]
    dK = lambda x: [[
        np.array([[0, 0], [0, -0.5/np.sqrt(1-x)]]),
        np.array([[0, 0.5/np.sqrt(x)], [0, 0]])
    ]]
    xspan_K = [np.linspace(0.1, 0.5, 2)]
    result_K, result_dK = BayesInput(xspan_K, K, dK, channel="Kraus")
    expected_K = [
        [
            np.array([[1, 0], [0, np.sqrt(0.9)]]),
            np.array([[0, np.sqrt(0.1)], [0, 0]])
        ],
        [
            np.array([[1, 0], [0, np.sqrt(0.5)]]),
            np.array([[0, np.sqrt(0.5)], [0, 0]])
        ]
    ]
    expected_dK = [
        [
            np.array([[0, 0], [0, -0.5/np.sqrt(0.9)]]),
            np.array([[0, 0.5/np.sqrt(0.1)], [0, 0]])
        ],
        [
            np.array([[0, 0], [0, -0.5/np.sqrt(0.5)]]),
            np.array([[0, 0.5/np.sqrt(0.5)], [0, 0]])
        ]
    ]
    assert all(np.allclose(result_K[i], expected_K[i]) for i in range(2))
    assert all(np.allclose(result_dK[i], expected_dK[i]) for i in range(2))


def test_extract_ele():
    """Test extract_ele generator for recursive element extraction."""
    # Test with depth 0
    element = [1, 2, 3]
    n = 0
    result = list(extract_ele(element, n))
    assert result == [element]

    # Test with depth 1
    n = 1
    result = list(extract_ele(element, n))
    assert result == [1, 2, 3]

    # Test with nested lists and depth 2
    nested = [[1, 2], [3, [4, 5]]]
    n = 2
    result = list(extract_ele(nested, n))
    assert result == [1, 2, 3, [4, 5]]

def test_fidelity():
    """Test fidelity function for quantum states."""
    rho1 = np.array([[0.5, 0.5], [0.5, 0.5]])
    rho2 = np.array([[1, 0], [0, 0]])
    result = fidelity(rho1, rho2)
    expected = 0.5
    assert np.isclose(result, expected)

    # Test with vectors
    psi = np.array([1, 0])
    phi = np.array([0, 1])
    result_vec = fidelity(psi, phi)
    expected_vec = 0.
    assert np.isclose(result_vec, expected_vec)

    rho3 = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        fidelity(rho1, rho3)
    
    rho4 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        fidelity(rho1, rho4)

    with pytest.raises(TypeError):
        fidelity([], rho2)