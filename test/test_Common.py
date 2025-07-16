import pytest
import numpy as np
from quanestimation.Common.Common import basis, gramschmidt, suN_generator, BayesInput
from quanestimation.Common.Common import mat_vec_convert, SIC, annihilation, brgd


def test_basis():
    """
    Test the basis function for generating quantum state basis vectors.
    This test checks the generation of basis vectors for a 2-dimensional quantum system.
    """
    result = basis(2, 0)
    assert np.allclose(result, np.array([[1], [0]])) == 1

def test_gramschmidt():
    """
    Test the Gram-Schmidt process for orthonormalizing a set of vectors.
    This test checks the orthonormalization of a set of vectors in a 2-dimensional complex space.
    """
    A = np.array([[1., 0.], [1., 1.]], dtype=np.complex128)
    result = gramschmidt(A)
    assert np.allclose(result, np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])) == 1

def test_suN_generator():
    """
    Test the generation of SU(N) generators for 2- and 3-dimensional quantum system.
    This test checks the correctness of the SU(2) and SU(3) generators, which are the Pauli 
    matrices and Gell-Mann matrices.  
    """
    result = suN_generator(2)
    sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) 
    sy = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
    sz = np.array([[1., 0.j], [0.j, -1.]], dtype=np.complex128)
    sall = [sx, sy, sz]
    all(np.allclose(result[i], sall[i]) for i in range(3)) == 1 

    su3 = suN_generator(3)
    expected_1 = np.array(
        [[0, 1, 0], 
         [1, 0, 0], 
         [0, 0, 0]])
    expected_2 = np.array(
        [[0, -1j, 0], 
         [1j, 0, 0], 
         [0, 0, 0]])
    expected_3 = np.array(   
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 0]])
    expected_4 = np.array(
        [[0, 0, 1], 
         [0, 0, 0], 
         [1, 0, 0]])
    expected_5 = np.array(
        [[0, 0, -1j], 
         [0, 0, 0], 
         [1j, 0, 0]])
    expected_6 = np.array(
        [[0, 0, 0.], 
         [0, 0, 1.], 
         [0., 1., 0]])
    expected_7 = np.array(
        [[0, 0, 0.], 
         [0, 0, -1j], 
         [0., 1j, 0]])
    expected_8 = (1/(np.sqrt(3)))*np.array(  
        [[1, 0, 0], 
         [0, 1, 0], 
         [0, 0, -2]])
    expect = [expected_1, expected_2, expected_3, expected_4, expected_5, expected_6, expected_7, expected_8]
    assert all(np.allclose(su3[i], expect[i]) for i in range(8)) == 1
    

def test_mat_vec_convert():
    """
    Test the conversion of a matrix to a vector and vice versa.
    This test checks the conversion of a 2x2 matrix to a vector and back to a matrix.
    """
    A = np.array([[1., 2.], [3., 4.]])
    result_A = mat_vec_convert(A)
    expected_A = np.array([[1.], [2.], [3.], [4.]])
    result_inv = mat_vec_convert(result_A)
    assert np.allclose(result_A, expected_A) == 1  
    assert np.allclose(result_inv, A) == 1
    
def test_SIC():
    """
    Test the generation of SIC-POVM (Symmetric Informationally Complete Positive Operator-Valued Measure).
    This test checks the generation of SIC-POVM for a 2-dimensional quantum system.
    """
    result = SIC(2)
    expected = [
        np.array([[0.39433757+0.j, 0.14433757+0.14433757j],
                  [0.14433757-0.14433757j, 0.10566243+0.j]]), 
        np.array([[ 0.39433757+0.j, -0.14433757-0.14433757j],
                  [-0.14433757+0.14433757j,  0.10566243+0.j]]), 
        np.array([[0.10566243+0.j, 0.14433757-0.14433757j],
                  [0.14433757+0.14433757j, 0.39433757+0.j]]), 
        np.array([[ 0.10566243+0.j, -0.14433757+0.14433757j],
                  [-0.14433757-0.14433757j,  0.39433757+0.j]])]
    assert all(np.allclose(result[i], expected[i]) for i in range(4)) == 1    

    with pytest.raises(ValueError):
        SIC(200)

def test_annilation():
    """
    Test the generation of annihilation operator for a 2-dimensional quantum system.
    This test checks the generation of the annihilation operator for a 2-dimensional quantum system.
    """
    result = annihilation(2)
    expected = np.array([[0., 1.], [0., 0.]])
    assert np.allclose(result, expected) == 1

def test_brgd():
    result1 = brgd(1)
    expected1 = ["0", "1"]
    result2 = brgd(2)
    expected2 = ['00', '01', '11', '10']
    assert result1 == expected1   
    assert result2 == expected2  

def test_BayesInput():
    """
    Test the BayesInput function for generating a quantum state based on input parameters.
    This test checks the generation of a quantum state for a 2-dimensional system with specific parameters.
    """
    # Test with a simple Hamiltonian
    H = lambda x: [x * np.array([[1, 0], [0, -1]])]
    dH = lambda x: [np.array([[1, 0], [0, -1]])]
    xspan = [np.linspace(0, 1, 2)]
    result_H, result_dH = BayesInput(xspan, H, dH, channel="dynamics")
    expected_H = [np.array([[0., 0.], [0., 0.]]), 
                  np.array([[1., 0.], [0., -1.]])]
    excepted_dH = [np.array([[1, 0], [0, -1]]), 
                   np.array([[1, 0], [0, -1]])]
    assert all(np.allclose(result_H[i], expected_H[i]) for i in range(2)) == 1
    assert all(np.allclose(result_dH[i], excepted_dH[i]) for i in range(2)) == 1

    # Test with invalid channel type
    with pytest.raises(ValueError):
        BayesInput(xspan, H, dH, channel="invalid_channel")

    # Test with a set of Kraus operators 
    K = lambda x: [np.array([[1, 0], [0, np.sqrt(1-x)]]), np.array([[0, np.sqrt(x)], [0, 0]])]
    dK = lambda x: [[np.array([[0, 0], [0, -0.5/np.sqrt(1-x)]]), np.array([[0, 0.5/np.sqrt(x)], [0, 0]])]]
    xspan_K = [np.linspace(0.1, 0.5, 2)]
    result_K, result_dK = BayesInput(xspan_K, K, dK, channel="Kraus")
    expected_K = [[np.array([[1, 0], [0, np.sqrt(0.9)]]), np.array([[0, np.sqrt(0.1)], [0, 0]])],
                  [np.array([[1, 0], [0, np.sqrt(0.5)]]), np.array([[0, np.sqrt(0.5)], [0, 0]])]]
    expected_dK = [[np.array([[0, 0], [0, -0.5/np.sqrt(0.9)]]), np.array([[0, 0.5/np.sqrt(0.1)], [0, 0]])], 
                   [np.array([[0, 0], [0, -0.5/np.sqrt(0.5)]]), np.array([[0, 0.5/np.sqrt(0.5)], [0, 0]])]]
    assert all(np.allclose(result_K[i], expected_K[i]) for i in range(2)) == 1
    assert all(np.allclose(result_dK[i], expected_dK[i]) for i in range(2)) == 1