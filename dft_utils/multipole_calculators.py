import numpy as np
from numpy import einsum

from .units import Units


def multmomsq_to_epsA_prefactor(pow, energy, epsr):
    """Process multmomsq to epsA prefactor."""
    if pow == 1:
        num = np.pi * Units.constants["NA"] * energy
        denom = (
            3.0
            * Units.constants["eps0"]
            * np.sqrt(epsr)
            * Units.constants["hbar"]
            * Units.constants["c"]
            * np.log(10.0)
        )
    elif pow == 2:
        num = np.pi * Units.constants["NA"] * (energy**3) * np.sqrt(epsr)
        denom = (
            Units.constants["eps0"]
            * (Units.constants["hbar"] ** 3)
            * (Units.constants["c"] ** 3)
            * np.log(10.0)
        )
    else:
        raise NotImplementedError()
    return num / denom


def dip_moment_calculator(d):
    """Process dip moment calculator."""
    assert d.shape[-1] == 3

    return np.sum(d * np.conjugate(d), axis=-1)


def quad_moment_calculator(q):
    """Process quad moment calculator."""
    assert np.all([q.shape[-(i + 1)] == 3 for i in range(2)])
    qsq1 = 3.0 / 2 * np.sum(q * np.conjugate(q), axis=(-1, -2))
    qsq2 = np.trace(q, axis1=-2, axis2=-1)
    qsq2 = (-1.0 / 2) * qsq2 * np.conjugate(qsq2)
    return qsq1 + qsq2


def vary_quadrupole_tensor_displacement(quad, dip, dx):
    """Process vary quadrupole tensor displacement."""
    assert dip.ndim == 1
    assert dx.ndim == 1
    d1 = -dx[:, None] * dip[None, :]
    d2 = -dx[None, :] * dip[:, None]
    return quad + d1 + d2


def wrapper_func_quad(x, quad, dip):
    """Process wrapper func quad."""
    new_quad = vary_quadrupole_tensor_displacement(quad, dip, x)
    return quad_moment_calculator(new_quad)


def octo_moment_calculator_reals(o):
    """Process octo moment calculator reals."""
    assert o.ndim == 3
    assert np.all([o.shape[i] == 3 for i in range(3)])
    diag_term = np.sum([o[i, i, i] ** 2 for i in range(3)])
    off_diag_term1 = 6.0 * (
        o[0, 0, 1] ** 2
        + o[0, 0, 2] ** 2
        + o[0, 1, 1] ** 2
        + o[0, 2, 2] ** 2
        + o[1, 1, 2] ** 2
        + o[1, 2, 2] ** 2
    )
    off_diag_term2 = 15.0 * o[0, 1, 2] ** 2
    off_diag_term3 = 3.0 * (
        o[0, 1, 1] * o[0, 2, 2]
        + o[0, 0, 0] * o[0, 1, 1]
        + o[0, 0, 0] * o[0, 2, 2]
        + o[0, 0, 2] * o[1, 1, 2]
        + o[1, 1, 1] * o[1, 2, 2]
        + o[0, 0, 1] * o[1, 1, 1]
        + o[0, 0, 1] * o[1, 2, 2]
        + o[0, 0, 2] * o[2, 2, 2]
        + o[1, 1, 2] * o[2, 2, 2]
    )
    out = diag_term + off_diag_term1 + off_diag_term2 - off_diag_term3
    return out


def octo_moment_calculator(o):
    """Process octo moment calculator."""
    assert o.ndim == 3
    assert np.all([o.shape[i] == 3 for i in range(3)])
    out = (
        4
        / 3
        * (
            np.abs(o[0, 0, 0]) ** 2
            + 6 * np.abs(o[0, 0, 1]) ** 2
            + 6 * np.abs(o[0, 0, 2]) ** 2
            - 3 * np.imag(o[0, 0, 0]) * np.imag(o[0, 1, 1])
            + 6 * np.abs(o[0, 1, 1]) ** 2
            + 15 * np.abs(o[0, 1, 2]) ** 2
            - 3 * np.imag(o[0, 0, 0]) * np.imag(o[0, 2, 2])
            - 3 * np.imag(o[0, 1, 1]) * np.imag(o[0, 2, 2])
            + 6 * np.abs(o[0, 2, 2]) ** 2
            - 3 * np.imag(o[0, 0, 1]) * np.imag(o[1, 1, 1])
            + np.abs(o[1, 1, 1]) ** 2
            - 3 * np.imag(o[0, 0, 2]) * np.imag(o[1, 1, 2])
            + 6 * np.abs(o[1, 1, 2]) ** 2
            - 3 * np.imag(o[0, 0, 1]) * np.imag(o[1, 2, 2])
            - 3 * np.imag(o[1, 1, 1]) * np.imag(o[1, 2, 2])
            + 6 * np.abs(o[1, 2, 2]) ** 2
            - 3 * np.imag(o[0, 0, 2]) * np.imag(o[2, 2, 2])
            - 3 * np.imag(o[1, 1, 2]) * np.imag(o[2, 2, 2])
            + np.abs(o[2, 2, 2]) ** 2
            - 3 * np.real(o[0, 0, 0]) * np.real(o[0, 1, 1])
            - 3 * np.real(o[0, 0, 0]) * np.real(o[0, 2, 2])
            - 3 * np.real(o[0, 1, 1]) * np.real(o[0, 2, 2])
            - 3 * np.real(o[0, 0, 1]) * np.real(o[1, 1, 1])
            - 3 * np.real(o[0, 0, 2]) * np.real(o[1, 1, 2])
            - 3 * np.real(o[0, 0, 1]) * np.real(o[1, 2, 2])
            - 3 * np.real(o[1, 1, 1]) * np.real(o[1, 2, 2])
            - 3 * np.real(o[0, 0, 2]) * np.real(o[2, 2, 2])
            - 3 * np.real(o[1, 1, 2]) * np.real(o[2, 2, 2])
        )
    )
    return out


def vary_octupole_tensor_displacement(octu, quad, dip, dx):
    """Process vary octupole tensor displacement."""
    assert quad.ndim == 2
    assert dip.ndim == 1
    assert dx.ndim == 1
    ddip1 = dx[:, None, None] * dx[None, :, None] * dip[None, None, :]
    ddip2 = dx[:, None, None] * dx[None, None, :] * dip[None, :, None]
    ddip3 = dx[None, :, None] * dx[None, None, :] * dip[:, None, None]
    ddip_tot = ddip1 + ddip2 + ddip3
    dquad1 = dx[None, None, :] * quad[:, :, None]
    dquad2 = dx[None, :, None] * quad[:, None, :]
    dquad3 = dx[:, None, None] * quad[None, :, :]
    dquad_tot = dquad1 + dquad2 + dquad3
    return octu - dquad_tot + ddip_tot


def wrapper_func_octo(x, octu, quad, dip):
    """Process wrapper func octo."""
    new_octu = vary_octupole_tensor_displacement(octu, quad, dip, x)
    return octo_moment_calculator(new_octu)


def calc_dip_moment(x):
    """Process calc dip moment."""
    return [dip_moment_calculator(x[i, :]) for i in range(x.shape[0])]


def calc_quad_moment(x):
    """Process calc quad moment."""
    return [quad_moment_calculator(x[i, :, :]) for i in range(x.shape[0])]


def calc_oct_moment(x):
    """Process calc oct moment."""
    return [octo_moment_calculator(x[i, :, :, :]) for i in range(x.shape[0])]


def list_to_symmetric_matrix(tensor_list):
    """Convert list [XX, XY, XZ, YY, YZ, ZZ] to symmetric matrix"""
    return np.moveaxis(
        np.array(
            [
                [tensor_list[..., 0], tensor_list[..., 1], tensor_list[..., 2]],
                [tensor_list[..., 1], tensor_list[..., 3], tensor_list[..., 4]],
                [tensor_list[..., 2], tensor_list[..., 4], tensor_list[..., 5]],
            ]
        ),
        (0, 1),
        (-2, -1),
    )


def list_to_symmetric_matrix_ORCA(tensor_list):
    """Convert list [XX, YY, ZZ, XY, XZ, YZ] to symmetric matrix"""
    arr = np.array(
        [
            [tensor_list[..., 0], tensor_list[..., 3], tensor_list[..., 4]],
            [
                tensor_list[..., 3].conj(),
                tensor_list[..., 1],
                tensor_list[..., 5],
            ],
            [
                tensor_list[..., 4].conj(),
                tensor_list[..., 5].conj(),
                tensor_list[..., 2],
            ],
        ]
    )
    ndim = arr.ndim
    return np.moveaxis(arr, (0, 1), (ndim - 2, ndim - 1))


def list_to_symmetric_rank3_tensor(tensor_list):
    """Convert list [XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ] to symmetric rank-3 tensor."""
    tensor = np.zeros(tensor_list[..., 0].shape + (3, 3, 3))

    indices = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 2),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ]

    for i, (a, b, c) in enumerate(indices):
        tensor[..., a, b, c] = tensor[..., a, c, b] = tensor[..., b, a, c] = tensor[
            ..., b, c, a
        ] = tensor[..., c, a, b] = tensor[..., c, b, a] = tensor_list[..., i]

    return tensor


def make_traceless(matrix):
    """
    Make a square matrix traceless by subtracting the mean of its diagonal elements.

    Parameters:
    - matrix: A square numpy array (n x n).

    Returns:
    - traceless_matrix: The traceless version of the input matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    trace = np.trace(matrix)

    n = matrix.shape[0]
    traceless_matrix = matrix - (trace / n) * np.eye(n)

    return traceless_matrix
