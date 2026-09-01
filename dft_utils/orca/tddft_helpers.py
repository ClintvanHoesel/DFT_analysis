import numpy as np
from numpy import einsum


def intsAO_to_intsMO_restricted(ints, mo_coeff):
    """Process intsAO to intsMO restricted."""
    out = np.tensordot(mo_coeff.conj(), ints, axes=(0, -2))
    out = np.tensordot(mo_coeff, out, axes=(0, -1))
    out = np.moveaxis(out, 1, -1)
    out = np.moveaxis(out, 0, -1)
    return out


def intsAO_to_intsMO_unrestricted(ints, mo_coeff):
    """Process intsAO to intsMO unrestricted."""
    out = np.tensordot(mo_coeff.conj(), ints, axes=(1, -2))
    out = np.tensordot(out, mo_coeff, axes=(-1, 1))
    out = np.diagonal(out, axis1=0, axis2=-2)
    out = np.moveaxis(out, 0, -3)
    out = np.moveaxis(out, -1, 0)
    return out


def _mo_occ_masks_restricted(mo_occ):
    """Handle mo occ masks restricted internally."""
    mask = mo_occ == 2
    mask2 = np.logical_not(mask)
    return mask, mask2


def _mo_occ_masks_unrestricted(mo_occ):
    """Handle mo occ masks unrestricted internally."""
    return (mo_occ[0] == 1, mo_occ[0] == 0, mo_occ[1] == 1, mo_occ[1] == 0)


def static_multipole_restricted_MO(intsMO, mo_occ):
    """Process static multipole restricted MO."""
    intsMO = -1.0 * np.diagonal(intsMO, axis1=-2, axis2=-1)
    return np.tensordot(mo_occ, intsMO, axes=(0, -1))


def static_multipole_unrestricted_MO(intsMO, mo_occ):
    """Process static multipole unrestricted MO."""
    intsMO = -1.0 * np.diagonal(intsMO, axis1=-2, axis2=-1)
    return np.tensordot(mo_occ, intsMO, axes=([0, 1], [0, -1]))


def intsMO_cut_restricted(intsMO, mo_occ):
    """Process intsMO cut restricted."""
    masko, maskv = _mo_occ_masks_restricted(mo_occ)
    intsOO = intsMO[..., masko, :][..., masko]
    intsOV = intsMO[..., masko, :][..., maskv]
    intsVV = intsMO[..., maskv, :][..., maskv]
    return intsOO, intsOV, intsVV


def intsMO_cut_unrestricted(intsMO, mo_occ):
    """Process intsMO cut unrestricted."""
    maskoa, maskva, maskob, maskvb = _mo_occ_masks_unrestricted(mo_occ)
    intsOaOa = intsMO[0][..., maskoa, :][..., maskoa]
    intsOaVa = intsMO[0][..., maskoa, :][..., maskva]
    intsVaVa = intsMO[0][..., maskva, :][..., maskva]
    intsObOb = intsMO[1][..., maskob, :][..., maskob]
    intsObVb = intsMO[1][..., maskob, :][..., maskvb]
    intsVbVb = intsMO[1][..., maskvb, :][..., maskvb]
    return intsOaOa, intsOaVa, intsVaVa, intsObOb, intsObVb, intsVbVb


def _contract_multipole_restricted_MO(intsMOov, x):
    """ints is the integral tensor of a spin-independent operator"""
    X = np.stack(x, axis=0)

    pol = 2 * np.tensordot(X, intsMOov, axes=([1, 2], [-2, -1]))
    return -pol


def _contract_multipole_unrestricted_MO(intsMOaov, intsMObov, x):
    """Handle contract multipole unrestricted MO internally."""
    Xa = np.stack([xv[0] for xv in x], axis=0)
    Xb = np.stack([xv[1] for xv in x], axis=0)

    pola = np.tensordot(Xa, intsMOaov, axes=([1, 2], [-2, -1]))
    polb = np.tensordot(Xb, intsMObov, axes=([1, 2], [-2, -1]))
    pol = pola + polb
    return -pol


def _contract_multipole_restricted_double_MO(intsMOoo, intsMOvv, static, x1, x2):
    """ints is the integral tensor of a spin-independent operator"""
    X1 = np.stack(x1, axis=0)
    X2c = np.stack([v.conj() for v in x2], axis=0)

    T1 = np.tensordot(
        np.tensordot(intsMOoo, X1, axes=([-2], [1])),
        X2c,
        axes=([-3, -1], [1, 2]),
    )
    T1 = np.moveaxis(T1, -1, 0)
    T1 = np.moveaxis(T1, -1, 0)

    T2 = -np.tensordot(
        np.tensordot(intsMOvv, X1, axes=([-2], [2])),
        X2c,
        axes=([-3, -1], [2, 1]),
    )
    T2 = np.moveaxis(T2, -1, 0)
    T2 = np.moveaxis(T2, -1, 0)

    T5 = np.tensordot(X1, X2c, axes=([1, 2], [1, 2]))
    T5 = np.expand_dims(T5, axis=tuple(i + 2 for i in range(static.ndim)))
    static = np.expand_dims(static, axis=(0, 1))
    T5 = T5 * static
    pol = 2 * (T1 + T2 + T5)
    return pol


def _contract_multipole_restricted(ints, x, mo_coeff, mo_occ):
    """ints is the integral tensor of a spin-independent operator"""

    orbo = mo_coeff[:, mo_occ == 2]
    orbv = mo_coeff[:, mo_occ == 0]

    ints = einsum("...pq,pi,qj->...ij", ints, orbo, orbv.conj(), optimize="optimal")
    X = np.stack(x, axis=0)
    pol = 2 * einsum("...ij,nij->n...", ints, X, optimize="optimal")
    return -pol


def _contract_multipole_unrestricted(ints, x, mo_coeff, mo_occ):
    """Handle contract multipole unrestricted internally."""
    orbo_a = mo_coeff[0][:, mo_occ[0] == 1]
    orbv_a = mo_coeff[0][:, mo_occ[0] == 0]
    orbo_b = mo_coeff[1][:, mo_occ[1] == 1]
    orbv_b = mo_coeff[1][:, mo_occ[1] == 0]

    ints_a = einsum(
        "...pq,pi,qj->...ij", ints, orbo_a, orbv_a.conj(), optimize="optimal"
    )
    ints_b = einsum(
        "...pq,pi,qj->...ij", ints, orbo_b, orbv_b.conj(), optimize="optimal"
    )
    Xa = np.stack([xv[0] for xv in x], axis=0)
    Xb = np.stack([xv[1] for xv in x], axis=0)
    pola = einsum("...ij,nij->n...", ints_a, Xa, optimize="optimal")
    polb = einsum("...ij,nij->n...", ints_b, Xb, optimize="optimal")
    pol = pola + polb
    return -pol


def static_multipole(ints, mo_coeff, mo_occ):
    """Process static multipole."""
    return einsum(
        "i,ji,ki,...jk->...",
        mo_occ,
        mo_coeff.conj(),
        mo_coeff,
        -ints,
        optimize="optimal",
    )


def static_multipole_unrestricted(ints, mo_coeff, mo_occ):
    """Process static multipole unrestricted."""
    return einsum(
        "ai,aji,aki,...jk->...",
        mo_occ,
        mo_coeff.conj(),
        mo_coeff,
        -ints,
        optimize="optimal",
    )


def _contract_multipole_restricted_double(ints, x1, x2, mo_coeff, mo_occ):
    """ints is the integral tensor of a spin-independent operator"""

    orbo = mo_coeff[:, mo_occ == 2]
    orbv = mo_coeff[:, mo_occ == 0]

    static = static_multipole(ints, mo_coeff, mo_occ)

    intsvv = einsum("...pq,pi,qj->...ij", ints, orbv, orbv.conj(), optimize="optimal")
    intsoo = einsum("...pq,pi,qj->...ij", ints, orbo, orbo.conj(), optimize="optimal")

    X1 = np.stack(x1, axis=0)
    X2c = np.stack([v.conj() for v in x2], axis=0)

    T1 = einsum("...ij,nip,mjp->nm...", intsoo, X1, X2c, optimize="optimal")
    T2 = einsum("...pq,nip,miq->nm...", -intsvv, X1, X2c, optimize="optimal")
    T5 = einsum("...,nip,mip->nm...", static, X1, X2c, optimize="optimal")
    pol = 2 * (T1 + T2 + T5)
    return pol


def _contract_multipole_unrestricted_double(ints, x1, x2, mo_coeff, mo_occ):
    """Handle contract multipole unrestricted double internally."""
    raise NotImplementedError()


def to_soc(ints, soc_vectors):
    """
    Rotate CIS transition dipoles into SOC basis via
    μ^SOC_c(u,v) = U_{u k} μ^CIS_c(k,l) U_{v l}^T

    Parameters
    ----------
    soc_vectors : ndarray, shape (n_soc, n_cis)
        SOC eigenvectors U_{u k}.
    mu_cis : ndarray, shape (n_cis, n_cis, any)
        CIS transition dipole arrays for each Cartesian component.

    Returns
    -------
    mu_soc : ndarray, shape (n_soc, n_soc, any)
        Transition dipole matrix in SOC basis.
    """

    ints = np.tensordot(soc_vectors.conj(), ints, axes=(0, 0))
    ints = np.tensordot(ints, soc_vectors, axes=(1, 0))
    ints = np.moveaxis(ints, -1, 1)
    return ints
