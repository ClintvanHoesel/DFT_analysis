import argparse
import os
import re
from datetime import timedelta
from typing import List

import h5py
import numpy as np

from dft_utils.adaptive_utils import get_default_runner_kwargs, partial_deepcopy
from dft_utils.chkfile import load, save
from dft_utils.colour_utils import get_RGB_colour_from_spectrum
from dft_utils.frank_condon import Franck_Condon
from dft_utils.multipole_calculators import (
    dip_moment_calculator,
    list_to_symmetric_matrix_ORCA,
    quad_moment_calculator,
)
from dft_utils.spectrum_converter import (
    MP_to_Absorption,
    MP_to_Emission,
)
from dft_utils.units import Units

try:
    from pyscf.lib import einsum
except:
    print("Could not find PySCF einsum. Falling back to numpy.")
    from numpy import einsum


def parse_timedelta(time_str):
    """
    Parse a time string into a timedelta object.
    Supports formats like:
    - '5s', '30m', '2h', '1d' (single unit)
    - '1h30m', '2d12h30m15s' (multiple units)
    - '1:30:15' (HH:MM:SS format)
    - '90' (assumes seconds if no unit)
    """
    if not time_str:
        raise argparse.ArgumentTypeError("Empty time string")

    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            try:
                minutes, seconds = map(int, parts)
                return timedelta(minutes=minutes, seconds=seconds)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid MM:SS format: {time_str}")
        elif len(parts) == 3:
            try:
                hours, minutes, seconds = map(int, parts)
                return timedelta(hours=hours, minutes=minutes, seconds=seconds)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid HH:MM:SS format: {time_str}")
        else:
            raise argparse.ArgumentTypeError(f"Invalid time format: {time_str}")

    pattern = r"(\d+(?:\.\d+)?)\s*([smhd]?)"
    matches = re.findall(pattern, time_str.lower())

    if not matches:
        raise argparse.ArgumentTypeError(f"Invalid time format: {time_str}")

    total_seconds = 0

    for value_str, unit in matches:
        try:
            value = float(value_str)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid number: {value_str}")

        if unit == "" or unit == "s":
            total_seconds += value
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "h":
            total_seconds += value * 3600
        elif unit == "d":
            total_seconds += value * 86400
        else:
            raise argparse.ArgumentTypeError(f"Unknown time unit: {unit}")

    return timedelta(seconds=total_seconds)


def parse_args():
    """Parse the supplied input."""
    parser = argparse.ArgumentParser(
        description="Run electrostatic simulation with specified parameters."
    )

    parser.add_argument(
        "folder", type=str, help="Path to the folder containing input data."
    )

    parser.add_argument(
        "--named", type=str, default=None, help="Molecule to specifically check."
    )

    parser.add_argument(
        "--epsr", type=float, default=3.0, help="Relative permittivity (default: 3.0)."
    )
    parser.add_argument(
        "--mur", type=float, default=1.0, help="Relative permeability (default: 1.0)."
    )
    parser.add_argument(
        "--lambda_cl",
        type=float,
        default=0.01,
        help="Screening parameter λ_cl (default: 0.01).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300.0).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=10.0,
        help="Clip of FC factors per phonon energy (default: 10.0).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of parallel workers (default: 16).",
    )
    parser.add_argument(
        "--loss",
        type=float,
        default=1e-3,
        help="Convergence loss threshold (default: 1e-3).",
    )
    parser.add_argument(
        "--charges",
        type=str,
        nargs="+",
        default=["0", "1", "-1", "utriplet"][::-1],
        help="List of allowed charges (default: [0, 1, -1]).",
    )
    parser.add_argument(
        "--timedelta",
        type=parse_timedelta,
        default=timedelta(minutes=5),
        help="Interval between operations",
    )

    parser.add_argument("--skip", "-s", action="store_false", help="Enable skipping.")

    return parser.parse_args()


def to_mo(ints, orbo, orbv):
    """Process to mo."""
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]
    ints = einsum("xpq,pi,qj->xij", ints.reshape(-1, nao, nao), orbo, orbv.conj())
    ints = ints.reshape(*pol_shape, *ints.shape[-2:])


def get_krISC(
    hdf5_file,
    name,
    iTlist: List[int] = list(range(6)),
    iSlist: List[int] = list(range(3)),
    vib: bool = True,
    epsr=3.0,
    lambda_cl=0.01,
    temperature=300.0,
    clip=10.0,
):
    """Return krISC."""
    charge = 0
    if vib:
        vib_string = "vib"
    else:
        vib_string = "novib"

    basesave_folder = f"{name}/{charge}/{vib_string}"

    hessian = load(hdf5_file, f"{name}/{charge}/hessian")
    hessian = Units.convert(hessian, "au", "J/m^2")

    coords = load(hdf5_file, f"{name}/{charge}/coords")
    coords = Units.convert(coords, "au", "m")

    masses = load(hdf5_file, f"{name}/{charge}/masses")
    masses = Units.convert(masses, "au", "kg")

    exc_gradsS = load(hdf5_file, f"{name}/{charge}/exc_grads/SINGLET")
    exc_gradsT = load(hdf5_file, f"{name}/{charge}/exc_grads/TRIPLET")

    matSO = load(hdf5_file, f"{name}/{charge}/mat_SO")

    n_exc = (matSO.shape[0] - 1) // 4

    mat_SOS = matSO[: 1 + n_exc, : 1 + n_exc]
    eigvalsS, eigvecsS = np.linalg.eigh(mat_SOS)
    ind = np.argsort(np.real(eigvalsS))
    eigvalsS = eigvalsS[ind]
    eigvecsS = eigvecsS[:, ind]
    eigvecsStot = np.eye(matSO.shape[0], dtype=complex)
    eigvecsStot[: 1 + n_exc, : 1 + n_exc] = eigvecsS
    matSO = eigvecsStot.T.conj() @ matSO @ eigvecsStot

    mat_SOT = matSO[1 + n_exc :, 1 + n_exc :]
    eigvalsT, eigvecsT = np.linalg.eigh(mat_SOT)
    ind = np.argsort(np.real(eigvalsT))
    eigvalsT = eigvalsT[ind]
    eigvecsT = eigvecsT[:, ind]
    eigvecsTtot = np.eye(matSO.shape[0], dtype=complex)
    eigvecsTtot[1 + n_exc :, 1 + n_exc :] = eigvecsT
    matSO = eigvecsTtot.T.conj() @ matSO @ eigvecsTtot

    matSO = Units.convert(matSO, "au", "J")

    eigvalsT = eigvalsT - eigvalsS[0]
    eigvalsS = eigvalsS - eigvalsS[0]
    exc_enS = Units.convert(eigvalsS, "au", "J")
    exc_enS = np.real(exc_enS)
    exc_enT = Units.convert(eigvalsT, "au", "J")
    exc_enT = np.real(exc_enT)

    if vib:
        gsgrads = np.zeros((1, *exc_gradsS.shape[1:]))
        exc_gradsS = np.concatenate((gsgrads, exc_gradsS))
        exc_gradsS = np.tensordot(
            np.real(eigvecsS * eigvecsS.conj()), exc_gradsS, axes=(0, 0)
        )
        exc_gradsS = Units.convert(exc_gradsS, "au", "J/m")
        exc_gradsS = exc_gradsS.reshape(exc_gradsS.shape[0], -1)
        exc_gradsS = np.real(exc_gradsS)

        exc_gradsT = np.tile(exc_gradsT, (3, 1, 1))
        exc_gradsT = np.tensordot(
            np.real(eigvecsT * eigvecsT.conj()), exc_gradsT, axes=(0, 0)
        )
        exc_gradsT = Units.convert(exc_gradsT, "au", "J/m")
        exc_gradsT = exc_gradsT.reshape(exc_gradsT.shape[0], -1)
        exc_gradsT = np.real(exc_gradsT)

    if vib:
        fckwargsS = {
            "masses": masses,
            "_gradients": exc_gradsS,
            "hessian": hessian,
            "coords": coords,
            "constants": np.array([0.0]),
            "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
            "eps_r": epsr,
            "temperature": temperature,
            "clip": clip,
        }
        fcS = Franck_Condon(
            **fckwargsS,
        )
        exc_enS_corr = fcS.correct_energies(exc_enS)
        save(
            hdf5_file,
            f"{name}/{charge}/exc_en_Sphosp_corr",
            Units.convert(exc_enS_corr, "J", "au"),
        )

        fckwargsT = {
            "masses": masses,
            "_gradients": exc_gradsT,
            "hessian": hessian,
            "coords": coords,
            "constants": np.array([0.0]),
            "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
            "eps_r": epsr,
            "temperature": temperature,
            "clip": clip,
        }
        fcT = Franck_Condon(
            **fckwargsT,
        )
        exc_enT_corr = fcT.correct_energies(exc_enT)
        save(
            hdf5_file,
            f"{name}/{charge}/exc_en_Tphosp_corr",
            Units.convert(exc_enT_corr, "J", "au"),
        )

    save(
        hdf5_file,
        basesave_folder + "/lambda_cl",
        Units.convert(lambda_cl, "eV", "au"),
    )
    save(hdf5_file, basesave_folder + "/epsr", epsr)
    save(hdf5_file, basesave_folder + "/temperature", temperature)

    for iT in iTlist:
        save_folder = basesave_folder + f"/kRISC/{iT}"
        consts = matSO[1 + n_exc + iT, 1 : 1 + n_exc]
        consts = np.real(consts * consts.conj())

        fckwargs = {
            "energies": (
                exc_enS_corr[1:] - exc_enT_corr[iT]
                if vib
                else exc_enS[1:] - exc_enT[iT]
            ),
            "masses": masses,
            "_gradients": exc_gradsS[1:] - exc_gradsT[iT] if vib else None,
            "hessian": hessian,
            "coords": coords,
            "constants": consts,
            "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
            "eps_r": epsr,
            "temperature": temperature,
            "clip": clip,
        }
        if not vib:
            fckwargs["lambdas"] = None

        fc = Franck_Condon(
            **fckwargs,
        )

        f = fc.setup_spectrum("totj_abs")
        kRISC = f(0.0)
        save(hdf5_file, save_folder, kRISC)

    for iS in iSlist:
        save_folder = basesave_folder + f"/kISC/{iS}"
        consts = matSO[iS + 1, 1 + n_exc :]
        consts = np.real(consts * consts.conj())

        fckwargs = {
            "energies": (
                exc_enT_corr - exc_enS_corr[1:][iS]
                if vib
                else exc_enT - exc_enS[1:][iS]
            ),
            "masses": masses,
            "_gradients": exc_gradsT - exc_gradsS[1:][iS] if vib else None,
            "hessian": hessian,
            "coords": coords,
            "constants": consts,
            "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
            "eps_r": epsr,
            "temperature": temperature,
            "clip": clip,
        }
        if not vib:
            fckwargs["lambdas"] = None

        fc = Franck_Condon(
            **fckwargs,
        )

        f = fc.setup_spectrum("totj_abs")
        kISC = f(0.0)
        save(hdf5_file, save_folder, kISC)


def get_knonrad(
    hdf5_file,
    name,
    iexclist: List[int] = list(range(6)),
    vib: bool = True,
    epsr=3.0,
    lambda_cl=0.01,
    temperature=300.0,
    clip=10.0,
    phosp=True,
    charge=0,
):
    """Return knonrad."""
    if vib:
        vib_string = "vib"
    else:
        raise ValueError(
            "Are you really trying to calculate non-radiative decay with vibrations?"
        )
        vib_string = "novib"

    basesave_folder = f"{name}/{charge}/{vib_string}"

    hessian = load(hdf5_file, f"{name}/{charge}/hessian")
    hessian = Units.convert(hessian, "au", "J/m^2")

    coords = load(hdf5_file, f"{name}/{charge}/coords")
    coords = Units.convert(coords, "au", "m")

    masses = load(hdf5_file, f"{name}/{charge}/masses")
    masses = Units.convert(masses, "au", "kg")

    nacmes = load(hdf5_file, f"{name}/{charge}/nacmes/SINGLET")

    if phosp:

        exc_grads = load(hdf5_file, f"{name}/{charge}/soc_grads")
        exc_grads = exc_grads[1:] - exc_grads[0]

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_phosp")

        eigvecs_SO = load(hdf5_file, f"{name}/{charge}/eigvecs_SO")
        n_exc = (eigvecs_SO.shape[0] - 1) // 4
        assert n_exc >= nacmes.shape[0]
        eigvecs_SO = eigvecs_SO[1 : 1 + nacmes.shape[0], :]
        nacmes = np.tensordot(eigvecs_SO, nacmes, axes=(0, 0))[1:]
    else:

        exc_grads = load(hdf5_file, f"{name}/{charge}/exc_grads/SINGLET")

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_S")

    exc_grads = Units.convert(exc_grads, "au", "J/m")
    exc_grads = exc_grads.reshape(exc_grads.shape[0], -1)
    exc_grads = np.real(exc_grads)

    nacmes = Units.nconvert(nacmes, "au", "m", -1)
    nacmes = nacmes.reshape(nacmes.shape[0], -1)

    exc_en = Units.convert(exc_en, "au", "J")
    exc_en = np.real(exc_en)

    fckwargs = {
        "masses": masses,
        "_gradients": exc_grads,
        "hessian": hessian,
        "coords": coords,
        "nacmes": nacmes,
        "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
        "eps_r": epsr,
        "temperature": temperature,
        "clip": clip,
    }
    fc = Franck_Condon(**fckwargs)
    save(
        hdf5_file,
        f"{basesave_folder}/phonon_energies",
        Units.convert(fc.phonon_energies, "J", "au"),
    )
    save(
        hdf5_file,
        f"{basesave_folder}/reorganization_energies",
        Units.convert(fc.reorganization_energies, "J", "au"),
    )
    fc.energies = fc.correct_energies(exc_en)
    if phosp:
        save(
            hdf5_file,
            f"{basesave_folder}/exc_en_phosp_corr",
            Units.convert(fc.energies, "J", "au"),
        )
    else:
        save(
            hdf5_file,
            f"{basesave_folder}/exc_en_S_corr",
            Units.convert(fc.energies, "J", "au"),
        )
    save(
        hdf5_file,
        basesave_folder + "/lambda_cl",
        Units.convert(lambda_cl, "eV", "au"),
    )
    save(hdf5_file, basesave_folder + "/epsr", epsr)
    save(hdf5_file, basesave_folder + "/temperature", temperature)
    return fc

    for iexc in iexclist:
        fckwargs = {
            "masses": masses,
            "_gradients": exc_grads[iexc],
            "hessian": hessian,
            "coords": coords,
            "nacmes": nacmes[iexc],
            "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
            "eps_r": epsr,
            "temperature": temperature,
            "clip": clip,
            "energies": fc.energies[iexc],
        }
        fc = Franck_Condon(
            **fckwargs,
        )

    f = fc.setup_spectrum("nonrad")
    kISC = f(0.0)
    save(hdf5_file, basesave_folder + f"/{iexc}/kISC", kISC)


def calc_single_excited(
    hdf5_file,
    name,
    charge,
    phosp,
    emis,
    order,
    magn,
    i_exc: int,
    epsr=3.0,
    lambda_cl=0.01,
    temperature=300.0,
    clip=10.0,
    max_workers=16,
    timedelta=timedelta(minutes=1),
    runner_kwargs=None,
    skip=True,
    loss=1e-3,
    vib=True,
    mur=1.0,
):
    """Process calc single excited."""
    if phosp and str(charge) != "0":
        raise ValueError("Type phosp and charge != 0 cannot go at the same time.")

    if runner_kwargs is None:
        runner_kwargs = get_default_runner_kwargs(
            dt=timedelta, loss=loss, max_workers=max_workers
        )

    if not (phosp and (str(charge) == "0")):
        raise NotImplementedError("Can only do SOC for charge=0.")

    if not magn:
        if order == 1:
            order_string = "tdms"
        elif order == 2:
            order_string = "tqms"
        elif order == 3:
            order_string = "toms"
    elif magn:
        if order == 1:
            order_string = "tmdms"
        elif order == 2:
            order_string = "tmqms"
        elif order == 3:
            order_string = "tmoms"

    if phosp:
        phosp_string = "phosp"
    else:
        phosp_string = "fluor"

    if vib:
        vib_string = "vib"
    else:
        vib_string = "novib"

    if emis:
        em_string = "emission_spectrum"
    else:
        em_string = "absorption_spectrum"

    save_folder = f"{name}/{charge}/{phosp_string}/{em_string}/{vib_string}/{order_string}/{i_exc}"
    if skip:
        with h5py.File(hdf5_file, "r") as f:
            group_name = save_folder + "/y_spec"
            if group_name in f:
                print(f"Group {group_name} exists")
                return
            else:
                print(f"Group {group_name} does not exist")

    hessian = load(hdf5_file, f"{name}/{charge}/hessian")
    hessian = Units.convert(hessian, "au", "J/m^2")

    coords = load(hdf5_file, f"{name}/{charge}/coords")
    coords = Units.convert(coords, "au", "m")

    masses = load(hdf5_file, f"{name}/{charge}/masses")
    masses = Units.convert(masses, "au", "kg")

    if phosp:

        exc_grads = load(hdf5_file, f"{name}/{charge}/soc_grads")
        exc_grads = exc_grads[1:] - exc_grads[0]
        exc_gradsi = exc_grads - exc_grads[i_exc]

        tdms = load(hdf5_file, f"{name}/{charge}/{order_string}_soc")[i_exc + 1, 1:, :]

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_phosp")
    else:

        exc_grads = load(hdf5_file, f"{name}/{charge}/exc_grads/SINGLET")

        tdms = load(hdf5_file, f"{name}/{charge}/{order_string}_S")

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_S")

    exc_grads = Units.convert(exc_grads, "au", "J/m")
    exc_grads = exc_grads.reshape(exc_grads.shape[0], -1)
    exc_grads = np.real(exc_grads)

    exc_gradsi = Units.convert(exc_gradsi, "au", "J/m")
    exc_gradsi = exc_gradsi.reshape(exc_gradsi.shape[0], -1)
    exc_gradsi = np.real(exc_gradsi)

    tdms = Units.nconvert(
        Units.convert(
            tdms,
            "au",
            "C_m",
        ),
        "au",
        "m",
        order - 1,
    )
    if magn:
        tdms = Units.convert(
            tdms,
            "au",
            "m/s",
        )

    if order == 1:
        tdms_sq = dip_moment_calculator(tdms)
    elif order == 2:
        tdms = list_to_symmetric_matrix_ORCA(tdms)
        tdms_sq = quad_moment_calculator(tdms)

    else:
        NotImplementedError()

    exc_en = Units.convert(exc_en, "au", "J")
    exc_en = np.real(exc_en)

    fckwargs = {
        "masses": masses,
        "_gradients": exc_grads,
        "hessian": hessian,
        "coords": coords,
        "constants": tdms_sq,
        "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
        "eps_r": epsr,
        "temperature": temperature,
        "clip": clip,
    }
    if not vib:
        fckwargs["lambdas"] = None

    fc = Franck_Condon(
        **fckwargs,
    )

    if vib:
        save(
            hdf5_file,
            f"{name}/{charge}/{phosp_string}/phonon_energies",
            Units.convert(fc.phonon_energies, "J", "au"),
        )
        save(
            hdf5_file,
            f"{name}/{charge}/{phosp_string}/reorganization_energies",
            Units.convert(fc.reorganization_energies, "J", "au"),
        )

    save(
        hdf5_file,
        save_folder + "/lambda_cl",
        Units.convert(lambda_cl, "eV", "au"),
    )
    save(hdf5_file, save_folder + "/epsr", epsr)
    save(hdf5_file, save_folder + "/temperature", temperature)

    if vib:
        exc_en0 = fc.correct_energies(exc_en)
        exc_eni = exc_en0[i_exc]

        mask_en = np.arange(len(exc_en0)) != i_exc
        exc_en = exc_en0[mask_en] - exc_eni

        exc_grads = exc_gradsi[mask_en]

        tdms_sq = tdms_sq[mask_en]

        if phosp:
            save(
                hdf5_file,
                f"{name}/{charge}/exc_en_phosp_corr",
                Units.convert(exc_en0, "J", "au"),
            )
        else:
            save(
                hdf5_file,
                f"{name}/{charge}/exc_en_S_corr",
                Units.convert(exc_en0, "J", "au"),
            )

    fckwargs = {
        "energies": exc_en,
        "masses": masses,
        "_gradients": exc_grads,
        "hessian": hessian,
        "coords": coords,
        "constants": tdms_sq,
        "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
        "eps_r": epsr,
        "temperature": temperature,
        "clip": clip,
    }
    if not vib:
        fckwargs["lambdas"] = None

    fc = Franck_Condon(
        **fckwargs,
    )

    if emis:
        E_interval = [
            0,
            np.min(fc.energies) + 30 * max(fc.lclkBT, fc.kBT),
        ]

        xe, ye = fc.adap_spectrum(
            "readied_emis",
            rkwargs=partial_deepcopy(runner_kwargs, ["executor"]),
            E_interval=E_interval,
        )
        y_emis = MP_to_Emission(xe, ye, mp_order=1, epsr=epsr, magn=magn, mur=mur).y
        x_emis = MP_to_Emission(xe, ye, mp_order=1, epsr=epsr, magn=magn, mur=mur).x

        wav = 1240 / Units.convert(x_emis, "J", "eV")
        rgb = get_RGB_colour_from_spectrum(wav, y_emis * 1240 / (wav * wav))

        save(
            hdf5_file,
            save_folder + "/x",
            xe,
        )
        save(
            hdf5_file,
            save_folder + "/y",
            ye,
        )
        save(
            hdf5_file,
            save_folder + "/x_spec",
            x_emis,
        )
        save(
            hdf5_file,
            save_folder + "/y_spec",
            y_emis,
        )
        save(
            hdf5_file,
            save_folder + "/rgb",
            rgb,
        )
    else:
        E_interval = [
            0,
            np.max(fc.energies),
        ]

        x, y = fc.adap_spectrum(
            "readied_abs",
            rkwargs=partial_deepcopy(runner_kwargs, ["executor"]),
            E_interval=E_interval,
        )
        x_abs = MP_to_Absorption(x, y, mp_order=order, epsr=epsr, magn=magn, mur=mur).x
        y_abs = MP_to_Absorption(x, y, mp_order=order, epsr=epsr, magn=magn, mur=mur).y

        save(
            hdf5_file,
            save_folder + "/x",
            x,
        )
        save(
            hdf5_file,
            save_folder + "/y",
            y,
        )
        save(
            hdf5_file,
            save_folder + "/x_spec",
            x_abs,
        )
        save(
            hdf5_file,
            save_folder + "/y_spec",
            y_abs,
        )


def calc_single(
    hdf5_file,
    name,
    charge,
    phosp,
    emis,
    order,
    magn,
    epsr=3.0,
    lambda_cl=0.01,
    temperature=300.0,
    clip=10.0,
    max_workers=16,
    timedelta=timedelta(minutes=1),
    runner_kwargs=None,
    skip=True,
    loss=1e-3,
    vib=True,
    mur=1.0,
):
    """Process calc single."""
    if phosp and str(charge) != "0":
        raise ValueError("Type phosp and charge != 0 cannot go at the same time.")

    if runner_kwargs is None:
        runner_kwargs = get_default_runner_kwargs(
            dt=timedelta, loss=loss, max_workers=max_workers
        )

    if not magn:
        if order == 1:
            order_string = "tdms"
        elif order == 2:
            order_string = "tqms"
        elif order == 3:
            order_string = "toms"
    elif magn:
        if order == 1:
            order_string = "tmdms"
        elif order == 2:
            order_string = "tmqms"
        elif order == 3:
            order_string = "tmoms"

    if phosp:
        phosp_string = "phosp"
    else:
        phosp_string = "fluor"

    if vib:
        vib_string = "vib"
    else:
        vib_string = "novib"

    if emis:
        em_string = "emission_spectrum"
    else:
        em_string = "absorption_spectrum"

    save_folder = (
        f"{name}/{charge}/{phosp_string}/{em_string}/{vib_string}/{order_string}"
    )
    if skip:
        with h5py.File(hdf5_file, "r") as f:
            group_name = save_folder + "/y_spec"
            if group_name in f:
                print(f"Group {group_name} exists")
                return
            else:
                print(f"Group {group_name} does not exist")

    hessian = load(hdf5_file, f"{name}/{charge}/hessian")
    hessian = Units.convert(hessian, "au", "J/m^2")

    coords = load(hdf5_file, f"{name}/{charge}/coords")
    coords = Units.convert(coords, "au", "m")

    masses = load(hdf5_file, f"{name}/{charge}/masses")
    masses = Units.convert(masses, "au", "kg")

    if phosp:

        exc_grads = load(hdf5_file, f"{name}/{charge}/soc_grads")
        exc_grads = exc_grads[1:, :, :] - exc_grads[0, :, :]

        tdms = load(hdf5_file, f"{name}/{charge}/{order_string}_soc")[0, 1:]

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_phosp")
    else:

        exc_grads = load(hdf5_file, f"{name}/{charge}/exc_grads/SINGLET")

        tdms = load(hdf5_file, f"{name}/{charge}/{order_string}_S")

        exc_en = load(hdf5_file, f"{name}/{charge}/exc_en_S")

    exc_grads = Units.convert(exc_grads, "au", "J/m")
    exc_grads = exc_grads.reshape(exc_grads.shape[0], -1)
    exc_grads = np.real(exc_grads)

    tdms = Units.nconvert(
        Units.convert(
            tdms,
            "au",
            "C_m",
        ),
        "au",
        "m",
        order - 1,
    )
    if magn:
        tdms = Units.convert(
            tdms,
            "au",
            "m/s",
        )

    if order == 1:
        tdms_sq = dip_moment_calculator(tdms)
    elif order == 2:
        tdms = list_to_symmetric_matrix_ORCA(tdms)
        tdms_sq = quad_moment_calculator(tdms)

    else:
        NotImplementedError()

    exc_en = Units.convert(exc_en, "au", "J")
    exc_en = np.real(exc_en)

    fckwargs = {
        "masses": masses,
        "_gradients": exc_grads,
        "hessian": hessian,
        "coords": coords,
        "constants": tdms_sq,
        "lambda_cl": Units.convert(lambda_cl, "eV", "J"),
        "eps_r": epsr,
        "temperature": temperature,
        "clip": clip,
    }
    if not vib:
        fckwargs["lambdas"] = None

    fc = Franck_Condon(
        **fckwargs,
    )

    if vib:
        save(
            hdf5_file,
            f"{name}/{charge}/{phosp_string}/phonon_energies",
            Units.convert(fc.phonon_energies, "J", "au"),
        )
        save(
            hdf5_file,
            f"{name}/{charge}/{phosp_string}/reorganization_energies",
            Units.convert(fc.reorganization_energies, "J", "au"),
        )

    save(
        hdf5_file,
        save_folder + "/lambda_cl",
        Units.convert(lambda_cl, "eV", "au"),
    )
    save(hdf5_file, save_folder + "/epsr", epsr)
    save(hdf5_file, save_folder + "/temperature", temperature)

    if vib:
        fc.energies = fc.correct_energies(exc_en)
        if phosp:
            save(
                hdf5_file,
                f"{name}/{charge}/exc_en_phosp_corr",
                Units.convert(fc.energies, "J", "au"),
            )
        else:
            save(
                hdf5_file,
                f"{name}/{charge}/exc_en_S_corr",
                Units.convert(fc.energies, "J", "au"),
            )
    else:
        fc.energies = exc_en

    if emis:
        E_interval = [
            0,
            np.min(fc.energies) + 30 * max(fc.lclkBT, fc.kBT),
        ]

        xe, ye = fc.adap_spectrum(
            "readied_emis",
            rkwargs=partial_deepcopy(runner_kwargs, ["executor"]),
            E_interval=E_interval,
        )
        y_emis = MP_to_Emission(xe, ye, mp_order=1, epsr=epsr, magn=magn, mur=mur).y
        x_emis = MP_to_Emission(xe, ye, mp_order=1, epsr=epsr, magn=magn, mur=mur).x

        wav = 1240 / Units.convert(x_emis, "J", "eV")
        rgb = get_RGB_colour_from_spectrum(wav, y_emis * 1240 / (wav * wav))

        save(
            hdf5_file,
            save_folder + "/x",
            xe,
        )
        save(
            hdf5_file,
            save_folder + "/y",
            ye,
        )
        save(
            hdf5_file,
            save_folder + "/x_spec",
            x_emis,
        )
        save(
            hdf5_file,
            save_folder + "/y_spec",
            y_emis,
        )
        save(
            hdf5_file,
            save_folder + "/rgb",
            rgb,
        )
    else:
        E_interval = [
            0,
            np.max(fc.energies),
        ]

        x, y = fc.adap_spectrum(
            "readied_abs",
            rkwargs=partial_deepcopy(runner_kwargs, ["executor"]),
            E_interval=E_interval,
        )
        x_abs = MP_to_Absorption(x, y, mp_order=order, epsr=epsr, magn=magn, mur=mur).x
        y_abs = MP_to_Absorption(x, y, mp_order=order, epsr=epsr, magn=magn, mur=mur).y

        save(
            hdf5_file,
            save_folder + "/x",
            x,
        )
        save(
            hdf5_file,
            save_folder + "/y",
            y,
        )
        save(
            hdf5_file,
            save_folder + "/x_spec",
            x_abs,
        )
        save(
            hdf5_file,
            save_folder + "/y_spec",
            y_abs,
        )


def main(
    folder,
    epsr=3.0,
    mur=1.0,
    lambda_cl=0.01,
    temperature=300.0,
    clip=10.0,
    max_workers=16,
    loss=1e-3,
    charges=["0", "1", "-1", "utriplet"],
    phosps=[True, False],
    emiss=[True, False],
    orders=[1, 2],
    magns=[False, True],
    vibs=[True, False],
    i_excs=list(range(15)),
    timedelta=timedelta(minutes=5),
    do_rISC=True,
    skip=True,
    named=None,
):
    """Process main."""
    runner_kwargs = get_default_runner_kwargs(
        dt=timedelta, loss=loss, max_workers=max_workers
    )
    files = os.listdir(folder)

    files = [os.path.splitext(f)[0] for f in files if os.path.splitext(f)[1] == ".hdf5"]

    for name in files:
        if named:
            if named != name:
                continue
        hdf5_outfile = os.path.join(folder, name + ".hdf5")

        for charge in charges:
            for phosp in phosps:
                if phosp and str(charge) != "0":
                    print("phosp cannot be true if charge is not 0.")
                    continue
                for emis in emiss:
                    if emis and str(charge) != "0":
                        print("emis cannot be true if charge is not 0.")
                        continue
                    for order in orders:
                        for magn in magns:
                            if order > 1 and magn:
                                continue
                            for vib in vibs:
                                print(
                                    hdf5_outfile,
                                    name,
                                    charge,
                                    phosp,
                                    emis,
                                    order,
                                    magn,
                                    vib,
                                )
                                try:
                                    calc_single(
                                        hdf5_outfile,
                                        name,
                                        charge,
                                        phosp,
                                        emis,
                                        order,
                                        magn,
                                        epsr=epsr,
                                        lambda_cl=lambda_cl,
                                        temperature=temperature,
                                        max_workers=max_workers,
                                        timedelta=timedelta,
                                        runner_kwargs=runner_kwargs,
                                        skip=skip,
                                        loss=loss,
                                        vib=vib,
                                        mur=mur,
                                    )
                                    print("Succes.")
                                except Exception as e:
                                    print("Failed. ", e)
                                if str(charge) == "0" and (phosp) and (not emis):
                                    for i_exc in i_excs:
                                        try:
                                            calc_single_excited(
                                                hdf5_outfile,
                                                name,
                                                charge,
                                                phosp,
                                                emis,
                                                order,
                                                magn,
                                                i_exc,
                                                epsr=epsr,
                                                lambda_cl=lambda_cl,
                                                temperature=temperature,
                                                max_workers=max_workers,
                                                timedelta=timedelta,
                                                runner_kwargs=runner_kwargs,
                                                skip=skip,
                                                loss=loss,
                                                vib=vib,
                                                mur=mur,
                                            )
                                            print("Succes.")
                                        except Exception as e:
                                            print("Failed. ", e)
        if do_rISC:
            try:
                get_krISC(hdf5_outfile, name)
            except Exception as e:
                print("Failed. ", e)


if __name__ == "__main__":
    args = parse_args()
    main(
        folder=args.folder,
        epsr=args.epsr,
        lambda_cl=args.lambda_cl,
        temperature=args.temperature,
        max_workers=args.max_workers,
        loss=args.loss,
        charges=args.charges,
        timedelta=args.timedelta,
        skip=args.skip,
        named=args.named,
    )
