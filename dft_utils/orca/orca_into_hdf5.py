import argparse
import gc
import os
from collections.abc import Iterable
from itertools import product

import h5py
import numpy as np

from dft_utils.chkfile import load, save
from dft_utils.orca.file_utils import (
    get_tar_gz_parsers_per_mol,
    remove_directory_pathlib,
    safe_get_energy_or_nan,
)
from dft_utils.orca.reader import find_all_grouped_jobs
from dft_utils.units import Units

try:
    from pyscf.lib import einsum
except:
    print("Could not find PySCF einsum. Falling back to numpy.")
    from numpy import einsum


def parse_args():
    """Parse the supplied input."""
    parser = argparse.ArgumentParser(
        description="Run electrostatic simulation with specified parameters."
    )

    parser.add_argument(
        "folder", type=str, help="Path to the folder containing input data."
    )

    parser.add_argument(
        "--charges",
        type=str,
        nargs="+",
        default=["0", "1", "-1"],
        help="List of allowed charges (default: [0, 1, -1]).",
    )

    parser.add_argument(
        "--named",
        type=str,
        default=None,
        help="Name of molecule to analyse.",
    )

    parser.add_argument(
        "--remove", "-r", action="store_true", help="Enable cleaning up."
    )

    return parser.parse_args()


def to_mo(ints, mo_coeff):
    """Process to mo."""
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]
    ints = einsum(
        "xpq,pi,qj->xij", ints.reshape(-1, nao, nao), mo_coeff, mo_coeff.conj()
    )
    ints = ints.reshape(*pol_shape, *ints.shape[-2:])
    return ints


def to_mo_occvirt(ints, orbo, orbv):
    """Process to mo occvirt."""
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]
    ints = einsum("xpq,pi,qj->xij", ints.reshape(-1, nao, nao), orbo, orbv.conj())
    ints = ints.reshape(*pol_shape, *ints.shape[-2:])
    return ints


def is_empty(x):
    """Return whether empty."""
    return (
        x is None
        or (hasattr(x, "__len__") and len(x) == 0)
        or (hasattr(x, "size") and x.size == 0)
    )


def main(
    folder,
    charges=["0", "1", "-1"],
    named=None,
    remove=True,
    skip_parsing=False,
):
    """Process main."""
    group = os.path.basename(folder)

    mparsers = get_tar_gz_parsers_per_mol(folder)
    for name, parsers in mparsers.items():
        print(named, name)
        if named is not None:
            if named != name:
                continue
        if not skip_parsing:
            for parser in parsers:
                if "reorg" in parser.name:
                    print("Parsing reorg", parser.name)
                    parser.extract_by_pattern(".txt")
                if "geoopt" in parser.name:
                    print("Parsing geoopt", parser.name)
                    parser.extract_by_pattern(".txt")

                if "exc" in parser.name:
                    print("Parsing exc", parser.name)
                    parser.extract_by_pattern(".gbw")
                    parser.extract_by_pattern(".densitiesinfo")
                    parser.extract_by_pattern(".cis")
                    parser.extract_by_pattern(".out2")
                    parser.extract_by_pattern(".txt")
                if "exrocis" in parser.name:
                    print("Parsing exrocis", parser.name)
                    parser.extract_by_pattern(".gbw")
                    parser.extract_by_pattern(".ci")
                    parser.extract_by_pattern(".out2")
                if "hess" in parser.name:
                    print("Parsing hess.", parser.name)
                    parser.extract_by_pattern(".hess")
                    parser.extract_by_pattern(".out2")
                if "nonrad" in parser.name:
                    print("Parsing hess.", parser.name)
                    parser.extract_by_pattern(".out2")
        tot_pars = find_all_grouped_jobs(folder)[name]

        hdf5_outfile = os.path.join(folder, name + ".hdf5")

        json_type = ".json"

        try:
            jobs = tot_pars

            if not hasattr(jobs, "jobs"):
                jobs = [jobs]
            else:
                jobs = jobs.jobs

            for job in jobs:
                print(job.name)
                try:
                    parsers = job.parsers[".gbw"]
                except:
                    continue
                parsers = [parsers] if not isinstance(parsers, Iterable) else parsers
                if not skip_parsing:
                    for parser in parsers:
                        try:
                            if "_exc_" in str(parser.path.name) and "Guess" not in str(
                                parser.path
                            ):
                                print(f"Converting to json {parser.path.name}")
                                mol = job.mol

                                newp = os.path.join(
                                    folder,
                                    parser.path.name.removesuffix(".gbw"),
                                    parser.path.name.removesuffix(".gbw") + json_type,
                                )
                                print(newp)

                                if not os.path.exists(newp):
                                    parser._convert_json()
                                else:
                                    print(f"Found JSON {name}")

                        except Exception as e:
                            print(f"Failed to convert jsons. {e}")

        except Exception as e:
            print(f"Failed to convert jsons. {e}")

        print(f"Going with {json_type}")
        tot_pars = find_all_grouped_jobs(folder)[name]
        gc.collect()

        save(hdf5_outfile, f"{name}/group/", group)

        reorg_energies = np.array(
            [
                [*iarr, safe_get_energy_or_nan(tot_pars.get_delta_energy, *iarr)]
                for iarr in product(charges, repeat=4)
            ]
        )

        save(
            hdf5_outfile,
            f"{name}/reorg_energies/",
            reorg_energies.astype(h5py.string_dtype(encoding="utf-8")),
        )
        total_energies = np.array(
            [
                [*iarr, safe_get_energy_or_nan(tot_pars.get_total_energy, *iarr)]
                for iarr in product(charges, repeat=2)
            ]
        )

        save(
            hdf5_outfile,
            f"{name}/total_energies/",
            total_energies.astype(h5py.string_dtype(encoding="utf-8")),
        )

        for charge in charges:
            try:
                col_jobs = tot_pars.get_jobs_by_charge(charge)

                coords = Units.convert(col_jobs.mol_coords, "Angstrom", "au")
                save(hdf5_outfile, f"{name}/{charge}/coords", coords)

                masses = np.array(col_jobs.mol_masses)
                save(hdf5_outfile, f"{name}/{charge}/masses", masses)

                try:
                    hessian = col_jobs.hessian
                    save(hdf5_outfile, f"{name}/{charge}/hessian", hessian)
                except Exception as e:
                    print("Did not get hessian", name, charge, e)

                try:
                    ZPTE = col_jobs.get_jobs_by_type("hess").ZPTE
                    save(hdf5_outfile, f"{name}/{charge}/ZPTE", ZPTE)

                    ZPVE = col_jobs.get_jobs_by_type("hess").ZPVE
                    save(hdf5_outfile, f"{name}/{charge}/ZPVE", ZPVE)
                except Exception as e:
                    print("Did not get ZPTE or ZPVE", name, charge, e)

                mo_energy = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].mo_energy
                )
                save(hdf5_outfile, f"{name}/{charge}/mo_energy", mo_energy)

                mo_occ = col_jobs.get_jobs_by_type("exc").parsers[json_type].mo_occ
                save(hdf5_outfile, f"{name}/{charge}/mo_occ", mo_occ)

                mo_coeff = col_jobs.get_jobs_by_type("exc").parsers[json_type].mo_coeff
                save(hdf5_outfile, f"{name}/{charge}/mo_coeff", mo_coeff)

                dipole_AO = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].dipole_AO
                )
                save(hdf5_outfile, f"{name}/{charge}/dipole", dipole_AO)

                dipoleRel_AO = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].dipoleRel_AO
                )
                save(hdf5_outfile, f"{name}/{charge}/dipoleRel", dipoleRel_AO)

                quadrupole_AO = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].quadrupole_AO
                )
                save(hdf5_outfile, f"{name}/{charge}/quadrupole", quadrupole_AO)

                quadrupoleRel_AO = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].quadrupoleRel_AO
                )
                save(hdf5_outfile, f"{name}/{charge}/quadrupoleRel", quadrupoleRel_AO)

                angular_momentum = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].angular_momentum
                )
                save(
                    hdf5_outfile, f"{name}/{charge}/angular_momentum", angular_momentum
                )

                magnetic_quadrupole = (
                    col_jobs.get_jobs_by_type("exc")
                    .parsers[json_type]
                    .magnetic_quadrupole
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/magnetic_quadrupole",
                    magnetic_quadrupole,
                )

                velocity = col_jobs.get_jobs_by_type("exc").parsers[json_type].velocity
                save(hdf5_outfile, f"{name}/{charge}/velocity", velocity)

                electric_quadrupole_velocity = (
                    col_jobs.get_jobs_by_type("exc")
                    .parsers[json_type]
                    .electric_quadrupole_velocity
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/electric_quadrupole_velocity",
                    electric_quadrupole_velocity,
                )

                electric_octupole_length = (
                    col_jobs.get_jobs_by_type("exc")
                    .parsers[json_type]
                    .electric_octupole_length
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/electric_octupole_length",
                    electric_octupole_length,
                )

                electric_octupole_velocity = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].overlap_AO
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/electric_octupole_velocity",
                    electric_octupole_velocity,
                )

                s_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].s_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/s_matrix",
                    s_matrix,
                )

                t_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].t_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/t_matrix",
                    t_matrix,
                )

                h_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].h_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/h_matrix",
                    h_matrix,
                )

                v_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].v_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/v_matrix",
                    v_matrix,
                )

                f_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].f_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/f_matrix",
                    f_matrix,
                )

                j_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].j_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/j_matrix",
                    j_matrix,
                )

                k_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].k_matrix
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/k_matrix",
                    k_matrix,
                )

                vxc_matrix = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].vxc_matrix
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/vxc_matrix",
                    vxc_matrix,
                )

                vsol_matrix = (
                    col_jobs.get_jobs_by_type("exc").parsers[json_type].vsol_matrix
                )
                save(
                    hdf5_outfile,
                    f"{name}/{charge}/vsol_matrix",
                    vsol_matrix,
                )

                try:
                    nacmes = col_jobs.get_jobs_by_type("nonrad").nacmes_per_mult
                    save(hdf5_outfile, f"{name}/{charge}/nacmes", nacmes)
                except Exception as e:
                    print(f"Did not manage to get NACMEs. Error: {e}.")

                try:
                    calcstatd = col_jobs.get_jobs_by_type("exc").output_dipole_moment
                    save(hdf5_outfile, f"{name}/{charge}/calcstatd", calcstatd)
                except Exception as e:
                    print(
                        f"Did not manage to get the calculated dipole moment. Error: {e}."
                    )

                exc_grads = col_jobs.get_jobs_by_type(
                    "exc"
                ).excitonic_gradients_per_mult
                save(hdf5_outfile, f"{name}/{charge}/exc_grads", exc_grads)

                nucd = col_jobs.get_jobs_by_type("exc").nuclear_dipole
                save(hdf5_outfile, f"{name}/{charge}/nucd", nucd)

                statd = col_jobs.get_jobs_by_type("exc").multipole_staticMO_self(
                    "dipole_AO"
                )
                save(hdf5_outfile, f"{name}/{charge}/statd", statd)

                statq = col_jobs.get_jobs_by_type("exc").multipole_staticMO_self(
                    "quadrupole_AO"
                )
                save(hdf5_outfile, f"{name}/{charge}/statq", statq)

                stato = col_jobs.get_jobs_by_type("exc").multipole_staticMO_self(
                    "electric_octupole_length"
                )
                save(hdf5_outfile, f"{name}/{charge}/stato", stato)

                statangmom = col_jobs.get_jobs_by_type("exc").multipole_staticMO_self(
                    "angular_momentum"
                )
                save(hdf5_outfile, f"{name}/{charge}/statangmom", statangmom)

                statmq = col_jobs.get_jobs_by_type("exc").multipole_staticMO_self(
                    "magnetic_quadrupole"
                )
                save(hdf5_outfile, f"{name}/{charge}/statmq", statmq)

                cissed = col_jobs.get_jobs_by_type("exc").parsers[".cis"].parse()
                for i_exc, exc in enumerate(cissed):
                    save(hdf5_outfile, f"{name}/{charge}/CIS/{i_exc}/outa", exc[4])
                    save(hdf5_outfile, f"{name}/{charge}/CIS/{i_exc}/outb", exc[5])
                    save(hdf5_outfile, f"{name}/{charge}/CIS/{i_exc}/mult", exc[6])
                    save(hdf5_outfile, f"{name}/{charge}/CIS/{i_exc}/en", exc[0][0])

                if str(charge) == "0":
                    mat_SO = col_jobs.get_jobs_by_type("exc").parsers[".out2"].mat_SO
                    save(hdf5_outfile, f"{name}/{charge}/mat_SO", mat_SO)

                    eigvecs_SO = (
                        col_jobs.get_jobs_by_type("exc").parsers[".out2"].eigvecs_SO
                    )
                    save(hdf5_outfile, f"{name}/{charge}/eigvecs_SO", eigvecs_SO)

                    soc_matrix = col_jobs.get_jobs_by_type("exc").parsers[json_type].soc
                    save(
                        hdf5_outfile,
                        f"{name}/{charge}/soc_matrix",
                        soc_matrix,
                    )

                    socRel_matrix = (
                        col_jobs.get_jobs_by_type("exc").parsers[json_type].socRel
                    )
                    save(
                        hdf5_outfile,
                        f"{name}/{charge}/socRel_matrix",
                        socRel_matrix,
                    )

                    exc_en_S = col_jobs.get_jobs_by_type("exc").singlet_en
                    save(hdf5_outfile, f"{name}/{charge}/exc_en_S", exc_en_S)

                    exc_en_T = col_jobs.get_jobs_by_type("exc").triplet_en
                    save(hdf5_outfile, f"{name}/{charge}/exc_en_T", exc_en_T)

                    tdms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "dipole_AO"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tdms_S", tdms_S)

                    tqms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "quadrupole_AO"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tqms_S", tqms_S)

                    toms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "electric_octupole_length"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/toms_S", toms_S)

                    angmoms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "angular_momentum"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/angmoms_S", angmoms_S)

                    g_s = 2.0
                    g_l = 1.0
                    tmdms_S = 0.5 * (
                        g_l * load(hdf5_outfile, f"{name}/{charge}/angmoms_S")
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmdms_S", tmdms_S)

                    tmqms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "magnetic_quadrupole"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmqms_S", tmqms_S)

                    exc_en_phosp = (
                        col_jobs.get_jobs_by_type("exc").parsers[".out2"].E_exc_SO
                    )
                    save(hdf5_outfile, f"{name}/{charge}/exc_en_phosp", exc_en_phosp)

                    soc_grads = col_jobs.get_jobs_by_type("exc").soc_gradients
                    save(hdf5_outfile, f"{name}/{charge}/soc_grads", soc_grads)

                    tdms_soc, (__, __, tdms_SS, tdms_TT, __) = (
                        col_jobs.get_jobs_by_type(
                            "exc"
                        ).multipole_multipole_SOSO_MO_self("dipole_AO", return_all=True)
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tdms_soc", tdms_soc)
                    save(hdf5_outfile, f"{name}/{charge}/tdms_SS", tdms_SS)
                    save(hdf5_outfile, f"{name}/{charge}/tdms_TT", tdms_TT)

                    tqms_soc, (__, __, tqms_SS, tqms_TT, __) = (
                        col_jobs.get_jobs_by_type(
                            "exc"
                        ).multipole_multipole_SOSO_MO_self(
                            "quadrupole_AO", return_all=True
                        )
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tqms_soc", tqms_soc)
                    save(hdf5_outfile, f"{name}/{charge}/tqms_SS", tqms_SS)
                    save(hdf5_outfile, f"{name}/{charge}/tqms_TT", tqms_TT)

                    toms_soc = col_jobs.get_jobs_by_type(
                        "exc"
                    ).multipole_multipole_SOSO_MO_self(
                        "electric_octupole_length", return_all=False
                    )
                    save(hdf5_outfile, f"{name}/{charge}/toms_soc", toms_soc)

                    toms_vel_soc = col_jobs.get_jobs_by_type(
                        "exc"
                    ).multipole_multipole_SOSO_MO_self(
                        "electric_octupole_velocity", return_all=False
                    )
                    save(hdf5_outfile, f"{name}/{charge}/toms_vel_soc", toms_vel_soc)

                    angmoms_soc, (
                        __,
                        __,
                        angmoms_SS,
                        angmoms_TT,
                        __,
                    ) = col_jobs.get_jobs_by_type(
                        "exc"
                    ).multipole_multipole_SOSO_MO_self(
                        "angular_momentum", return_all=True
                    )
                    save(hdf5_outfile, f"{name}/{charge}/angmoms_soc", angmoms_soc)
                    save(hdf5_outfile, f"{name}/{charge}/angmoms_SS", angmoms_SS)
                    save(hdf5_outfile, f"{name}/{charge}/angmoms_TT", angmoms_TT)

                    tmqms_soc = col_jobs.get_jobs_by_type(
                        "exc"
                    ).multipole_multipole_SOSO_MO_self(
                        "magnetic_quadrupole", return_all=False
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmqms_soc", tmqms_soc)

                    spins_soc = col_jobs.get_jobs_by_type(
                        "exc"
                    ).multipole_multipole_SOSO_MO_self("spin", return_all=False)
                    save(hdf5_outfile, f"{name}/{charge}/spins_soc", spins_soc)

                    g_s = 2.0
                    g_l = 1.0
                    tmdms_soc = 0.5 * (
                        g_l * load(hdf5_outfile, f"{name}/{charge}/angmoms_soc")
                        + g_s * load(hdf5_outfile, f"{name}/{charge}/spins_soc")
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmdms_soc", tmdms_soc)
                else:
                    exc_en_S = col_jobs.get_jobs_by_type("exc").all_en
                    save(hdf5_outfile, f"{name}/{charge}/exc_en_S", exc_en_S)

                    tdms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "dipole_AO"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tdms_S", tdms_S)

                    tqms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "quadrupole_AO"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tqms_S", tqms_S)

                    toms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "electric_octupole_length"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/toms_S", toms_S)

                    angmoms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "angular_momentum"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/angmoms_S", angmoms_S)

                    g_s = 2.0
                    g_l = 1.0
                    tmdms_S = 0.5 * (
                        g_l * load(hdf5_outfile, f"{name}/{charge}/angmoms_S")
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmdms_S", tmdms_S)

                    tmqms_S = col_jobs.get_jobs_by_type("exc").multipole_GSMO_self(
                        "magnetic_quadrupole"
                    )
                    save(hdf5_outfile, f"{name}/{charge}/tmqms_S", tmqms_S)

            except Exception as e:
                print(name, charge, e)

        if remove:
            for parser in parsers:
                extraction_dir = parser.path.parent / parser.name
                print(f"Removing {extraction_dir}")
                remove_directory_pathlib(extraction_dir)

        if named is not None:
            return tot_pars


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(
        folder=args.folder,
        charges=args.charges,
        remove=args.remove,
        named=args.named,
    )
