"""
Created on Wed Mar  9 14:43:07 2022

@author: s164097
"""

import csv
import datetime
import os
import warnings

from .base_utils import ensure_folder, flatten


def write_lines_file(path, lines):
    """Process write lines file."""
    if os.path.exists(path):
        warnings.warn(f"Overwriting {path}", stacklevel=2)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def join_line(line, delim=","):
    """Process join line."""
    s = delim.join(str(v) for v in line)
    return s


def list_of_list_to_lines(lines, delim=","):
    """Process list of list to lines."""
    return [f"{join_line(line, delim)}\n" for line in lines]


def write_list_of_lists(
    list_of_list, path, header=None, units=None, comments=None, delim=","
):
    """Process write list of lists."""
    if comments:
        list_of_list = [comments] + list_of_list
    if units:
        list_of_list = [units] + list_of_list
    if header:
        list_of_list = [header] + list_of_list
    write_lines_file(path, list_of_list_to_lines(list_of_list, delim))


def fix_path_csv(path, name, folder, datstr):
    """Process fix path csv."""
    if folder:
        path = os.path.join(path, folder)
    ensure_folder(path)
    if name:
        path = os.path.join(path, name)
    if path[-4:] != ".csv":
        if datstr:
            path += f"_{datstr}"
        path += ".csv"
    return path


def write_csv(
    list_of_list,
    path=None,
    name="Unknown",
    folder=None,
    datstr=None,
    **kwargs,
):
    """Process write csv."""
    path = fix_path_csv(
        os.getcwd() if path is None else path,
        name,
        folder,
        (
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if datstr is None
            else datstr
        ),
    )

    write_list_of_lists(list_of_list, path, delim=",", **kwargs)


def write_csv_rowdict(
    d,
    path=None,
    name="Unknown",
    folder=None,
    datstr=None,
    **kwargs,
):
    """Process write csv rowdict."""
    path = fix_path_csv(
        os.getcwd() if path is None else path,
        name,
        folder,
        (
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if datstr is None
            else datstr
        ),
    )
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in d.items():
            try:
                row_data = [key, *list(flatten(value))]
            except TypeError:
                row_data = [key, value]
            writer.writerow(row_data)
    return path


def process_header_to_units(header):
    """Process process header to units."""
    units = [
        item.split("(")[-1].split(")")[0].strip() if "(" in item else " "
        for item in header
    ]
    header = [
        "(".join(item.split("(")[:-1]).strip() if "(" in item else item
        for item in header
    ]
    return header, units


def load_csv(file_path, delimiter=","):
    """
    Load a CSV file into a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.
        delimiter (str): The delimiter used in the CSV file (default is comma).

    Returns:
        list of dict: A list where each item is a dictionary representing a row in the CSV file.
    """
    try:
        with open(file_path, mode="r", newline="") as file:
            csv_reader = csv.DictReader(file, delimiter=delimiter)

            data = list(csv_reader)

        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except OSError as e:
        print(f"Error: {e}")
        return []
