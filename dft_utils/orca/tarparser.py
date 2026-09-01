import os
import re
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

from .FileParser import FileParser


def _safe_member_path(output_dir, member):
    """Return the extraction path for a validated archive member.

    Archive members are validated as POSIX paths because that is the format
    used by tar files. Windows path rules are checked as well so that an
    archive cannot become unsafe when processed on another operating system.
    Symbolic links, hard links, and special files are rejected because they
    can redirect extraction or create device-like filesystem entries.
    """
    name = member.name
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    invalid_parts = {".."}

    if (
        not name
        or "\x00" in name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in invalid_parts for part in posix_path.parts)
        or any(part in invalid_parts for part in windows_path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {name!r}")

    if member.issym() or member.islnk():
        raise ValueError(f"Links are not allowed in archives: {name!r}")
    if not member.isdir() and not member.isfile():
        raise ValueError(f"Unsupported archive member type: {name!r}")

    root = Path(output_dir).resolve()
    target = (root / Path(*posix_path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member path: {name!r}") from exc
    return target


def _extract_member(tar, member, output_dir):
    """Safely extract one regular file or directory and return its path."""
    target = _safe_member_path(output_dir, member)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read archive member: {member.name!r}")
    with source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return target


class TarGzParser(FileParser):
    def __init__(self, path: str):
        """Initialize the object."""
        super().__init__(path)
        self.name = self.path.stem.replace(".tar", "")

    def extract_all(self, output_dir=None):
        """
        Extract all files from the tar.gz archive to current_folder/self.name.

        Parameters:
        -----------
        output_dir : str or Path, optional
            Base directory where extraction folder will be created.
            If None, uses the parent directory of the tar.gz file.

        Returns:
        --------
        str
            Path to the extraction directory (current_folder/self.name).
        """

        if output_dir is None:
            base_dir = self.path.parent
        else:
            base_dir = Path(output_dir)

        extraction_dir = base_dir / self.name
        os.makedirs(extraction_dir, exist_ok=True)

        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                _safe_member_path(extraction_dir, member)
            for member in members:
                _extract_member(tar, member, extraction_dir)

        print(f"Extracted all files to: {extraction_dir}")
        return str(extraction_dir)

    def list_contents(self):
        """
        List all files and directories in the tar.gz archive.

        Returns:
        --------
        list
            List of member names in the archive.
        """
        with tarfile.open(self.path, "r:gz") as tar:
            return [member.name for member in tar.getmembers()]

    def extract_by_pattern(self, pattern, output_dir=None, skip=True):
        """
        Extract files matching a specific regex pattern.

        Parameters:
        -----------
        pattern : str or re.Pattern
            Regex pattern to match file names.
        output_dir : str or Path, optional
            Directory where extracted files will be saved.
            If None, uses current_folder/self.name.

        Returns:
        --------
        list
            List of paths to extracted files.
        """

        if output_dir is None:
            output_dir = self.path.parent / self.name
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        extracted_files = []

        output_dir = Path(output_dir)
        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()
            matched_members = [m for m in members if pattern.search(m.name)]

            for member in sorted(matched_members, key=lambda m: m.name):
                extracted_path = _safe_member_path(output_dir, member)
                if skip:
                    if extracted_path.exists():
                        continue
                _extract_member(tar, member, output_dir)
                now = time.time()
                os.utime(extracted_path, (now, now))
                extracted_files.append(str(extracted_path))

        return extracted_files

    def extract_by_extension(self, extension, output_dir=None):
        """
        Extract all files with a specific extension.

        Parameters:
        -----------
        extension : str
            File extension to match (e.g., '.txt', '.root', '.log').
        output_dir : str or Path, optional
            Directory where extracted files will be saved.
            If None, uses current_folder/self.name.

        Returns:
        --------
        list
            List of paths to extracted files.
        """

        if not extension.startswith("."):
            extension = "." + extension

        pattern = rf".*\{re.escape(extension)}$"
        return self.extract_by_pattern(pattern, output_dir)

    def get_archive_info(self):
        """
        Get detailed information about the tar.gz archive.

        Returns:
        --------
        dict
            Dictionary containing archive information.
        """
        info = {
            "path": str(self.path),
            "name": self.name,
            "size_bytes": self.path.stat().st_size,
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "largest_file": None,
            "largest_file_size": 0,
        }

        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()
            info["total_files"] = len([m for m in members if m.isfile()])
            info["total_directories"] = len([m for m in members if m.isdir()])

            for member in members:
                if member.isfile():

                    ext = Path(member.name).suffix.lower()
                    if ext:
                        info["file_types"][ext] = info["file_types"].get(ext, 0) + 1
                    else:
                        info["file_types"]["[no extension]"] = (
                            info["file_types"].get("[no extension]", 0) + 1
                        )

                    if member.size > info["largest_file_size"]:
                        info["largest_file_size"] = member.size
                        info["largest_file"] = member.name

        return info

    def extract_hess_root_files(self, output_dir=None, name=None):
        """
        Extract all files matching the pattern "{name}.hess.root{i}" from the tar.gz archive.

        Parameters:
        -----------
        output_dir : str, optional
            Directory where extracted files will be saved. If None, uses the current directory.
        name : str, optional
            The base name to look for. If None, extracts all .hess.root files regardless of name prefix.

        Returns:
        --------
        list
            List of paths to extracted files.
        """

        if output_dir is None:
            output_dir = self.path.parent

        os.makedirs(output_dir, exist_ok=True)

        if name is None:
            name = self.path.stem[:-4]
            name += ".ES"

        pattern = re.compile(rf"{re.escape(name)}\.hess\.root\d*$")

        extracted_files = []

        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()

            hess_root_files = [m for m in members if pattern.search(m.name)]
            for member in hess_root_files:
                _safe_member_path(output_dir, member)

            for member in sorted(hess_root_files, key=lambda m: m.name):
                extracted_path = _extract_member(tar, member, output_dir)
                now = time.time()
                os.utime(extracted_path, (now, now))
                extracted_files.append(str(extracted_path))

        return extracted_files

    def extract_grad_root_files(self, output_dir=None, name=None):
        """
        Extract all files matching either:
            "<name>.engrad.<WORD>.root<DIGITS>.grad.txt"
        or (if those are the true file‐names)
            "<name>.hess.root<DIGITS>"
        from the tar.gz archive.

        Parameters:
        -----------
        output_dir : str, optional
            Directory where extracted files will be saved. If None, uses the current directory.
        name : str, optional
            The base name to look for. If None, it is inferred from `self.path` by stripping ".tar.gz".

        Returns:
        --------
        list of str
            List of full paths to extracted files.
        """

        if output_dir is None:
            output_dir = self.path.parent
        os.makedirs(output_dir, exist_ok=True)

        if name is None:

            full_name = self.path.name
            if full_name.endswith(".tar.gz"):
                name = full_name[: -len(".tar.gz")]
            else:
                name = self.path.stem

        regex_str = (
            rf".*?{re.escape(name)}\.engrad\.\w+\.root\d+\.grad\.txt$"
            rf"|"
            rf".*?{re.escape(name)}\.hess\.root\d+$"
        )
        pattern = re.compile(regex_str)

        extracted_files = []
        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()

            matched_members = [m for m in members if pattern.search(m.name)]
            for member in matched_members:
                _safe_member_path(output_dir, member)

            for member in sorted(matched_members, key=lambda m: m.name):
                out_path = _extract_member(tar, member, output_dir)
                now = time.time()
                os.utime(out_path, (now, now))
                extracted_files.append(str(out_path))

        return extracted_files

    def parse(self):
        """Parse the supplied input."""
        with tarfile.open(self.path, "r:gz") as tar:
            members = tar.getmembers()
        return members
