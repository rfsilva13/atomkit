"""
LS to ICR coupling converter for AUTOSTRUCTURE calculations.

Provides functions to convert AUTOSTRUCTURE input files from LS coupling
to ICR (Intermediate Coupling Representation) coupling, using optimized
lambda scaling parameters from variational calculations.

Compatible with modern Python 3.13+ and integrates with atomkit readers.

Original Author: Tomás Campante (October 2025)
Adapted by: Ricardo Silva (rfsilva@lip.pt)
Date: October 2025
"""

from pathlib import Path
from typing import Optional, List, Tuple
import re
import shutil
import subprocess
import logging

import numpy as np

from ..readers.autostructure import read_as_lambdas

logger = logging.getLogger(__name__)


def convert_ls_to_icr(
    ls_input_file: str | Path,
    icr_output_file: Optional[str | Path] = None,
    olg_file: str | Path = "olg",
    as_executable: Optional[str | Path] = None,
    run_ls_calculation: bool = True,
    lambdas: Optional[np.ndarray] = None,
) -> Tuple[Path, np.ndarray]:
    """
    Convert AUTOSTRUCTURE LS coupling input to ICR coupling.

    This function performs the following steps:
    1. Optionally runs the LS calculation using AUTOSTRUCTURE
    2. Extracts optimized lambda scaling parameters from the output
    3. Creates an ICR input file with the optimized lambdas

    Parameters
    ----------
    ls_input_file : str or Path
        Path to the LS coupling AUTOSTRUCTURE input file
    icr_output_file : str or Path, optional
        Path for the ICR output file. If None, creates one based on ls_input_file
    olg_file : str or Path, optional
        Path to the AUTOSTRUCTURE output file containing lambdas (default: 'olg')
    as_executable : str or Path, optional
        Path to AUTOSTRUCTURE executable. If None, uses './autostructure.x'
    run_ls_calculation : bool, optional
        Whether to run the LS calculation first (default: True)
    lambdas : array-like, optional
        Pre-computed lambda values to use instead of extracting from olg file

    Returns
    -------
    tuple of (Path, np.ndarray)
        - Path to the created ICR input file
        - Array of lambda scaling parameters used

    Raises
    ------
    FileNotFoundError
        If input files don't exist
    RuntimeError
        If AUTOSTRUCTURE calculation fails
    ValueError
        If no lambda parameters can be extracted

    Examples
    --------
    >>> # Run LS calculation and create ICR file
    >>> icr_file, lambdas = convert_ls_to_icr('Fe_LS.inp', 'Fe_ICR.inp')
    >>> print(f"Created {icr_file} with {len(lambdas)} lambda parameters")

    >>> # Use pre-computed lambdas without running LS calculation
    >>> icr_file, _ = convert_ls_to_icr(
    ...     'Fe_LS.inp',
    ...     run_ls_calculation=False,
    ...     lambdas=np.array([1.0, 0.95, 0.98])
    ... )

    Notes
    -----
    Preserves original logic from Tomás Campante's LSintoICR.py script.
    The function modifies the LS input to work with ICR coupling by:
    - Changing CUP="LS" to CUP="ICR"
    - Removing variational flags (INCLUD, NVAR)
    - Adding optimized lambda values
    """
    ls_path = Path(ls_input_file)
    olg_path = Path(olg_file)

    if not ls_path.exists():
        raise FileNotFoundError(f"LS input file not found: {ls_path}")

    # Determine ICR output file path
    if icr_output_file is None:
        icr_path = ls_path.with_stem(f"{ls_path.stem}_ICR")
    else:
        icr_path = Path(icr_output_file)

    logger.info(f"Converting {ls_path} to ICR coupling")

    # Run LS calculation if requested
    if run_ls_calculation:
        if lambdas is not None:
            logger.warning(
                "Both run_ls_calculation=True and lambdas provided. "
                "Running LS calculation and ignoring provided lambdas."
            )

        lambdas_result = run_autostructure_ls(ls_path, olg_path, as_executable)
    else:
        if lambdas is None:
            # Try to read from existing olg file
            logger.info(f"Reading lambdas from existing {olg_path}")
            if not olg_path.exists():
                raise FileNotFoundError(
                    f"olg file not found: {olg_path}. "
                    "Run LS calculation first or provide lambdas parameter."
                )
            _, lambdas_result = read_as_lambdas(olg_path)
        else:
            lambdas_result = np.asarray(lambdas, dtype=np.float64)

    if len(lambdas_result) == 0:
        raise ValueError("No lambda parameters available for ICR conversion")

    # Create ICR input file
    create_icr_input(ls_path, icr_path, lambdas_result)

    logger.info(f"Successfully created ICR input file: {icr_path}")
    logger.info(f"Used {len(lambdas_result)} lambda parameters")

    return icr_path, lambdas_result


def run_autostructure_ls(
    ls_input_file: str | Path,
    olg_file: str | Path = "olg",
    as_executable: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Run AUTOSTRUCTURE calculation in LS coupling and extract lambdas.

    Parameters
    ----------
    ls_input_file : str or Path
        Path to LS coupling input file
    olg_file : str or Path, optional
        Path where AUTOSTRUCTURE writes output (default: 'olg')
    as_executable : str or Path, optional
        Path to AUTOSTRUCTURE executable (default: './autostructure.x')

    Returns
    -------
    np.ndarray
        Array of extracted lambda scaling parameters

    Raises
    ------
    FileNotFoundError
        If executable or input file not found
    RuntimeError
        If calculation fails
    ValueError
        If no lambdas can be extracted from output
    """
    ls_path = Path(ls_input_file)

    if not ls_path.exists():
        raise FileNotFoundError(f"LS input file not found: {ls_path}")

    # Determine executable path
    if as_executable is None:
        exec_path = Path("./autostructure.x")
    else:
        exec_path = Path(as_executable)

    if not exec_path.exists():
        raise FileNotFoundError(
            f"AUTOSTRUCTURE executable not found: {exec_path}\n"
            f"Provide the path via as_executable parameter."
        )

    logger.info(f"Running AUTOSTRUCTURE LS calculation: {ls_path}")

    try:
        # Run AUTOSTRUCTURE with input redirection
        result = subprocess.run(
            f"{exec_path} < {ls_path}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"AUTOSTRUCTURE failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(
                f"AUTOSTRUCTURE calculation failed (exit code {result.returncode})"
            )

        logger.info("LS calculation completed successfully")

    except subprocess.TimeoutExpired:
        raise RuntimeError("AUTOSTRUCTURE calculation timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Error running AUTOSTRUCTURE: {e}")

    # Extract lambdas from output
    olg_path = Path(olg_file)
    if not olg_path.exists():
        raise RuntimeError(f"AUTOSTRUCTURE output file not created: {olg_path}")

    nl_array, lambdas = read_as_lambdas(olg_path)

    if len(lambdas) == 0:
        raise ValueError(
            "No lambda parameters found in AUTOSTRUCTURE output. "
            "Ensure the LS input uses variational method for lambda optimization."
        )

    logger.info(f"Extracted {len(lambdas)} lambda parameters")
    return lambdas


def create_icr_input(
    ls_input_file: str | Path, icr_output_file: str | Path, lambdas: np.ndarray
) -> None:
    """
    Create ICR coupling input file from LS input with optimized lambdas.

    Modifies the LS input file to work with ICR coupling by:
    - Changing CUP="LS" to CUP="ICR"
    - Removing variational optimization flags (INCLUD, NVAR)
    - Adding the lambda scaling parameters

    Parameters
    ----------
    ls_input_file : str or Path
        Path to LS coupling input file
    icr_output_file : str or Path
        Path for ICR output file
    lambdas : array-like
        Lambda scaling parameters to insert

    Raises
    ------
    FileNotFoundError
        If LS input file doesn't exist
    ValueError
        If lambdas array is empty or invalid
    """
    ls_path = Path(ls_input_file)
    icr_path = Path(icr_output_file)

    if not ls_path.exists():
        raise FileNotFoundError(f"LS input file not found: {ls_path}")

    lambdas_array = np.asarray(lambdas, dtype=np.float64)
    if len(lambdas_array) == 0:
        raise ValueError("Empty lambda array provided")

    logger.info(f"Creating ICR input: {icr_path}")

    # Copy LS file to ICR file
    shutil.copy(ls_path, icr_path)

    # Read ICR file
    with open(icr_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Modify lines
    modified_lines = []
    sminim_found = False

    for i, line in enumerate(lines):
        modified_line = line

        # Replace CUP="LS" with CUP="ICR"
        if re.search(r'CUP\s*=\s*[\'"]LS[\'"]', line, re.IGNORECASE):
            modified_line = re.sub(
                r'CUP\s*=\s*[\'"]LS[\'"]', 'CUP="ICR"', line, flags=re.IGNORECASE
            )
            logger.debug("Replaced CUP='LS' with CUP='ICR'")

        # Find &SMINIM section and modify it
        if line.strip().startswith("&SMINIM") and not sminim_found:
            # Remove variational flags
            modified_line = re.sub(r"\bINCLUD\s*=\s*[^ \t&]+", "", modified_line)
            modified_line = re.sub(r"\bNVAR\s*=\s*[^ \t&]+", "", modified_line)
            modified_line = re.sub(r"\s+&END", " &END", modified_line)

            modified_lines.append(modified_line)

            # Add lambda values on the next line
            formatted_lambdas = "  ".join(f"{x:.10f}" for x in lambdas_array)
            modified_lines.append(f" {formatted_lambdas}\n")

            sminim_found = True
            logger.debug(
                f"Modified &SMINIM section and added {len(lambdas_array)} lambdas"
            )

            # Skip the original next line (it will be replaced by lambdas)
            if i + 1 < len(lines):
                continue
        else:
            modified_lines.append(modified_line)

    if not sminim_found:
        logger.warning("&SMINIM section not found in input file")

    # Write modified ICR file
    with open(icr_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)

    logger.info(f"Successfully wrote ICR input to {icr_path}")


def run_autostructure_icr(
    icr_input_file: str | Path, as_executable: Optional[str | Path] = None
) -> subprocess.CompletedProcess:
    """
    Run AUTOSTRUCTURE calculation with ICR coupling input.

    Parameters
    ----------
    icr_input_file : str or Path
        Path to ICR coupling input file
    as_executable : str or Path, optional
        Path to AUTOSTRUCTURE executable (default: './autostructure.x')

    Returns
    -------
    subprocess.CompletedProcess
        Result of the AUTOSTRUCTURE run

    Raises
    ------
    FileNotFoundError
        If executable or input file not found
    RuntimeError
        If calculation fails
    """
    icr_path = Path(icr_input_file)

    if not icr_path.exists():
        raise FileNotFoundError(f"ICR input file not found: {icr_path}")

    # Determine executable path
    if as_executable is None:
        exec_path = Path("./autostructure.x")
    else:
        exec_path = Path(as_executable)

    if not exec_path.exists():
        raise FileNotFoundError(
            f"AUTOSTRUCTURE executable not found: {exec_path}\n"
            f"Provide the path via as_executable parameter."
        )

    logger.info(f"Running AUTOSTRUCTURE ICR calculation: {icr_path}")

    try:
        result = subprocess.run(
            f"{exec_path} < {icr_path}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"AUTOSTRUCTURE failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(
                f"AUTOSTRUCTURE calculation failed (exit code {result.returncode})"
            )

        logger.info("ICR calculation completed successfully")
        return result

    except subprocess.TimeoutExpired:
        raise RuntimeError("AUTOSTRUCTURE calculation timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Error running AUTOSTRUCTURE: {e}")


def ls_to_icr_full_workflow(
    ls_input_file: str | Path,
    icr_output_file: Optional[str | Path] = None,
    as_executable: Optional[str | Path] = None,
    run_icr_calculation: bool = False,
) -> dict:
    """
    Complete workflow: LS calculation → lambda extraction → ICR file creation → optional ICR run.

    This is a convenience function that performs the complete workflow:
    1. Run LS coupling calculation
    2. Extract optimized lambda parameters
    3. Create ICR coupling input file
    4. Optionally run ICR calculation

    Parameters
    ----------
    ls_input_file : str or Path
        Path to LS coupling input file
    icr_output_file : str or Path, optional
        Path for ICR output file (auto-generated if None)
    as_executable : str or Path, optional
        Path to AUTOSTRUCTURE executable
    run_icr_calculation : bool, optional
        Whether to run the ICR calculation (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'ls_input': Path to LS input file
        - 'icr_output': Path to created ICR input file
        - 'lambdas': Array of lambda parameters
        - 'icr_result': subprocess result if run_icr_calculation=True, else None

    Examples
    --------
    >>> result = ls_to_icr_full_workflow('Fe_LS.inp', run_icr_calculation=True)
    >>> print(f"ICR file: {result['icr_output']}")
    >>> print(f"Lambdas: {result['lambdas']}")
    """
    logger.info("Starting LS→ICR full workflow")

    # Convert LS to ICR
    icr_path, lambdas = convert_ls_to_icr(
        ls_input_file,
        icr_output_file,
        as_executable=as_executable,
        run_ls_calculation=True,
    )

    result = {
        "ls_input": Path(ls_input_file),
        "icr_output": icr_path,
        "lambdas": lambdas,
        "icr_result": None,
    }

    # Optionally run ICR calculation
    if run_icr_calculation:
        logger.info("Running ICR calculation")
        icr_result = run_autostructure_icr(icr_path, as_executable)
        result["icr_result"] = icr_result

    logger.info("LS→ICR workflow completed successfully")
    return result
