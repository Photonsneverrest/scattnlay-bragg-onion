from __future__ import annotations

from pathlib import Path, PureWindowsPath
import importlib.util
import platform
import pandas as pd


def running_in_wsl() -> bool:
    """
    Return True if running inside WSL/Linux.
    """
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            txt = f.read().lower()
        return ("microsoft" in txt) or ("wsl" in txt)
    except Exception:
        return False


def windows_to_wsl_path(win_path: str | Path) -> Path:
    r"""
    Convert a Windows path like:
        C:\Users\Name\folder
    to a WSL path like:
        /mnt/c/Users/Name/folder
    """
    p = PureWindowsPath(str(win_path))
    drive = p.drive.rstrip(":").lower()
    parts = p.parts[1:]  # skip drive part
    return Path("/mnt") / drive / Path(*parts)


def detect_parquet_support() -> tuple[bool, bool, bool]:
    """
    Returns:
        has_pyarrow, has_fastparquet, has_parquet_support
    """
    has_pyarrow = importlib.util.find_spec("pyarrow") is not None
    has_fastparquet = importlib.util.find_spec("fastparquet") is not None
    has_parquet_support = has_pyarrow or has_fastparquet
    return has_pyarrow, has_fastparquet, has_parquet_support


def _resolve_source_output_dir(source_output_dir: str | Path) -> Path:
    """
    Resolve a source output directory in a WSL-aware way.

    If running in WSL and the incoming path is a Windows-style path, it is converted
    to /mnt/<drive>/...
    """
    source_output_dir = str(source_output_dir)

    if running_in_wsl():
        # crude but effective detection of Windows-style path:
        # e.g. "C:\..."
        if len(source_output_dir) >= 2 and source_output_dir[1] == ":":
            return windows_to_wsl_path(source_output_dir)

    return Path(source_output_dir)


def load_sweep_outputs(
    source_output_dir: str | Path,
    *,
    combine_if_needed: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path, bool]:
    """
    Load sweep outputs from a directory.

    Load priority:
      1) df_all.parquet
      2) df_all.pkl
      3) combine chunk_*.parquet
      4) combine chunk_*.pkl

    Returns
    -------
    df_all : pd.DataFrame
    output_dir : pathlib.Path
    use_parquet : bool
    """
    output_dir = _resolve_source_output_dir(source_output_dir)
    checkpoint_dir = output_dir / "primary_sweep_checkpoints"

    has_pyarrow, has_fastparquet, has_parquet_support = detect_parquet_support()

    combined_parquet = output_dir / "df_all.parquet"
    combined_pickle = output_dir / "df_all.pkl"

    parquet_parts = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    pickle_parts = sorted(checkpoint_dir.glob("chunk_*.pkl"))

    if verbose:
        print("SOURCE_OUTPUT_DIR:", output_dir)
        print("CHECKPOINT_DIR    :", checkpoint_dir)
        print("SOURCE_OUTPUT_DIR exists:", output_dir.exists())
        print("CHECKPOINT_DIR exists    :", checkpoint_dir.exists())
        print("Parquet support   :", has_parquet_support)
        print("combined_parquet exists:", combined_parquet.exists())
        print("combined_pickle  exists:", combined_pickle.exists())
        print("number of parquet chunks:", len(parquet_parts))
        print("number of pickle  chunks:", len(pickle_parts))

        if output_dir.exists():
            print("\nTop-level contents of SOURCE_OUTPUT_DIR:")
            for p in sorted(output_dir.iterdir())[:30]:
                print("  ", p.name)

        if checkpoint_dir.exists():
            print("\nFirst contents of CHECKPOINT_DIR:")
            for p in sorted(checkpoint_dir.iterdir())[:30]:
                print("  ", p.name)

    if combined_parquet.exists():
        if not has_parquet_support:
            raise ImportError(
                f"Found {combined_parquet}, but pyarrow/fastparquet is not installed."
            )
        df_all = pd.read_parquet(combined_parquet)
        return df_all, output_dir, True

    if combined_pickle.exists():
        df_all = pd.read_pickle(combined_pickle)
        return df_all, output_dir, False

    if parquet_parts:
        if not has_parquet_support:
            raise ImportError(
                f"Found parquet chunk files in {checkpoint_dir}, but pyarrow/fastparquet is not installed."
            )
        df_all = pd.concat([pd.read_parquet(p) for p in parquet_parts], ignore_index=True)
        if combine_if_needed:
            df_all.to_parquet(combined_parquet, index=False)
            if verbose:
                print(f"\nCombined dataframe saved to: {combined_parquet}")
        return df_all, output_dir, True

    if pickle_parts:
        df_all = pd.concat([pd.read_pickle(p) for p in pickle_parts], ignore_index=True)
        if combine_if_needed:
            df_all.to_pickle(combined_pickle)
            if verbose:
                print(f"\nCombined dataframe saved to: {combined_pickle}")
        return df_all, output_dir, False

    raise FileNotFoundError(
        f"No combined dataframe and no checkpoint chunk files were found.\n"
        f"Checked:\n"
        f"  - {combined_parquet}\n"
        f"  - {combined_pickle}\n"
        f"  - {checkpoint_dir / 'chunk_*.parquet'}\n"
        f"  - {checkpoint_dir / 'chunk_*.pkl'}"
    )