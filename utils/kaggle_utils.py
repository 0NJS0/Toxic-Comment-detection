"""Kaggle dataset snapshot helpers for cross-session resume.

``/kaggle/working`` is wiped when a Kaggle session ends. To resume heavy work
(teacher pass, federated runs) across sessions we snapshot the output
directories as a Kaggle Dataset and restore them at the start of the next
session. Everything is best-effort: if the ``kaggle`` CLI or credentials are
missing, the same tarball + commands are printed for manual upload, and calls
never raise (guarded by ``sessions.snapshot``).
"""

import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import List, Optional


def restore_from_dataset(output_dir: str, slug: Optional[str] = None) -> bool:
    """Extract a snapshot tarball (from a previously pulled Kaggle dataset)
    into ``output_dir``. Returns True if anything was restored."""
    out = Path(output_dir)
    if slug:
        dataset_dir = Path("/kaggle/input") / slug.replace("/", "/")
        matches = sorted(dataset_dir.glob("**/*.tgz"))
        if not matches:
            print(f"  [hint] Kaggle dataset '{slug}' not mounted under "
                  f"/kaggle/input - is it attached to this notebook?")
        else:
            for tgz in matches:
                print(f"  Restoring {tgz} -> {out}")
                with tarfile.open(tgz, "r:gz") as tar:
                    tar.extractall(out)
            return True
    # Local fallback (developer machine / same-session resume).
    tarball = out.parent / "snapshot.latest.tgz"
    if tarball.exists():
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(out)
        print(f"  Restored local {tarball}")
        return True
    return False


def snapshot_dataset(
    output_dirs: List[str],
    slug: str = "toxic-early-detection-llm",
    message: str = "session snapshot",
    base_dir: str = ".",
) -> Optional[str]:
    """Tar the given output dirs and, if possible, publish as a Kaggle
    dataset. Returns the tarball path (never raises on failure)."""
    base = Path(base_dir)
    tarball = base / "snapshot.latest.tgz"

    with tarfile.open(tarball, "w:gz") as tar:
        for d in output_dirs:
            p = Path(d)
            if p.exists() and any(iter(p)):
                tar.add(str(p), arcname=p.name)
    size_mb = tarball.stat().st_size / 1e6
    print(f"\n  Snapshot tarball: {tarball} ({size_mb:.1f} MB)")

    if shutil.which("kaggle") is None:
        print("  [hint] `kaggle` CLI not found - to upload manually:\n"
              f"    kaggle datasets init -p {base}\n"
              f"    kaggle datasets version -p {base} -m \"{message}\"")
        return str(tarball)

    try:
        subprocess.run(["kaggle", "datasets", "version", "-p", str(base),
                        "-m", message], check=True,
                       capture_output=True, text=True)
        print(f"  [ok] Published Kaggle dataset: {slug}")
    except (subprocess.CalledProcessError, OSError) as err:
        print(f"  [hint] Dataset upload failed: {err}\n"
              "  Attach the dataset and re-run to resume.")
    return str(tarball)