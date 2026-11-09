"""
APEX AI Research Paper - Master Figure Generation Pipeline
Sequentially executes all 7 individual figure generation scripts,
validates output file presence, checks PDF headers, and audits PNG 300+ DPI density.
"""

import os
import sys
import time
import importlib
from pathlib import Path

# Add scripts directory and project root to Python module search path
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent.parent
FIGURES_DIR = SCRIPTS_DIR.parent / "figures"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Optional Pillow import for image metadata inspection
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


FIGURE_MODULES = [
    ("Figure 1: System Architecture Diagram", "fig1_system_architecture", "fig1_system_architecture"),
    ("Figure 2: Circuit Taxonomy & Downforce", "fig2_circuit_taxonomy", "fig2_circuit_taxonomy"),
    ("Figure 3: Micro-Sector Dynamics Telemetry", "fig3_micro_sector_dynamics", "fig3_micro_sector_dynamics"),
    ("Figure 4: Non-Linear Tire Degradation", "fig4_tire_degradation", "fig4_tire_degradation"),
    ("Figure 5: Walk-Forward Validation Scheme", "fig5_walk_forward_validation", "fig5_walk_forward_validation"),
    ("Figure 6: Model Performance & ROC-AUC", "fig6_model_performance", "fig6_model_performance"),
    ("Figure 7: Feature Importance & Ablation", "fig7_feature_ablation", "fig7_feature_ablation"),
]


def audit_png(png_path: Path):
    """Verifies PNG file exists, has valid magic bytes, >= 1200x600 px, and >= 300 DPI."""
    if not png_path.is_file():
        return False, f"Missing file: {png_path.name}"

    size_bytes = png_path.stat().st_size
    if size_bytes == 0:
        return False, f"Zero-byte file: {png_path.name}"

    with open(png_path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return False, f"Invalid PNG magic bytes: {png_path.name}"

    if HAS_PILLOW:
        try:
            with Image.open(png_path) as img:
                w, h = img.size
                dpi = img.info.get("dpi")
                dpi_x = dpi[0] if dpi else None
                dpi_y = dpi[1] if dpi else None

                if w < 1200 or h < 600:
                    return False, f"Dimension {w}x{h} < 1200x600 px"

                if dpi_x is not None and (round(dpi_x) < 300 or round(dpi_y) < 300):
                    return False, f"DPI ({dpi_x:.1f}, {dpi_y:.1f}) < 300"

                return True, f"{w}x{h} px | {dpi_x:.1f} DPI | {size_bytes / 1024:.1f} KB"
        except Exception as e:
            return False, f"Pillow read error: {e}"

    return True, f"Valid PNG ({size_bytes / 1024:.1f} KB)"


def audit_pdf(pdf_path: Path):
    """Verifies PDF file exists, has valid %PDF- header, and size >= 1024 bytes."""
    if not pdf_path.is_file():
        return False, f"Missing file: {pdf_path.name}"

    size_bytes = pdf_path.stat().st_size
    if size_bytes < 1024:
        return False, f"File too small ({size_bytes} bytes < 1024)"

    with open(pdf_path, "rb") as f:
        header = f.read(1024)
        if b"%PDF-" not in header:
            return False, "Missing %PDF- magic header"

    return True, f"Valid PDF ({size_bytes / 1024:.1f} KB)"


def run_pipeline():
    print("=" * 82)
    print("APEX AI: Reproducible IEEE Publication Figure Generation Pipeline")
    print(f"Working Directory: {ROOT_DIR}")
    print(f"Output Directory:  {FIGURES_DIR}")
    print("=" * 82)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    total_start = time.time()
    results = []
    failures = []

    for idx, (desc, mod_name, file_stem) in enumerate(FIGURE_MODULES, start=1):
        print(f"\n[{idx}/7] Executing {mod_name}.py ({desc})...")
        t0 = time.time()
        try:
            # Dynamically import or reload module
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)

            if hasattr(mod, "create_figure"):
                mod.create_figure(output_dir=str(FIGURES_DIR))
            else:
                raise AttributeError(f"Module {mod_name} does not define create_figure()")

            elapsed = time.time() - t0

            # Audit outputs
            pdf_path = FIGURES_DIR / f"{file_stem}.pdf"
            png_path = FIGURES_DIR / f"{file_stem}.png"

            pdf_ok, pdf_msg = audit_pdf(pdf_path)
            png_ok, png_msg = audit_png(png_path)

            if pdf_ok and png_ok:
                print(f"  ✓ SUCCESS in {elapsed:.2f}s")
                print(f"    - PDF: {pdf_msg}")
                print(f"    - PNG: {png_msg}")
                results.append((idx, desc, elapsed, "PASS", pdf_msg, png_msg))
            else:
                err_details = []
                if not pdf_ok:
                    err_details.append(f"PDF error: {pdf_msg}")
                if not png_ok:
                    err_details.append(f"PNG error: {png_msg}")
                print(f"  ✗ AUDIT FAILED in {elapsed:.2f}s: {'; '.join(err_details)}")
                failures.append((mod_name, "; ".join(err_details)))
                results.append((idx, desc, elapsed, "FAIL", pdf_msg, png_msg))

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ RUNTIME ERROR in {elapsed:.2f}s: {e}")
            failures.append((mod_name, str(e)))
            results.append((idx, desc, elapsed, "ERROR", str(e), str(e)))

    total_elapsed = time.time() - total_start

    # Summary Table
    print("\n" + "=" * 82)
    print("FIGURE GENERATION SUMMARY & PUBLICATION AUDIT REPORT")
    print("=" * 82)
    print(f"{'#':<3} | {'Figure Description':<38} | {'Status':<6} | {'Runtime':<8} | {'Details'}")
    print("-" * 82)
    for idx, desc, elapsed, status, pdf_msg, png_msg in results:
        details = png_msg if status == "PASS" else pdf_msg
        print(f"{idx:<3} | {desc:<38} | {status:<6} | {elapsed:>6.2f}s | {details}")
    print("=" * 82)
    print(f"Total Pipeline Runtime: {total_elapsed:.2f}s | Success: {len(results) - len(failures)}/7 | Failures: {len(failures)}")
    print("=" * 82)

    # Verification of total generated files
    all_files = sorted(os.listdir(FIGURES_DIR))
    print(f"\nFiles present in {FIGURES_DIR}:")
    for f in all_files:
        fp = FIGURES_DIR / f
        print(f"  • {f:<42} ({fp.stat().st_size:>8} bytes)")

    if failures:
        print(f"\n[FAILURE] Figure pipeline encountered {len(failures)} failures.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All 7 publication figures successfully generated and validated.")
        sys.exit(0)


if __name__ == "__main__":
    run_pipeline()
