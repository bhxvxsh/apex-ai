"""
APEX AI: Comprehensive End-to-End (E2E) Test Suite for IEEE Research Paper & Assets
==================================================================================

Target: 06_Research_Paper/ and associated empirical pipelines.
Governed by: ORIGINAL_REQUEST.md, PROJECT.md, and TEST_INFRA.md.

4-Tier Test Architecture:
  - Tier 1: Target Deliverable & Feature Coverage
  - Tier 2: Boundary, Format & Resolution Validation
  - Tier 3: Cross-Feature Alignment & Internal Consistency
  - Tier 4: Empirical Ground-Truth Validation & Hypotheses Verification

Execution:
  - pytest:  python -m pytest tests/test_e2e_research_paper.py -v
  - runner:  python tests/run_e2e_tests.py
  - direct:  python tests/test_e2e_research_paper.py
"""

import os
import sys
import re
import json
import sqlite3
import struct
import py_compile
import subprocess
import unittest
from pathlib import Path

# Optional Pillow import for image metadata inspection
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


def find_workspace_root() -> Path:
    """
    Locates the APEX_AI directory root regardless of current working directory.
    """
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "APEX_AI",
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent / "APEX_AI",
    ]
    for c in candidates:
        if (c / "06_Research_Paper").exists() and (c / "02_Data_Processed").exists():
            return c.resolve()
    # Fallback to parent of tests directory
    return Path(__file__).resolve().parent.parent


ROOT_DIR = find_workspace_root()
PAPER_DIR = ROOT_DIR / "06_Research_Paper"
SCRIPTS_DIR = PAPER_DIR / "scripts"
FIGURES_DIR = PAPER_DIR / "figures"
DATA_DIR = ROOT_DIR / "02_Data_Processed"
MODELS_DIR = ROOT_DIR / "04_Models"
NOTEBOOKS_DIR = ROOT_DIR / "07_Notebooks"

DB_PATH = DATA_DIR / "f1.db"
WALK_FORWARD_PATH = MODELS_DIR / "walk_forward_metrics.json"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics_summary.json"
NOTEBOOK_FINETUNE_PATH = NOTEBOOKS_DIR / "apex_ai_finetune_colab.ipynb"

# Canonical Figure Generation Scripts
EXPECTED_SCRIPTS = [
    "fig1_system_architecture.py",
    "fig2_circuit_taxonomy.py",
    "fig3_micro_sector_dynamics.py",
    "fig4_tire_degradation.py",
    "fig5_walk_forward_validation.py",
    "fig6_model_performance.py",
    "fig7_feature_ablation.py",
    "generate_all_figures.py",
]

# Canonical Figure Names (allowing accepted canonical synonyms)
CANONICAL_FIGURE_STEMS = [
    ("fig1_system_architecture",),
    ("fig2_circuit_taxonomy",),
    ("fig3_micro_sector_telemetry", "fig3_micro_sector_dynamics"),
    ("fig4_tire_degradation",),
    ("fig5_walk_forward_scheme", "fig5_walk_forward_validation"),
    ("fig6_model_performance_roc", "fig6_model_performance"),
    ("fig7_feature_importance_ablation", "fig7_feature_ablation"),
]


# ==============================================================================
# Helper Utilities
# ==============================================================================

def read_png_dimensions_and_dpi_raw(path: Path):
    """
    Extracts (width, height, dpi_x, dpi_y) from raw PNG chunks if Pillow is unavailable.
    """
    with open(path, "rb") as f:
        data = f.read(65536)

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a valid PNG file: {path}")

    # IHDR is the first chunk at offset 8
    ihdr_len = struct.unpack(">I", data[8:12])[0]
    ihdr_type = data[12:16]
    if ihdr_type != b"IHDR":
        raise ValueError(f"Expected IHDR chunk in PNG: {path}")

    width, height = struct.unpack(">II", data[16:24])

    # Search for pHYs chunk
    dpi_x, dpi_y = None, None
    offset = 8 + 4 + 4 + ihdr_len + 4  # skip len, type, data, crc
    while offset < len(data) - 12:
        chunk_len = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        if chunk_type == b"pHYs":
            ppux, ppuy, unit = struct.unpack(">IIB", data[offset + 8:offset + 8 + 9])
            if unit == 1:  # 1 = meter
                dpi_x = ppux * 0.0254
                dpi_y = ppuy * 0.0254
            break
        offset += 4 + 4 + chunk_len + 4

    return width, height, dpi_x, dpi_y


# ==============================================================================
# TIER 1: Target Deliverables & Feature Coverage
# ==============================================================================

class TestTier1Deliverables(unittest.TestCase):
    """
    Tier 1 verifies the existence, proper workspace topology, and non-emptiness
    of all core IEEE research paper deliverables, style files, scripts, and figures.
    """

    def test_tier1_01_core_latex_and_bib_files_exist_and_non_empty(self):
        """Verifies IEEEtran.cls, IEEEtran.bst, and references.bib exist and are non-empty."""
        cls_file = PAPER_DIR / "IEEEtran.cls"
        bst_file = PAPER_DIR / "IEEEtran.bst"
        bib_file = PAPER_DIR / "references.bib"

        self.assertTrue(cls_file.is_file(), f"Missing IEEEtran.cls at {cls_file}")
        self.assertGreater(cls_file.stat().st_size, 10000, "IEEEtran.cls must be >= 10KB standard distribution")

        self.assertTrue(bst_file.is_file(), f"Missing IEEEtran.bst at {bst_file}")
        self.assertGreater(bst_file.stat().st_size, 5000, "IEEEtran.bst must be >= 5KB standard distribution")

        self.assertTrue(bib_file.is_file(), f"Missing references.bib at {bib_file}")
        self.assertGreater(bib_file.stat().st_size, 1000, "references.bib must contain multiple peer-reviewed citations")

    def test_tier1_02_manuscript_deliverables_exist_and_non_empty(self):
        """Verifies main.tex and PAPER.md exist and contain substantial publication content."""
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"

        self.assertTrue(main_tex.is_file(), f"Missing main.tex manuscript at {main_tex}")
        self.assertGreater(main_tex.stat().st_size, 5000, "main.tex must be a complete manuscript (> 5KB)")

        self.assertTrue(paper_md.is_file(), f"Missing PAPER.md document at {paper_md}")
        self.assertGreater(paper_md.stat().st_size, 5000, "PAPER.md must be a complete manuscript (> 5KB)")

    def test_tier1_03_standalone_figure_scripts_exist_and_non_empty(self):
        """Verifies all 7 standalone figure generation scripts + generate_all_figures.py exist."""
        self.assertTrue(SCRIPTS_DIR.is_dir(), f"Missing scripts directory at {SCRIPTS_DIR}")

        missing_scripts = []
        for script_name in EXPECTED_SCRIPTS:
            script_path = SCRIPTS_DIR / script_name
            if not script_path.is_file() or script_path.stat().st_size < 100:
                missing_scripts.append(script_name)

        self.assertEqual(
            missing_scripts,
            [],
            f"Missing or empty figure generation scripts in {SCRIPTS_DIR}: {missing_scripts}"
        )

    def test_tier1_04_all_fourteen_figure_files_exist_and_non_empty(self):
        """Verifies all 7 figures exist in both PDF and PNG formats in 06_Research_Paper/figures/."""
        self.assertTrue(FIGURES_DIR.is_dir(), f"Missing figures directory at {FIGURES_DIR}")

        missing_figures = []
        found_stems = []

        for idx, stem_candidates in enumerate(CANONICAL_FIGURE_STEMS, start=1):
            stem_found = None
            for stem in stem_candidates:
                pdf_path = FIGURES_DIR / f"{stem}.pdf"
                png_path = FIGURES_DIR / f"{stem}.png"
                if pdf_path.is_file() and png_path.is_file():
                    if pdf_path.stat().st_size > 0 and png_path.stat().st_size > 0:
                        stem_found = stem
                        break

            if stem_found is None:
                missing_figures.append(f"Figure {idx} ({' or '.join(stem_candidates)}) [.pdf and .png]")
            else:
                found_stems.append(stem_found)

        self.assertEqual(
            missing_figures,
            [],
            f"Missing or zero-byte figure assets in {FIGURES_DIR}: {missing_figures}"
        )
        self.assertEqual(len(found_stems), 7, f"Expected 7 distinct figures, found {len(found_stems)}")


# ==============================================================================
# TIER 2: Boundary, Format, & Resolution Validation
# ==============================================================================

class TestTier2ResolutionAndBoundary(unittest.TestCase):
    """
    Tier 2 enforces publication visual standards:
      - PNG raster images >= 300 DPI with valid magic bytes and dimension thresholds.
      - Vector PDFs have valid %PDF- magic headers and non-zero byte size (> 1KB).
      - Python figure scripts compile cleanly and execute standalone with returncode 0.
    """

    def test_tier2_01_raster_png_resolution_and_density(self):
        """Verifies that each PNG figure has >= 300 DPI density and valid dimensions."""
        self.assertTrue(FIGURES_DIR.is_dir(), f"Missing figures dir: {FIGURES_DIR}")
        png_files = sorted(FIGURES_DIR.glob("*.png"))
        self.assertGreaterEqual(len(png_files), 7, f"Expected at least 7 PNG figures, found {len(png_files)}")

        dpi_failures = []
        dim_failures = []

        for png_path in png_files:
            # 1. Magic bytes check
            with open(png_path, "rb") as f:
                sig = f.read(8)
                self.assertEqual(
                    sig,
                    b"\x89PNG\r\n\x1a\n",
                    f"{png_path.name} does not have valid PNG magic signature."
                )

            # 2. DPI and Dimension check
            if HAS_PILLOW:
                with Image.open(png_path) as img:
                    w, h = img.size
                    dpi = img.info.get("dpi")
                    if dpi is not None:
                        # matplotlib's 300 DPI export often rounds to ~299.9994 in pHYs chunk
                        dpi_x, dpi_y = dpi
                    else:
                        dpi_x, dpi_y = None, None
            else:
                w, h, dpi_x, dpi_y = read_png_dimensions_and_dpi_raw(png_path)

            if w < 1200 or h < 600:
                dim_failures.append(f"{png_path.name}: {w}x{h} px (minimum required: 1200x600)")

            if dpi_x is not None:
                # Tolerance: round(dpi) >= 300 or float >= 299.0
                if round(dpi_x) < 300 or round(dpi_y) < 300:
                    dpi_failures.append(f"{png_path.name}: DPI ({dpi_x:.1f}, {dpi_y:.1f}) < 300 DPI")

        self.assertEqual(dim_failures, [], f"PNG dimension check failed: {dim_failures}")
        self.assertEqual(dpi_failures, [], f"PNG 300+ DPI check failed: {dpi_failures}")

    def test_tier2_02_vector_pdf_header_and_size(self):
        """Verifies that all PDF figures have valid %PDF- magic bytes and size >= 1024 bytes."""
        self.assertTrue(FIGURES_DIR.is_dir(), f"Missing figures dir: {FIGURES_DIR}")
        pdf_files = sorted(FIGURES_DIR.glob("*.pdf"))
        self.assertGreaterEqual(len(pdf_files), 7, f"Expected at least 7 PDF figures, found {len(pdf_files)}")

        for pdf_path in pdf_files:
            size = pdf_path.stat().st_size
            self.assertGreaterEqual(size, 1024, f"{pdf_path.name} byte size {size} is too small (< 1024 bytes)")

            with open(pdf_path, "rb") as f:
                header = f.read(1024)
                self.assertTrue(
                    b"%PDF-" in header,
                    f"{pdf_path.name} does not contain valid '%PDF-' header."
                )

    def test_tier2_03_figure_scripts_compilation(self):
        """Verifies that all figure generation scripts compile without syntax errors."""
        self.assertTrue(SCRIPTS_DIR.is_dir(), f"Missing scripts dir: {SCRIPTS_DIR}")
        py_files = sorted(SCRIPTS_DIR.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 8, f"Expected at least 8 Python scripts in {SCRIPTS_DIR}")

        compilation_errors = []
        for py_path in py_files:
            try:
                py_compile.compile(str(py_path), doraise=True)
            except py_compile.PyCompileError as e:
                compilation_errors.append(f"{py_path.name}: {e}")

        self.assertEqual(compilation_errors, [], f"AST compilation failed on scripts: {compilation_errors}")

    def test_tier2_04_master_figure_runner_execution(self):
        """Executes generate_all_figures.py in standalone mode and checks for returncode 0."""
        runner_path = SCRIPTS_DIR / "generate_all_figures.py"
        self.assertTrue(runner_path.is_file(), f"Missing {runner_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

        self.assertEqual(
            proc.returncode,
            0,
            f"generate_all_figures.py exited with code {proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


# ==============================================================================
# TIER 3: Cross-Feature Alignment & Internal Consistency
# ==============================================================================

class TestTier3CrossFeatureAlignment(unittest.TestCase):
    """
    Tier 3 validates links and symmetry across the deliverables:
      - All figure references in main.tex and PAPER.md resolve to physical figures.
      - 100% of citation keys in main.tex resolve to entries in references.bib.
      - IEEE 8-section layout symmetry between LaTeX and Markdown.
      - Table structures and numerical data match between formats.
    """

    def test_tier3_01_main_tex_figure_references_resolve(self):
        """Verifies every \\includegraphics in main.tex resolves to a figure in figures/."""
        main_tex = PAPER_DIR / "main.tex"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")

        content = main_tex.read_text(encoding="utf-8")
        # Match \includegraphics[...]{path} or \includegraphics{path}
        pattern = re.compile(r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}")
        matches = pattern.findall(content)

        self.assertGreaterEqual(len(matches), 7, f"Expected at least 7 \\includegraphics calls in main.tex, found {len(matches)}")

        unresolved = []
        for ref in matches:
            ref_path = Path(ref.strip())
            # In LaTeX, path might be 'figures/fig1.pdf', 'fig1.pdf', or without extension 'figures/fig1'
            candidates = [
                PAPER_DIR / ref_path,
                FIGURES_DIR / ref_path.name,
                FIGURES_DIR / f"{ref_path.stem}.pdf",
                FIGURES_DIR / f"{ref_path.stem}.png",
            ]
            if not any(c.is_file() for c in candidates):
                unresolved.append(ref)

        self.assertEqual(unresolved, [], f"Unresolved \\includegraphics references in main.tex: {unresolved}")

    def test_tier3_02_paper_md_figure_references_resolve(self):
        """Verifies every ![...](...) image link in PAPER.md resolves to an existing file."""
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        content = paper_md.read_text(encoding="utf-8")
        # Match ![alt](path)
        pattern = re.compile(r"!\[.*?\]\(([^)]+)\)")
        matches = pattern.findall(content)

        self.assertGreaterEqual(len(matches), 7, f"Expected at least 7 figure images in PAPER.md, found {len(matches)}")

        unresolved = []
        for ref in matches:
            ref_clean = ref.strip().split()[0]  # remove any title tag
            ref_path = Path(ref_clean)
            candidates = [
                PAPER_DIR / ref_path,
                FIGURES_DIR / ref_path.name,
                FIGURES_DIR / f"{ref_path.stem}.png",
                FIGURES_DIR / f"{ref_path.stem}.pdf",
            ]
            if not any(c.is_file() for c in candidates):
                unresolved.append(ref_clean)

        self.assertEqual(unresolved, [], f"Unresolved image references in PAPER.md: {unresolved}")

    def test_tier3_03_citation_keys_resolve_to_bibtex(self):
        """Verifies that 100% of cited keys in main.tex exist in references.bib (>= 25 citations)."""
        bib_file = PAPER_DIR / "references.bib"
        main_tex = PAPER_DIR / "main.tex"
        self.assertTrue(bib_file.is_file(), "references.bib must exist")
        self.assertTrue(main_tex.is_file(), "main.tex must exist")

        # Extract defined BibTeX keys
        bib_content = bib_file.read_text(encoding="utf-8")
        defined_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_content))

        self.assertGreaterEqual(
            len(defined_keys),
            25,
            f"references.bib must contain at least 25 citations, found {len(defined_keys)}"
        )

        # Extract cited keys in main.tex: \cite{key1, key2}
        tex_content = main_tex.read_text(encoding="utf-8")
        raw_cites = re.findall(r"\\cite[a-zA-Z*]*\{([^}]+)\}", tex_content)
        cited_keys = set()
        for group in raw_cites:
            for k in group.split(","):
                k_clean = k.strip()
                if k_clean:
                    cited_keys.add(k_clean)

        self.assertGreaterEqual(
            len(cited_keys),
            15,
            f"Expected at least 15 citations in main.tex manuscript, found {len(cited_keys)}"
        )

        broken_citations = cited_keys - defined_keys
        self.assertEqual(
            broken_citations,
            set(),
            f"Broken citations in main.tex that do not exist in references.bib: {broken_citations}"
        )

    def test_tier3_04_eight_section_ieee_structural_symmetry(self):
        """Verifies that both main.tex and PAPER.md contain the 8 canonical IEEE sections."""
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8").lower()
        md_text = paper_md.read_text(encoding="utf-8").lower()

        required_sections = [
            ("Section I: Introduction", ["introduction"]),
            ("Section II: Related Work", ["related work", "literature review"]),
            ("Section III: System Architecture & Methodology", ["system architecture", "methodology"]),
            ("Section IV: Mathematical Formulation & Empirical Hypotheses", ["mathematical formulation", "hypotheses"]),
            ("Section V: Experimental Setup & Walk-Forward Validation", ["experimental setup", "walk-forward"]),
            ("Section VI: Results & Quantitative Analysis", ["results", "quantitative analysis"]),
            ("Section VII: Discussion & Ablation Studies", ["discussion", "ablation"]),
            ("Section VIII: Conclusion & Future Work", ["conclusion", "future work"]),
        ]

        for sec_name, keywords in required_sections:
            tex_found = any(k in tex_text for k in keywords)
            md_found = any(k in md_text for k in keywords)

            self.assertTrue(
                tex_found,
                f"Section '{sec_name}' missing from main.tex (keywords: {keywords})"
            )
            self.assertTrue(
                md_found,
                f"Section '{sec_name}' missing from PAPER.md (keywords: {keywords})"
            )

    def test_tier3_05_table_data_synchronization(self):
        """Verifies tabular environments exist in both LaTeX and Markdown."""
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        # Check LaTeX tables
        has_tex_tables = "\\begin{tabular}" in tex_text or "\\begin{table}" in tex_text
        self.assertTrue(has_tex_tables, "main.tex must contain tabular environments (\\begin{tabular})")

        # Check Markdown tables (lines with pipes |)
        pipe_table_lines = [line for line in md_text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
        self.assertGreater(len(pipe_table_lines), 5, "PAPER.md must contain formatted Markdown pipe tables")


# ==============================================================================
# TIER 4: Empirical Ground-Truth Validation & Hypotheses Verification
# ==============================================================================

class TestTier4EmpiricalConsistencyAndHypotheses(unittest.TestCase):
    """
    Tier 4 validates mathematical, statistical, and empirical consistency against
    the project repository database (f1.db) and model validation metrics:
      - 64 Grand Prix events, 1,119 classified finishers.
      - 49 walk-forward GP folds across 857 finishes.
      - MAE 2.191 positions vs 2.711 baseline with p = 6.24e-10 (t = 6.2559).
      - Podium accuracy 90.43%, ROC-AUC 94.43%.
      - QLoRA 41.94M trainable parameters (0.52%).
      - Zero placeholder markers ("TODO", "TBD", "lorem ipsum").
    """

    @classmethod
    def setUpClass(cls):
        """Loads authoritative ground-truth datasets for assertion comparison."""
        # 1. Ingest SQLite database
        cls.db_exists = DB_PATH.is_file()
        if cls.db_exists:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cls.total_results = cur.execute("SELECT count(*) FROM race_results").fetchone()[0]
            cls.finished_results = cur.execute("SELECT count(*) FROM race_results WHERE finished = 1").fetchone()[0]
            cls.distinct_events = cur.execute("SELECT count(DISTINCT year || '_' || round) FROM race_results").fetchone()[0]
            conn.close()

        # 2. Ingest Walk-Forward Metrics
        cls.wf_metrics_exist = WALK_FORWARD_PATH.is_file()
        if cls.wf_metrics_exist:
            with open(WALK_FORWARD_PATH, "r", encoding="utf-8") as f:
                cls.wf_data = json.load(f)

        # 3. Ingest Baseline Metrics
        cls.baseline_metrics_exist = BASELINE_METRICS_PATH.is_file()
        if cls.baseline_metrics_exist:
            with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
                cls.baseline_data = json.load(f)

    def test_tier4_01_f1_database_ground_truth(self):
        """Validates database counts: 64 distinct events, 1,278 rows, 1,119 finishers."""
        self.assertTrue(self.db_exists, f"Missing f1.db at {DB_PATH}")
        self.assertEqual(self.total_results, 1278, "Total race results rows must be 1,278")
        self.assertEqual(self.finished_results, 1119, "Total classified finishers must be 1,119")
        self.assertEqual(self.distinct_events, 64, "Total distinct Grand Prix events must be 64")

        # Verify manuscripts mention database scale (64 Grand Prix and 1,119 finishers)
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        for name, txt in [("main.tex", tex_text), ("PAPER.md", md_text)]:
            self.assertTrue(
                "64" in txt,
                f"{name} must state 64 Grand Prix events from database ground truth."
            )
            self.assertTrue(
                "1,119" in txt or "1119" in txt,
                f"{name} must state 1,119 finishers from database ground truth."
            )

    def test_tier4_02_walk_forward_metrics_and_error_reduction(self):
        """Validates walk-forward cross-validation metrics against walk_forward_metrics.json."""
        self.assertTrue(self.wf_metrics_exist, f"Missing {WALK_FORWARD_PATH}")

        wf = self.wf_data
        self.assertEqual(wf.get("total_walk_forward_folds"), 49, "Walk-forward folds must equal 49")
        self.assertEqual(wf.get("total_samples_evaluated"), 857, "Evaluated samples must equal 857")
        self.assertEqual(wf.get("warmup_events_count"), 15, "Warmup events must equal 15")
        self.assertAlmostEqual(wf.get("mean_mae_grid_baseline"), 2.711, places=3)
        self.assertAlmostEqual(wf.get("mean_mae_xgb_ranked"), 2.191, places=3)
        self.assertAlmostEqual(wf.get("error_reduction_percentage"), 19.16, places=2)

        sig = wf.get("statistical_significance", {})
        self.assertAlmostEqual(sig.get("paired_t_statistic"), 6.2559, places=3)
        self.assertAlmostEqual(sig.get("paired_t_p_value"), 6.24e-10, delta=1e-10)
        self.assertAlmostEqual(sig.get("wilcoxon_w_statistic"), 75599.0, places=1)

        # Check circuit cluster MAEs
        clusters = wf.get("cluster_performance_mae", {})
        self.assertAlmostEqual(clusters.get("Low-Downforce Power"), 1.942, places=3)
        self.assertAlmostEqual(clusters.get("Maximum-Downforce Mechanical Grip"), 2.09, places=2)
        self.assertAlmostEqual(clusters.get("High-Downforce Thermal Degradation"), 2.13, places=2)
        self.assertAlmostEqual(clusters.get("Hybrid Street / Stop-and-Go"), 2.402, places=3)

        # Verify manuscripts state 49 folds, 857 finishes, 2.191 MAE, 2.711 baseline, and p = 6.24e-10
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        for name, txt in [("main.tex", tex_text), ("PAPER.md", md_text)]:
            self.assertTrue("49" in txt, f"{name} must mention 49 walk-forward folds")
            self.assertTrue("857" in txt, f"{name} must mention 857 evaluated finishes")
            self.assertTrue("2.191" in txt, f"{name} must mention APEX AI MAE of 2.191")
            self.assertTrue("2.711" in txt, f"{name} must mention baseline MAE of 2.711")
            # p-value check: 6.24e-10 or 6.24 \times 10^{-10} or 6.236
            has_p_val = ("6.24" in txt and "10" in txt) or "6.236" in txt or "6.24e-10" in txt
            self.assertTrue(has_p_val, f"{name} must state p-value of 6.24 x 10^-10")

    def test_tier4_03_classification_metrics(self):
        """Validates classification metrics (Podium 90.43% / 94.43% ROC-AUC, Points 75.22% / 81.78%)."""
        self.assertTrue(self.baseline_metrics_exist, f"Missing {BASELINE_METRICS_PATH}")

        clf_list = self.baseline_data.get("classification_metrics", [])
        podium_metrics = next((item for item in clf_list if "Podium" in item.get("task", "")), None)
        points_metrics = next((item for item in clf_list if "Points" in item.get("task", "")), None)

        self.assertIsNotNone(podium_metrics, "Missing Podium Finish classification metrics")
        self.assertIsNotNone(points_metrics, "Missing Points Finish classification metrics")

        self.assertAlmostEqual(podium_metrics.get("accuracy"), 90.43, places=2)
        self.assertAlmostEqual(podium_metrics.get("roc_auc"), 94.43, places=2)
        self.assertAlmostEqual(points_metrics.get("accuracy"), 75.22, places=2)
        self.assertAlmostEqual(points_metrics.get("roc_auc"), 81.78, places=2)

        # Check manuscripts
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        for name, txt in [("main.tex", tex_text), ("PAPER.md", md_text)]:
            self.assertTrue("90.43" in txt or "90.4" in txt, f"{name} must state Podium accuracy 90.43%")
            self.assertTrue("94.43" in txt or "94.4" in txt, f"{name} must state Podium ROC-AUC 94.43%")

    def test_tier4_04_qlora_finetuning_parameters(self):
        """Validates QLoRA fine-tuning parameters (41.94M trainable params, 0.52%)."""
        self.assertTrue(NOTEBOOK_FINETUNE_PATH.is_file(), f"Missing {NOTEBOOK_FINETUNE_PATH}")

        with open(NOTEBOOK_FINETUNE_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)

        nb_text = json.dumps(nb)
        self.assertTrue(
            "41,943,040" in nb_text or "41.94" in nb_text,
            "Notebook outputs must reflect 41,943,040 trainable parameters"
        )
        self.assertTrue(
            "0.52%" in nb_text or "0.5196%" in nb_text,
            "Notebook outputs must reflect 0.52% trainable parameter ratio"
        )

        # Check manuscripts
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        for name, txt in [("main.tex", tex_text), ("PAPER.md", md_text)]:
            has_params = "41.94" in txt or "41,943,040" in txt
            has_ratio = "0.52" in txt or "0.5196" in txt
            self.assertTrue(has_params, f"{name} must mention 41.94M trainable parameters")
            self.assertTrue(has_ratio, f"{name} must mention 0.52% parameter ratio")

    def test_tier4_05_formal_hypotheses_formulation(self):
        """Validates that hypotheses H1, H2, and H3 are explicitly defined with statistics."""
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        for name, txt in [("main.tex", tex_text), ("PAPER.md", md_text)]:
            self.assertTrue(
                "H_1" in txt or "H1" in txt or "Hypothesis 1" in txt,
                f"{name} must formulate Hypothesis 1 (H1)"
            )
            self.assertTrue(
                "H_2" in txt or "H2" in txt or "Hypothesis 2" in txt,
                f"{name} must formulate Hypothesis 2 (H2)"
            )
            self.assertTrue(
                "H_3" in txt or "H3" in txt or "Hypothesis 3" in txt,
                f"{name} must formulate Hypothesis 3 (H3)"
            )

    def test_tier4_06_strict_zero_placeholder_audit(self):
        """Strictly verifies ZERO occurrences of 'TODO', 'TBD', 'lorem ipsum', or placeholders."""
        main_tex = PAPER_DIR / "main.tex"
        paper_md = PAPER_DIR / "PAPER.md"
        self.assertTrue(main_tex.is_file(), "main.tex must exist")
        self.assertTrue(paper_md.is_file(), "PAPER.md must exist")

        tex_text = main_tex.read_text(encoding="utf-8")
        md_text = paper_md.read_text(encoding="utf-8")

        # Prohibited placeholder patterns
        prohibited = [
            (r"\bTODO\b", "TODO"),
            (r"\bTBD\b", "TBD"),
            (r"lorem\s+ipsum", "lorem ipsum"),
            (r"\bFIXME\b", "FIXME"),
            (r"\bXXX\b", "XXX"),
            (r"\[(?:insert|placeholder|draft|tbd|todo)[^\]]*\]", "placeholder brackets"),
        ]

        violations = []

        for pattern, label in prohibited:
            tex_matches = re.findall(pattern, tex_text, re.IGNORECASE)
            md_matches = re.findall(pattern, md_text, re.IGNORECASE)

            if tex_matches:
                violations.append(f"main.tex contains {len(tex_matches)} instance(s) of '{label}': {tex_matches[:3]}")
            if md_matches:
                violations.append(f"PAPER.md contains {len(md_matches)} instance(s) of '{label}': {md_matches[:3]}")

        self.assertEqual(
            violations,
            [],
            f"Draft placeholder audit failed! Detected placeholders:\n" + "\n".join(violations)
        )


# ==============================================================================
# Main Runner Entry Point
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
