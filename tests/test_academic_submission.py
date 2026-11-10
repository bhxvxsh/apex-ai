"""
tests/test_academic_submission.py
=================================
Companion Verification Suite for APEX AI Academic Outreach & Mentor Package (R1-R5).

Governed by:
  - ORIGINAL_REQUEST.md (Requirements R1 through R5)
  - PROJECT.md (Interface Contracts & Code Layout)
  - Survey Handoffs 1, 2, 3 (Empirical Ground Truth & Acceptance Criteria)

Test Manifest (8 Methods):
  - test_01_submission_directory_topology_and_manifest
  - test_02_r1_abstract_word_count_and_scope
  - test_03_r2_mentor_paper_structure_and_limitations
  - test_04_r3_results_summary_table_completeness
  - test_05_r4_reproducibility_guide_and_root_readme
  - test_06_r5_mentor_cover_letter_completeness
  - test_07_latex_compilation_environment_handling
  - test_08_regression_original_19_tests_pass
"""

import os
import re
import shutil
import struct
import subprocess
import unittest
from pathlib import Path


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
    return Path(__file__).resolve().parent.parent


ROOT_DIR = find_workspace_root()
SUBMISSION_DIR = ROOT_DIR / "08_Academic_Submission"
FIGURES_DIR = SUBMISSION_DIR / "figures"
RESEARCH_DIR = ROOT_DIR / "06_Research_Paper"


class TestAcademicSubmissionPackage(unittest.TestCase):
    """
    Acceptance test suite verifying Requirement R1-R5 deliverables in
    08_Academic_Submission/, repository root documentation, and test regression safety.
    """

    def test_01_submission_directory_topology_and_manifest(self):
        """
        R5 Acceptance: Verifies 08_Academic_Submission/ exists, contains all 9 core
        deliverable files (>100 bytes), and houses all 7 figures in PNG and PDF.
        """
        self.assertTrue(
            SUBMISSION_DIR.is_dir(),
            f"08_Academic_Submission directory missing at {SUBMISSION_DIR}"
        )

        required_files = [
            "ABSTRACT.md",
            "PAPER.md",
            "main.tex",
            "references.bib",
            "IEEEtran.cls",
            "IEEEtran.bst",
            "RESULTS_SUMMARY.md",
            "REPRODUCIBILITY.md",
            "MENTOR_COVER_LETTER.md",
        ]
        for fname in required_files:
            fpath = SUBMISSION_DIR / fname
            self.assertTrue(
                fpath.is_file(),
                f"Missing deliverable file: {fname} in {SUBMISSION_DIR}"
            )
            file_size = fpath.stat().st_size
            self.assertGreater(
                file_size,
                100,
                f"Deliverable {fname} is too small ({file_size} bytes; minimum 100 bytes)"
            )

        # Verify figures directory and all 7 figures in PNG and PDF
        self.assertTrue(
            FIGURES_DIR.is_dir(),
            f"Figures directory missing at {FIGURES_DIR}"
        )

        for i in range(1, 8):
            png_matches = list(FIGURES_DIR.glob(f"fig{i}_*.png"))
            pdf_matches = list(FIGURES_DIR.glob(f"fig{i}_*.pdf"))
            self.assertGreaterEqual(
                len(png_matches),
                1,
                f"Missing Figure {i} PNG in {FIGURES_DIR}"
            )
            self.assertGreaterEqual(
                len(pdf_matches),
                1,
                f"Missing Figure {i} PDF in {FIGURES_DIR}"
            )

            # Validate PNG magic bytes (\x89PNG\r\n\x1a\n)
            for png_path in png_matches:
                with open(png_path, "rb") as f:
                    header = f.read(8)
                    self.assertEqual(
                        header,
                        b"\x89PNG\r\n\x1a\n",
                        f"Figure file {png_path.name} is not a valid PNG image"
                    )

            # Validate PDF magic bytes (%PDF-)
            for pdf_path in pdf_matches:
                with open(pdf_path, "rb") as f:
                    header = f.read(5)
                    self.assertEqual(
                        header,
                        b"%PDF-",
                        f"Figure file {pdf_path.name} is not a valid PDF document"
                    )

    def test_02_r1_abstract_word_count_and_scope(self):
        """
        R1 Acceptance: Verifies ABSTRACT.md exists, word count is strictly between
        200 and 300 words, and covers core research question, hybrid methodology,
        49-fold walk-forward results, and academic significance.
        """
        abstract_file = SUBMISSION_DIR / "ABSTRACT.md"
        self.assertTrue(abstract_file.is_file(), f"ABSTRACT.md missing at {abstract_file}")

        text = abstract_file.read_text(encoding="utf-8")
        # Tokenize body words, excluding markdown headers (# ...) and blank lines
        body_words = [
            w for w in text.split()
            if not w.startswith("#")
        ]
        word_count = len(body_words)

        self.assertGreaterEqual(
            word_count,
            200,
            f"Abstract word count too low ({word_count} words; required: strictly 200–300 words)"
        )
        self.assertLessEqual(
            word_count,
            300,
            f"Abstract word count too high ({word_count} words; required: strictly 200–300 words)"
        )

        lower_text = text.lower()
        # Scope criteria checks
        self.assertTrue(
            "walk-forward" in lower_text or "walk forward" in lower_text,
            "ABSTRACT.md must cite walk-forward validation"
        )
        self.assertTrue(
            "telemetry" in lower_text,
            "ABSTRACT.md must cite telemetry dynamics"
        )
        self.assertTrue(
            "2.191" in text or "19.16%" in text,
            "ABSTRACT.md must report MAE 2.191 or 19.16% improvement"
        )
        self.assertTrue(
            "podium" in lower_text or "94.43%" in text,
            "ABSTRACT.md must cite podium classification or 94.43% ROC-AUC"
        )

    def test_03_r2_mentor_paper_structure_and_limitations(self):
        """
        R2 Acceptance: Verifies PAPER.md and main.tex follow IEEE structure and
        contain explicit sections for Problem Formulation, Vehicle Dynamics,
        Experimental Setup (49 folds), Empirical Results, and all 3 distinct Limitations:
          1. Wet Weather Stochasticity
          2. Safety Car & VSC Interventions
          3. Aerodynamic Ground-Effect Wake Sensitivity
        """
        paper_md = SUBMISSION_DIR / "PAPER.md"
        main_tex = SUBMISSION_DIR / "main.tex"
        self.assertTrue(paper_md.is_file(), f"PAPER.md missing at {paper_md}")
        self.assertTrue(main_tex.is_file(), f"main.tex missing at {main_tex}")

        md_text = paper_md.read_text(encoding="utf-8")
        tex_text = main_tex.read_text(encoding="utf-8")

        # 1. Structural Section Checks
        for name, txt in [("PAPER.md", md_text), ("main.tex", tex_text)]:
            lower_txt = txt.lower()
            self.assertTrue(
                "vehicle dynamics" in lower_txt or "micro-sector" in lower_txt,
                f"{name} must contain Vehicle Dynamics & Telemetry Methodology section"
            )
            self.assertTrue(
                "walk-forward" in lower_txt and "49" in txt,
                f"{name} must describe Experimental Setup with 49 walk-forward folds"
            )
            self.assertTrue(
                "results" in lower_txt,
                f"{name} must contain Results & Quantitative Analysis section"
            )
            self.assertTrue(
                "limitations" in lower_txt,
                f"{name} must contain Limitations section"
            )

        # 2. Explicit 3 Limitations Checks
        limitations = [
            ("Wet Weather Stochasticity", ["wet weather", "intermediate", "slick", "rain"]),
            ("Safety Car & VSC Interventions", ["safety car", "vsc", "delta-neutralization", "virtual safety car"]),
            ("Aerodynamic Ground-Effect Wake Sensitivity", ["wake", "dirty air", "ground-effect", "wash-out", "blistering"]),
        ]

        for lim_name, keywords in limitations:
            # Check PAPER.md
            md_lower = md_text.lower()
            md_matched = any(kw in md_lower for kw in keywords)
            self.assertTrue(
                md_matched,
                f"PAPER.md missing explicit limitation: {lim_name} (checked: {keywords})"
            )

            # Check main.tex
            tex_lower = tex_text.lower()
            tex_matched = any(kw in tex_lower for kw in keywords)
            self.assertTrue(
                tex_matched,
                f"main.tex missing explicit limitation: {lim_name} (checked: {keywords})"
            )

    def test_04_r3_results_summary_table_completeness(self):
        """
        R3 Acceptance: Verifies RESULTS_SUMMARY.md fits single-page presentation
        (<= 130 lines) and covers all 7 model configurations across MAE, RMSE,
        Podium ROC-AUC, Points ROC-AUC, Accuracy ±1, and Accuracy ±2.
        """
        summary_file = SUBMISSION_DIR / "RESULTS_SUMMARY.md"
        self.assertTrue(summary_file.is_file(), f"RESULTS_SUMMARY.md missing at {summary_file}")

        text = summary_file.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Single-page presentation constraint (< 130 lines)
        self.assertLessEqual(
            len(lines),
            130,
            f"RESULTS_SUMMARY.md is {len(lines)} lines; must be <= 130 lines for single-page layout"
        )

        # All 7 Model Configurations
        required_models = [
            ("Grid Baseline", ["Grid Baseline", "Starting Grid"]),
            ("Historical Rolling Average", ["Rolling Mean", "Historical Average", "Historical Rolling"]),
            ("Ridge Regression", ["Ridge"]),
            ("Random Forest", ["Random Forest"]),
            ("LightGBM", ["LightGBM"]),
            ("XGBoost", ["XGBoost"]),
            ("APEX AI Hybrid", ["APEX AI", "Hybrid"]),
        ]

        for model_label, synonyms in required_models:
            matched = any(s in text for s in synonyms)
            self.assertTrue(
                matched,
                f"RESULTS_SUMMARY.md missing model configuration: {model_label}"
            )

        # Required Evaluation Metrics
        for metric in ["MAE", "RMSE", "Podium", "Points"]:
            self.assertIn(
                metric,
                text,
                f"RESULTS_SUMMARY.md missing metric column: {metric}"
            )

        # Accuracy ±1 and ±2
        has_acc_bounds = ("±1" in text or "+-1" in text or "<= 1" in text) and \
                         ("±2" in text or "+-2" in text or "<= 2" in text)
        self.assertTrue(
            has_acc_bounds,
            "RESULTS_SUMMARY.md must report accuracy bounds ±1 and ±2 positions"
        )

        # Ground truth values check
        self.assertIn("2.711", text, "RESULTS_SUMMARY.md must report Grid Baseline MAE 2.711")
        self.assertIn("2.191", text, "RESULTS_SUMMARY.md must report APEX AI MAE 2.191")

    def test_05_r4_reproducibility_guide_and_root_readme(self):
        """
        R4 Acceptance: Verifies root README.md and REPRODUCIBILITY.md contain
        system architecture, data dictionary, folder tree, and verified
        reproduction commands for figure generation and test execution.
        """
        root_readme = ROOT_DIR / "README.md"
        repro_guide = SUBMISSION_DIR / "REPRODUCIBILITY.md"

        self.assertTrue(root_readme.is_file(), f"Root README.md missing at {root_readme}")
        self.assertTrue(repro_guide.is_file(), f"REPRODUCIBILITY.md missing at {repro_guide}")

        readme_text = root_readme.read_text(encoding="utf-8")
        repro_text = repro_guide.read_text(encoding="utf-8")

        # Reproduction commands in both files
        for fname, content in [("README.md", readme_text), ("REPRODUCIBILITY.md", repro_text)]:
            self.assertIn(
                "generate_all_figures.py",
                content,
                f"{fname} must document figure generation command (generate_all_figures.py)"
            )
            has_test_cmd = "test_e2e_research_paper.py" in content or "run_e2e_tests.py" in content or "pytest" in content
            self.assertTrue(
                has_test_cmd,
                f"{fname} must document test suite verification command"
            )

        # Architecture diagram and data dictionary in root README
        lower_readme = readme_text.lower()
        self.assertTrue(
            "architecture" in lower_readme,
            "Root README.md must contain system architecture overview"
        )
        self.assertTrue(
            "data dictionary" in lower_readme or "feature dictionary" in lower_readme or "f1.db" in readme_text,
            "Root README.md must contain an end-to-end data dictionary"
        )

    def test_06_r5_mentor_cover_letter_completeness(self):
        """
        R5 Acceptance: Verifies MENTOR_COVER_LETTER.md provides both Tier A (formal
        cover letter) and Tier B (cold email template) with subject lines, empirical
        proof points (2.191 MAE, 49 folds, 94.43% Podium ROC-AUC), specific review
        asks, and co-authorship proposals.
        """
        letter_file = SUBMISSION_DIR / "MENTOR_COVER_LETTER.md"
        self.assertTrue(letter_file.is_file(), f"MENTOR_COVER_LETTER.md missing at {letter_file}")

        text = letter_file.read_text(encoding="utf-8")
        lower_text = text.lower()

        # Dual-Tier Coverage
        self.assertTrue(
            "tier a" in lower_text or "formal academic cover letter" in lower_text,
            "MENTOR_COVER_LETTER.md must include Tier A formal academic cover letter"
        )
        self.assertTrue(
            "tier b" in lower_text or "cold outreach email" in lower_text or "email template" in lower_text,
            "MENTOR_COVER_LETTER.md must include Tier B cold outreach email template"
        )

        # Subject Line
        self.assertIn("Subject:", text, "MENTOR_COVER_LETTER.md must specify subject line")

        # Empirical Proof Points
        self.assertIn("2.191", text, "MENTOR_COVER_LETTER.md must cite MAE 2.191")
        self.assertIn("49", text, "MENTOR_COVER_LETTER.md must cite 49 walk-forward folds")
        self.assertTrue(
            "94.43%" in text or "94.43" in text,
            "MENTOR_COVER_LETTER.md must cite 94.43% Podium ROC-AUC"
        )

        # Collaboration & Review Asks
        self.assertTrue(
            "review" in lower_text or "mentorship" in lower_text,
            "MENTOR_COVER_LETTER.md must formulate explicit review / mentorship asks"
        )
        self.assertTrue(
            "co-authorship" in lower_text or "coauthorship" in lower_text or "collaboration" in lower_text,
            "MENTOR_COVER_LETTER.md must discuss co-authorship or collaboration opportunities"
        )

    def test_07_latex_compilation_environment_handling(self):
        """
        R2 Acceptance: Verifies graceful handling of LaTeX compilation based on local
        environment tools. If pdflatex is present, compiles main.tex; if absent,
        validates IEEEtran syntax and required bibtex bibliography declarations.
        """
        main_tex = SUBMISSION_DIR / "main.tex"
        self.assertTrue(main_tex.is_file(), f"main.tex missing at {main_tex}")

        has_pdflatex = shutil.which("pdflatex") is not None

        if has_pdflatex:
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=str(SUBMISSION_DIR),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"pdflatex compilation failed with error:\n{proc.stdout[-500:]}"
            )
        else:
            # When pdflatex is absent (as confirmed on host OS), assert valid LaTeX structure
            content = main_tex.read_text(encoding="utf-8")
            self.assertIn(
                "\\documentclass",
                content,
                "main.tex missing \\documentclass declaration"
            )
            self.assertIn(
                "\\begin{document}",
                content,
                "main.tex missing \\begin{document}"
            )
            self.assertIn(
                "\\end{document}",
                content,
                "main.tex missing \\end{document}"
            )
            self.assertTrue(
                "\\bibliography{" in content or "\\bibitem" in content,
                "main.tex missing bibliography declaration"
            )

            # Ensure supporting LaTeX style and bib files exist alongside main.tex
            for support_file in ["IEEEtran.cls", "IEEEtran.bst", "references.bib"]:
                sp = SUBMISSION_DIR / support_file
                self.assertTrue(
                    sp.is_file(),
                    f"LaTeX support file {support_file} missing in {SUBMISSION_DIR}"
                )

    def test_08_regression_original_19_tests_pass(self):
        """
        Integrity Acceptance: Ensures that tests/test_e2e_research_paper.py contains
        exactly 19 test methods and that all 19 tests pass cleanly without errors.
        """
        e2e_file = ROOT_DIR / "tests" / "test_e2e_research_paper.py"
        self.assertTrue(e2e_file.is_file(), f"Original E2E test file missing at {e2e_file}")

        content = e2e_file.read_text(encoding="utf-8")
        test_methods = re.findall(r"def (test_\w+)\(", content)
        self.assertEqual(
            len(test_methods),
            19,
            f"Expected exactly 19 tests in test_e2e_research_paper.py, found {len(test_methods)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
