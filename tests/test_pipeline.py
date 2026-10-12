"""Run with:  python -m unittest discover -s tests -v   (or: pytest)

Needs the dev dependencies (reportlab, and httpx for the API tests). OCR tests are skipped when the
tesseract binary is not installed; API tests are skipped when FastAPI is not installed.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_sample_resumes as samples  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.extractor import ExtractionError, extract_text  # noqa: E402
from app.parser import SkillMatcher, detect_education, estimate_experience, parse_contact  # noqa: E402
from app.preprocess import clean_text, split_sections  # noqa: E402
from app.roles import load_profiles  # noqa: E402
from app.scorer import Screener, UnknownRoleError  # noqa: E402

TODAY = date(2026, 9, 21)


class ScreenerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.screener = Screener.from_settings(get_settings())
        cls.pdfs = {key: samples.build_pdf(key) for key in samples.RESUMES}


class ExtractionTests(ScreenerTestCase):
    def test_single_column_text_and_sections(self):
        result = extract_text(self.pdfs["strong_data_scientist"])
        self.assertEqual(result.method, "pdfplumber")
        self.assertIn("scikit-learn", result.text)
        sections = split_sections(clean_text(result.text))
        for name in ("skills", "experience", "education", "projects"):
            self.assertIn(name, sections)

    def test_two_column_pdf_is_read_column_by_column(self):
        text = extract_text(self.pdfs["two_column_ml_engineer"]).text
        # header first, then the whole left column, then the right column
        self.assertLess(text.index("Karthik Naidu"), text.index("SKILLS"))
        self.assertLess(text.index("EDUCATION"), text.index("EXPERIENCE"))
        # a right-column entry stays together instead of being interleaved with the left column
        self.assertIn("VisionAI Labs - Aug\n2024 - Present", text)
        sections = split_sections(clean_text(text))
        self.assertIn("Anna University", sections["education"])
        self.assertNotIn("Anna University", sections["experience"])

    def test_rejects_bad_files(self):
        with self.assertRaises(ExtractionError):
            extract_text(b"")
        with self.assertRaises(ExtractionError):
            extract_text(b"just some text, not a pdf")
        with self.assertRaises(ExtractionError):
            extract_text(b"%PDF-1.4\nthis is not really a pdf")

    @unittest.skipUnless(shutil.which("tesseract"), "tesseract is not installed")
    def test_scanned_pdf_falls_back_to_ocr(self):
        result = extract_text(samples.build_scanned_pdf())
        self.assertEqual(result.method, "ocr")
        self.assertTrue(result.warnings)
        self.assertIn("scikit-learn", result.text.lower())


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.matcher = SkillMatcher({"sql": ("mysql", "postgresql"), "scikit-learn": ("sklearn",), "python": ()})

    def match(self, label, text):
        return self.matcher.search(label, text.lower())

    def test_aliases_and_spelling_variants(self):
        self.assertEqual(self.match("scikit-learn", "used sklearn for models").term, "sklearn")
        self.assertIsNotNone(self.match("scikit-learn", "Scikit learn and pandas"))
        self.assertIsNotNone(self.match("SQL", "worked with PostgreSQL daily"))

    def test_word_boundaries(self):
        self.assertIsNone(self.match("SQL", "MongoDB and other nosql stores"))
        self.assertIsNone(self.match("Python", "monty pythonic humour"))
        self.assertIsNotNone(self.match("C++", "systems programming in c++, rust"))
        self.assertIsNone(self.match("C", "c++ only"))

    def test_experience_ranges(self):
        info = estimate_experience("Intern, Acme - Oct 2025 - March 2026", {}, TODAY)
        self.assertEqual(info.years, 0.5)
        info = estimate_experience("Engineer, Acme - Jan 2024 - Present", {}, TODAY)
        self.assertEqual(info.years, 2.8)  # 33 months, both ends inclusive

    def test_overlapping_ranges_are_merged(self):
        text = "Job A: Jan 2022 - Dec 2023\nJob B: Jun 2023 - Dec 2024"
        self.assertEqual(estimate_experience(text, {}, TODAY).years, 3.0)

    def test_education_dates_do_not_count_as_experience(self):
        text = "B.Tech in Computer Science, SVCE, 2022-2025\nDiploma in Electronics, GPT, 2019-2022"
        self.assertEqual(estimate_experience(text, {}, TODAY).years, 0.0)

    def test_stated_experience_is_a_fallback(self):
        info = estimate_experience("Engineer with 4+ years of experience in data.", {}, TODAY)
        self.assertEqual((info.years, info.source), (4.0, "stated in resume"))

    def test_education_levels(self):
        self.assertEqual(detect_education("B.Tech in Computer Science", {}), 2)
        self.assertEqual(detect_education("M.Tech, IIT", {}), 3)
        self.assertEqual(detect_education("Diploma in Electronics", {}), 1)
        self.assertEqual(detect_education("Master's degree in Statistics", {}), 3)
        self.assertEqual(detect_education("I want to be a data scientist", {}), 0)  # 'be' is not a degree

    def test_contact(self):
        c = parse_contact("Ananya Rao\nananya@example.com | +91 98765 43210 | github.com/ananya\n")
        self.assertEqual((c.name, c.email, c.phone), ("Ananya Rao", "ananya@example.com", "+91 98765 43210"))
        self.assertEqual(c.links, ["github.com/ananya"])


class ScreeningTests(ScreenerTestCase):
    def screen(self, key, role):
        return self.screener.screen_pdf(self.pdfs[key], role)

    def test_strong_match_is_shortlisted_with_explanation(self):
        r = self.screen("strong_data_scientist", "data_scientist")
        self.assertEqual(r["decision"], "Shortlisted")
        self.assertGreaterEqual(r["score"], 80)
        self.assertEqual(r["summary"]["must_have_matched"], r["summary"]["must_have_total"])
        self.assertAlmostEqual(sum(item["earned"] for item in r["breakdown"]), r["score"], places=1)
        self.assertGreater(r["candidate"]["experience_years"], 3)
        self.assertEqual(r["candidate"]["email"], "ananya.rao@example.com")

    def test_unrelated_resume_is_not_shortlisted(self):
        r = self.screen("frontend_developer", "data_scientist")
        self.assertEqual(r["decision"], "Not shortlisted")
        self.assertEqual(r["summary"]["must_have_matched"], 0)

    def test_same_resume_different_roles(self):
        fit = self.screen("analyst_fresher", "data_analyst")["score"]
        stretch = self.screen("analyst_fresher", "data_scientist")["score"]
        self.assertGreater(fit, stretch + 20)

    def test_any_of_requirement_is_satisfied_by_one_option(self):
        r = self.screen("analyst_fresher", "data_analyst")  # lists Power BI but not Tableau
        labels = [h["label"] for h in r["skills"]["must_have"]["matched"]]
        self.assertIn("Power BI / Tableau", labels)

    def test_two_column_resume_scores_like_a_single_column_one(self):
        r = self.screen("two_column_ml_engineer", "machine_learning_engineer")
        self.assertEqual(r["decision"], "Shortlisted")
        self.assertEqual(r["candidate"]["education"], "Bachelor's degree")

    def test_skills_listed_but_never_used_earn_partial_credit(self):
        text = "Jane Doe\njane@example.com\nSKILLS\nPython, SQL, Pandas, Statistics, Machine Learning, scikit-learn\nEDUCATION\nB.Tech, 2020 - 2024"
        r = self.screener.screen_text(text, "data_scientist", today=TODAY)
        self.assertTrue(all(h["evidence"] == "listed" for h in r["skills"]["must_have"]["matched"]))
        self.assertAlmostEqual(r["breakdown"][0]["earned"], 45 * 0.8, places=1)

    def test_keyword_stuffing_is_flagged(self):
        text = "Jane Doe\nSKILLS\n" + "Python " * 15 + "\nEXPERIENCE\nIntern - Jan 2025 - Jun 2025"
        r = self.screener.screen_text(text, "data_scientist", today=TODAY)
        self.assertTrue(any("keyword stuffing" in w for w in r["warnings"]))

    def test_job_description_changes_similarity_only(self):
        base = self.screener.screen_text("Jane\nSKILLS\nPython, SQL", "data_scientist", today=TODAY)
        with_jd = self.screener.screen_text(
            "Jane\nSKILLS\nPython, SQL", "data_scientist", "We need python and sql for analytics work", today=TODAY
        )
        self.assertEqual(base["breakdown"][0], with_jd["breakdown"][0])  # must-have points unchanged
        self.assertNotEqual(base["breakdown"][3]["earned"], with_jd["breakdown"][3]["earned"])

    def test_ranking_orders_best_first_and_reports_errors(self):
        files = [
            ("front.pdf", self.pdfs["frontend_developer"]),
            ("broken.pdf", b"not a pdf"),
            ("ds.pdf", self.pdfs["strong_data_scientist"]),
        ]
        ranked = self.screener.screen_many(files, "data_scientist")
        self.assertEqual([r["filename"] for r in ranked], ["ds.pdf", "front.pdf", "broken.pdf"])
        self.assertIn("error", ranked[-1])

    def test_unknown_role(self):
        with self.assertRaises(UnknownRoleError):
            self.screener.screen_pdf(self.pdfs["strong_data_scientist"], "astronaut")


class ConfigTests(unittest.TestCase):
    def test_shipped_roles_load(self):
        profiles = load_profiles(get_settings().roles_path)
        self.assertGreaterEqual(len(profiles.roles), 5)
        self.assertAlmostEqual(sum(profiles.scoring.weights.values()), 1.0)

    def test_weights_must_add_up_to_one(self):
        raw = json.loads(get_settings().roles_path.read_text(encoding="utf-8"))
        raw["scoring"]["weights"]["must_have"] = 0.9
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(raw, fh)
        with self.assertRaises(ValueError):
            load_profiles(fh.name)


try:
    from fastapi.testclient import TestClient
except ImportError:  # FastAPI / httpx not installed
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi and httpx are not installed")
class ApiTests(ScreenerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app.main import app

        cls.client = TestClient(app)

    def post(self, name="cv.pdf", data=None, ctype="application/pdf", role="data_scientist"):
        payload = self.pdfs["strong_data_scientist"] if data is None else data
        return self.client.post("/api/screen", files={"file": (name, payload, ctype)}, data={"role": role})

    def test_health_and_roles(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        roles = self.client.get("/api/roles").json()
        self.assertIn("data_scientist", [r["id"] for r in roles])

    def test_screen_ok(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "Shortlisted")

    def test_error_codes(self):
        self.assertEqual(self.post(role="astronaut").status_code, 404)
        self.assertEqual(self.post(name="cv.txt", data=b"hello", ctype="text/plain").status_code, 415)
        self.assertEqual(self.post(data=b"hello, not a pdf").status_code, 422)
        too_big = b"%PDF-1.4" + b"0" * (get_settings().max_upload_bytes + 10)
        self.assertEqual(self.post(data=too_big).status_code, 413)

    def test_batch_is_ranked(self):
        files = [
            ("files", ("a.pdf", self.pdfs["frontend_developer"], "application/pdf")),
            ("files", ("b.pdf", self.pdfs["strong_data_scientist"], "application/pdf")),
        ]
        body = self.client.post("/api/screen/batch", files=files, data={"role": "data_scientist"}).json()
        self.assertEqual([r["filename"] for r in body["results"]], ["b.pdf", "a.pdf"])

    def test_security_headers(self):
        headers = self.client.get("/").headers
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertIn("default-src 'self'", headers["content-security-policy"])


if __name__ == "__main__":
    unittest.main()
