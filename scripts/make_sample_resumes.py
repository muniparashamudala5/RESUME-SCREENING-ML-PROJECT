"""Generate fictional sample resumes into ./samples (needs `pip install reportlab`).

    python scripts/make_sample_resumes.py

Also imported by the tests, which build the same PDFs in memory.
"""
from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph,
                                SimpleDocTemplate, Spacer)

STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=STYLES["Title"], fontSize=20, spaceAfter=2, alignment=0)
H2 = ParagraphStyle("H2", parent=STYLES["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=3,
                    textColor=colors.HexColor("#1c2b2d"))
BODY = ParagraphStyle("Body", parent=STYLES["BodyText"], fontSize=9.5, leading=12.5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11)

# Each resume is a list of (heading, [lines]). Lines starting with "- " become bullets.
RESUMES: dict[str, dict] = {
    "strong_data_scientist": {
        "name": "Ananya Rao",
        "contact": "ananya.rao@example.com | +91 98765 43210 | linkedin.com/in/ananya-rao | github.com/ananya-rao",
        "sections": [
            ("Summary", ["Data scientist with 3 years of experience building machine learning models for retail and fintech."]),
            ("Skills", ["Python, SQL, Pandas, NumPy, scikit-learn, XGBoost, TensorFlow, Statistics, Feature Engineering, Docker, Git, AWS, Jupyter"]),
            ("Experience", [
                "<b>Data Scientist</b>, RetailCo Analytics - Jan 2024 - Present",
                "- Built churn prediction models with scikit-learn and XGBoost, improving recall by 18%.",
                "- Wrote SQL and Pandas pipelines for feature engineering and EDA on 20M rows.",
                "- Ran A/B testing and hypothesis testing for pricing experiments; applied statistics to size effects.",
                "- Deployed models on AWS using Docker; tuned hyperparameters with cross validation.",
                "<b>Data Analyst Intern</b>, FinServe - Jun 2023 - Dec 2023",
                "- Cleaned data and built regression and time series forecasting models in Python.",
            ]),
            ("Education", ["B.Tech in Computer Science, National Institute of Technology, 2019 - 2023, CGPA 8.7"]),
            ("Projects", [
                "- Customer segmentation: clustering with scikit-learn and Pandas, visualised in Jupyter.",
                "- Sales forecasting: time series model in Python with a deep learning baseline in TensorFlow.",
            ]),
            ("Certifications", ["Deep Learning Specialization; AWS Cloud Practitioner"]),
        ],
    },
    "analyst_fresher": {
        "name": "Rohit Verma",
        "contact": "rohit.verma@example.com | +91 90123 45678 | github.com/rohitverma",
        "sections": [
            ("Objective", ["Fresher looking for a data analyst role."]),
            ("Skills", ["SQL, Excel, Power BI, Python, Statistics, Data Visualization"]),
            ("Internship", [
                "<b>Business Analytics Intern</b>, BrightMart - Jun 2025 - Nov 2025",
                "- Built Power BI dashboards tracking KPIs; wrote SQL queries and did data cleaning in Excel.",
            ]),
            ("Education", ["B.Sc in Statistics, Sri Venkateswara University, 2022 - 2025"]),
            ("Projects", ["- Retail sales dashboard in Power BI with SQL Server backend."]),
        ],
    },
    "frontend_developer": {
        "name": "Meera Iyer",
        "contact": "meera.iyer@example.com | +91 98450 11223 | github.com/meera-iyer",
        "sections": [
            ("Skills", ["HTML, CSS, JavaScript, React, TypeScript, Tailwind CSS, Git, Figma, Responsive Design"]),
            ("Experience", [
                "<b>Frontend Developer</b>, PixelWorks - Mar 2022 - Present",
                "- Built responsive React interfaces with TypeScript and Tailwind CSS; improved Lighthouse performance by 30%.",
                "- Wrote unit tests for components and consumed REST APIs.",
            ]),
            ("Education", ["Diploma in Computer Engineering, Government Polytechnic, 2019 - 2022"]),
        ],
    },
    # Two-column layout: header across the top, sidebar (contact/skills/education) left, main right.
    "two_column_ml_engineer": {
        "name": "Karthik Naidu",
        "contact": "karthik.naidu@example.com | +91 99887 76655 | github.com/karthik-naidu",
        "layout": "two_column",
        "left": [
            ("Skills", ["Python, PyTorch, TensorFlow, Machine Learning, Deep Learning, Docker, Git, FastAPI, MLflow, AWS, SQL, Model Deployment"]),
            ("Education", ["B.E in Computer Science, Anna University, 2020 - 2024"]),
            ("Certifications", ["Machine Learning Specialization", "AWS Cloud Practitioner"]),
        ],
        "right": [
            ("Experience", [
                "<b>Machine Learning Engineer</b>, VisionAI Labs - Aug 2024 - Present",
                "- Trained deep learning models in PyTorch and served them through FastAPI in Docker containers.",
                "- Tracked experiments with MLflow and set up Model Deployment on AWS.",
                "<b>ML Intern</b>, DataNest - Jan 2024 - Jul 2024",
                "- Built machine learning pipelines with Python and SQL.",
            ]),
            ("Projects", [
                "- Object detection service: TensorFlow model, Dockerised and deployed with CI/CD on GitHub Actions.",
                "- Resume parser using NLP and Hugging Face transformers.",
            ]),
        ],
    },
}


def _flowables(sections: list[tuple[str, list[str]]]) -> list:
    out = []
    for heading, lines in sections:
        out.append(Paragraph(heading.upper(), H2))
        for line in lines:
            out.append(Paragraph(line.replace("- ", "&bull; ", 1) if line.startswith("- ") else line, BODY))
    return out


def build_pdf(key: str) -> bytes:
    spec = RESUMES[key]
    buffer = io.BytesIO()
    header = [Paragraph(spec["name"], H1), Paragraph(spec["contact"], SMALL), Spacer(1, 6)]

    if spec.get("layout") == "two_column":
        width, height = A4
        margin, gap = 40, 24
        top_h = 70
        col_w = (width - 2 * margin - gap) / 2
        body_h = height - 2 * margin - top_h
        frames = [
            Frame(margin, height - margin - top_h, width - 2 * margin, top_h, id="head", showBoundary=0),
            Frame(margin, margin, col_w, body_h, id="left", showBoundary=0),
            Frame(margin + col_w + gap, margin, col_w, body_h, id="right", showBoundary=0),
        ]
        doc = BaseDocTemplate(buffer, pagesize=A4)
        doc.addPageTemplates([PageTemplate(id="two", frames=frames)])
        story = header + [FrameBreak()] + _flowables(spec["left"]) + [FrameBreak()] + _flowables(spec["right"])
        doc.build(story)
    else:
        doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=40)
        doc.build(header + _flowables(spec["sections"]))
    return buffer.getvalue()


def build_scanned_pdf(key: str = "strong_data_scientist") -> bytes:
    """Rasterise a sample resume into an image-only PDF (no text layer) to exercise OCR."""
    import pdfplumber

    with pdfplumber.open(io.BytesIO(build_pdf(key))) as pdf:
        image = pdf.pages[0].to_image(resolution=200).original.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "samples"
    out.mkdir(exist_ok=True)
    for key in RESUMES:
        (out / f"{key}.pdf").write_bytes(build_pdf(key))
    (out / "scanned_data_scientist.pdf").write_bytes(build_scanned_pdf())
    print(f"Wrote {len(RESUMES) + 1} sample PDFs to {out}")


if __name__ == "__main__":
    main()
