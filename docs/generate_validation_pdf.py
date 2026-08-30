"""One-off script to build VALIDATION.pdf: reflection write-up + screenshots.
Not part of the application - just a documentation build helper.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
)

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=12)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=10)

doc = SimpleDocTemplate(
    "VALIDATION.pdf",
    pagesize=letter,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    leftMargin=0.7 * inch, rightMargin=0.7 * inch,
)

story = []
story.append(Paragraph("What was the most challenging part?", h1))

story.append(Paragraph(
    "The hardest part of this assignment wasn't writing the model or the Dockerfiles - it "
    "was getting the actual Kubernetes training Job to run correctly, because two separate "
    "bugs only showed up once real data hit the pipeline, not during simple builds or health "
    "checks.", body,
))
story.append(Paragraph(
    "The first issue was a NumPy/PyTorch version conflict. Both requirements files pinned "
    "torch==2.2.2 but never pinned numpy, so pip quietly installed NumPy 2.x alongside it. "
    "The Docker image built fine, and the serving container's /health check even passed, "
    "because none of that touches real data. But the moment the training Job tried to load "
    "and transform an actual CIFAR-10 image, it crashed with \"RuntimeError: Numpy is not "
    "available\" - torch 2.2.2's compiled extensions expect NumPy 1.x's ABI. The fix was one "
    "line (numpy==1.26.4 in both requirements files), but finding it took an actual "
    "end-to-end run, since Docker-level testing alone never exercised that code path.", body,
))
story.append(Paragraph(
    "The second issue was sneakier: the training Job would sit there \"Running\" for 40+ "
    "minutes without producing a single log line, even though the pod's CPU usage looked "
    "fully active. It wasn't crashing or restarting, just seemingly stuck. Digging into "
    "/proc inside the pod showed the process was genuinely burning CPU the whole time, not "
    "idle - so it wasn't a real deadlock. The actual cause was that PyTorch defaults its "
    "internal thread pool to the machine's total CPU count (12, in this case), but the Job's "
    "resource limit only allowed it 2 CPU-cores worth of scheduling time. Twelve threads "
    "fighting over two cores' worth of quota meant most of the CPU time went to "
    "context-switching overhead instead of actual computation. Setting OMP_NUM_THREADS and "
    "MKL_NUM_THREADS to 2 (matching the CPU limit) fixed it - the same benchmark that hadn't "
    "finished after 5+ minutes at 12 threads finished predictably once capped correctly.", body,
))
story.append(Paragraph(
    "A smaller, more mundane challenge was just how slow CPU-only training is for something "
    "like ResNet-18 - a single epoch on the full CIFAR-10 dataset took roughly an hour even "
    "after fixing the threading issue, which made iterating on the Kubernetes manifests slow "
    "and required patience (and a scaled-down config) to validate the mechanics quickly "
    "before committing to a full run.", body,
))
story.append(Paragraph(
    "Overall, the biggest lesson was that Docker build success and a passing health check "
    "don't prove a pipeline actually works - only running it against real data on real "
    "infrastructure surfaces the failures that matter.", body,
))

story.append(PageBreak())
story.append(Image("docs/screenshots/validation-1-job-and-deployment.png", width=6.8 * inch, height=6.8 * inch * 1988 / 3456))
story.append(PageBreak())
story.append(Image("docs/screenshots/validation-2-predict-response.png", width=6.8 * inch, height=6.8 * inch * 1984 / 3456))

doc.build(story)
print("Wrote VALIDATION.pdf")
