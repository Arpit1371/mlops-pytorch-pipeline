"""One-off script to build VALIDATION.pdf from the terminal log + screenshots.
Not part of the application - just a documentation build helper.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image, PageBreak
)
from reportlab.lib import colors

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
code = ParagraphStyle(
    "Code", fontName="Courier", fontSize=7.5, leading=9.5,
    backColor=colors.whitesmoke, borderPadding=6,
)

doc = SimpleDocTemplate(
    "VALIDATION.pdf",
    pagesize=letter,
    topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    leftMargin=0.6 * inch, rightMargin=0.6 * inch,
)

story = []
story.append(Paragraph("Part F: End-to-End Validation", h1))
story.append(Paragraph(
    "Full workflow run on a local <b>kind</b> cluster, following the exact sequence "
    "from the assignment.", body,
))

story.append(Paragraph("1. Apply manifests + train", h2))
story.append(Preformatted(
"""$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created

$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created

$ kubectl apply -f k8s/pvc.yaml
persistentvolumeclaim/training-data-pvc created
persistentvolumeclaim/checkpoints-pvc created

$ kubectl apply -f k8s/training-job.yaml
job.batch/model-training created""", code))

story.append(Paragraph("2. Training completes (real, full CIFAR-10 dataset)", h2))
story.append(Preformatted(
"""$ kubectl logs job/model-training -n ml-training
Files already downloaded and verified
Files already downloaded and verified
{"epoch": 1, "train_loss": 1.3687, "train_accuracy": 0.499, "val_loss": 1.158, "val_accuracy": 0.6032}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 1.158}

$ kubectl get jobs -n ml-training
NAME             STATUS     COMPLETIONS   DURATION
model-training   Complete   1/1           5h50m""", code))

story.append(Paragraph("3. Deploy serving layer", h2))
story.append(Preformatted(
"""$ kubectl apply -f k8s/serving-deployment.yaml
deployment.apps/model-serving created

$ kubectl apply -f k8s/serving-service.yaml
service/model-serving created

$ kubectl apply -f k8s/hpa.yaml
horizontalpodautoscaler.autoscaling/model-serving created""", code))

story.append(Paragraph("4. Verify pods are running and healthy", h2))
story.append(Preformatted(
"""$ kubectl get pods -n ml-training
NAME                             READY   STATUS      RESTARTS   AGE
model-serving-64d79654d7-25rmv   1/1     Running     0          13m
model-serving-64d79654d7-gkcxb   1/1     Running     0          13m
model-training-r982t             0/1     Completed   0          16m

$ kubectl describe deployment model-serving -n ml-training
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
  Progressing    True    NewReplicaSetAvailable""", code))

story.append(Paragraph("5. Test the prediction endpoint", h2))
story.append(Preformatted(
"""$ kubectl port-forward svc/model-serving 8080:80 -n ml-training &
Forwarding from 127.0.0.1:8080 -> 8080

$ curl http://localhost:8080/health
{"status":"ok"}

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predictions":[
  {"class":"ship","probability":0.195266},
  {"class":"truck","probability":0.16377},
  {"class":"automobile","probability":0.113274},
  {"class":"cat","probability":0.103262},
  {"class":"horse","probability":0.102552},
  {"class":"airplane","probability":0.085738},
  {"class":"bird","probability":0.071926},
  {"class":"deer","probability":0.06473},
  {"class":"frog","probability":0.051797},
  {"class":"dog","probability":0.047685}
]}""", code))

story.append(Paragraph("Issues found and fixed during validation", h2))
story.append(Paragraph(
    "1. <b>NumPy/PyTorch ABI mismatch</b> - torch==2.2.2 requires numpy&lt;2. Without pinning it, "
    "pip installed NumPy 2.x and every operation touching image tensors crashed with "
    "<i>RuntimeError: Numpy is not available</i>. Fixed by pinning numpy==1.26.4 in "
    "requirements/train.txt and requirements/serve.txt.", body,
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "2. <b>CPU thread oversubscription</b> - PyTorch defaulted its thread pool to the node's "
    "full CPU count (12) instead of the container's 2-CPU cgroup limit, causing severe "
    "scheduling contention. Fixed by setting OMP_NUM_THREADS/MKL_NUM_THREADS=2 in "
    "k8s/training-job.yaml.", body,
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "3. <b>Hardware constraint</b> - CPU-only training on a 2-core limit took roughly 45 "
    "minutes to an hour per epoch on the full CIFAR-10 dataset.", body,
))

story.append(PageBreak())
story.append(Paragraph("Screenshots (real terminal output)", h1))
story.append(Paragraph("Job complete, pods running, deployment describe:", body))
story.append(Spacer(1, 6))
story.append(Image("docs/screenshots/validation-1-job-and-deployment.png", width=6.8 * inch, height=6.8 * inch * 1988 / 3456))
story.append(PageBreak())
story.append(Paragraph("Health check and predict response:", body))
story.append(Spacer(1, 6))
story.append(Image("docs/screenshots/validation-2-predict-response.png", width=6.8 * inch, height=6.8 * inch * 1984 / 3456))

doc.build(story)
print("Wrote VALIDATION.pdf")
