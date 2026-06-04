"""Central configuration for the document forgery detection POC.

ALL external API keys and tunable thresholds live here. Nothing outside this
module should hardcode an endpoint, key, or threshold.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
ADVERSARIAL_DIR = BASE_DIR / "adversarial_patterns"
MODELS_DIR = BASE_DIR / "models"
FAISS_DIR = BASE_DIR / "faiss_index"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_CACHE_DIR = TEMPLATES_DIR / "cache"  # azure_blob downloads cached here

for _d in (UPLOADS_DIR, TEMPLATES_DIR, ADVERSARIAL_DIR, MODELS_DIR, FAISS_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Persisted Step-3 OOD FAISS index + sidecar metadata (Agent 4 / Step 3).
TEMPLATE_INDEX_PATH = FAISS_DIR / "template_embeddings.index"
TEMPLATE_INDEX_META_PATH = FAISS_DIR / "template_embeddings.meta.json"
# Agent 9 saved (fine-tuned) weights, produced by scripts/finetune_agent9.py.
AGENT9_WEIGHTS_DIR = MODELS_DIR / "agent9_weights"

# --- External API Keys ---
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Azure OpenAI (Azure AI Foundry) for Step 5 cross-doc reasoning ---
# Preferred LLM backend. When all four are set, cross_doc_llm uses the
# AzureOpenAI client (deployment-based) ahead of plain OpenAI / Anthropic.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")          # https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")      # deployment name, e.g. gpt-4o
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

# --- Azure Blob Storage (future: authentic template corpus) ---
# Templates organised inside the container as: <doc_type>/<filename>
# e.g. payslip/cognizant_payslip_v1.pdf
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
AZURE_BLOB_TEMPLATE_CONTAINER = os.getenv(
    "AZURE_BLOB_TEMPLATE_CONTAINER", "authentic-templates")

# --- Template & OOD index source ---
# "local"      -> use the templates/ directory (POC default, no Azure dependency)
# "azure_blob" -> fetch from the Blob container above (future; flip this switch
#                 and provide AZURE_BLOB_CONNECTION_STRING, no code changes)
TEMPLATE_SOURCE = os.getenv("TEMPLATE_SOURCE", "local")

# --- Model Selection ---
CROSS_DOC_MODEL = os.getenv("CROSS_DOC_MODEL", "gpt-4-turbo")  # or claude-opus-4-6

# Agent 15 (VLLM field extraction) descoped.
# Claude Opus (Agents 13 + 14) covers the LLM reasoning layer.
# Revisit only if a 4-way OCR consensus signal is needed in future.

# --- Agent 13 (Claude Holistic Plausibility) ---
AGENT13_MODEL = os.getenv("AGENT13_MODEL", "claude-opus-4-6")

# --- Agent 14 (Claude Cross-Agent Adjudicator) ---
AGENT14_MODEL = os.getenv("AGENT14_MODEL", "claude-opus-4-6")

# --- Agent 9 backend selector ---
# "pca_autoencoder" = current approved fallback (no GPU required)
# "patchcore"       = upgrade path (requires anomalib + GPU, see requirements-optional.txt)
AGENT9_BACKEND = os.getenv("AGENT9_BACKEND", "pca_autoencoder")

# --- Hugging Face model IDs (Step 3 - Document Understanding) ---
DIT_MODEL_ID = os.getenv("DIT_MODEL_ID", "microsoft/dit-base")
LAYOUTLMV3_MODEL_ID = os.getenv("LAYOUTLMV3_MODEL_ID", "microsoft/layoutlmv3-base")
DONUT_MODEL_ID = os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base")

# --- Rendering ---
RASTER_DPI = int(os.getenv("RASTER_DPI", "150"))  # all bboxes are in 150-DPI px

# --- OCR engine toggles ---
# PaddleOCR (engine #2) downloads model weights from a hosting platform on first
# use. In offline/restricted environments this fails, so it is OFF by default;
# set ENABLE_PADDLEOCR=1 once the weights are reachable to activate the second
# OCR engine (which in turn activates Agent 10 - Cross-OCR Disagreement).
ENABLE_PADDLEOCR = os.getenv("ENABLE_PADDLEOCR", "0") == "1"

# --- Detection Thresholds ---
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "1.5"))
AGENT2_THRESHOLD = float(os.getenv("AGENT2_THRESHOLD", "0.4"))
AGENT4_SSIM_THRESHOLD = float(os.getenv("AGENT4_SSIM_THRESHOLD", "0.7"))
AGENT5_PHASH_THRESHOLD = int(os.getenv("AGENT5_PHASH_THRESHOLD", "10"))
AGENT9_THRESHOLD = float(os.getenv("AGENT9_THRESHOLD", "0.6"))
AGENT9_NOVELTY_HIGH = float(os.getenv("AGENT9_NOVELTY_HIGH", "0.75"))
AGENT10_THRESHOLD = float(os.getenv("AGENT10_THRESHOLD", "0.5"))
AGENT11_DELTA_THRESHOLD = float(os.getenv("AGENT11_DELTA_THRESHOLD", "0.2"))
RULE1_UNCERTAINTY_THRESHOLD = float(os.getenv("RULE1_UNCERTAINTY_THRESHOLD", "0.25"))

# --- Band Thresholds (per BRD) ---
P2_MAX = 0.39
P1_MAX = 0.74
# P0: above P1_MAX

# --- Agent Weights for Ensemble ---
AGENT_WEIGHTS = {
    "agent_1": 0.7,
    "agent_2": 1.0,
    "agent_3": 0.8,
    "agent_4": 0.8,
    "agent_5": 0.6,
    "agent_6": 0.7,
    "agent_7": 0.7,
    "agent_8": 0.5,
    "agent_9": 0.9,
    "agent_10": 0.9,
    "agent_11": 0.6,
    "agent_13": 0.85,
}

# Human-readable agent names (used by UI + SHAP panel).
AGENT_NAMES = {
    "agent_1": "Metadata Analysis",
    "agent_2": "Image Forensics",
    "agent_3": "Font & Layout",
    "agent_4": "Template Matching",
    "agent_5": "Duplicate / Similarity",
    "agent_6": "Temporal Consistency",
    "agent_7": "NER / Semantic",
    "agent_8": "QR / Barcode",
    "agent_9": "Novel Fraud",
    "agent_10": "Cross-OCR Disagreement",
    "agent_11": "Adversarial Robustness",
}

# Known PDF editor producer/creator strings (Agent 1). Extend as needed.
KNOWN_PDF_EDITORS = [
    "ilovepdf",
    "smallpdf",
    "adobe acrobat",
    "acrobat",
    "pdfescape",
    "foxit",
    "nitro",
    "pdf-xchange",
    "sejda",
    "soda pdf",
    "pdffiller",
    "libreoffice",  # often used to re-author PDFs
    "microsoft: print to pdf",
    "wkhtmltopdf",
    "pdf24",
    "cam scanner",
    "camscanner",
]

# Known legitimate entity names (Agent 7 misspelling detection).
KNOWN_ENTITIES = [
    "Infosys",
    "Tata Consultancy Services",
    "Wipro",
    "Cognizant",
    "Accenture",
    "Capgemini",
    "HCL Technologies",
    "Tech Mahindra",
    "IBM",
    "Microsoft",
    "Google",
    "Amazon",
    "Deloitte",
]

# Document types the system recognises (also used for the UI dropdown).
DOCUMENT_TYPES = [
    "payslip",
    "experience_letter",
    "offer_letter",
    "form16",
    "education_certificate",
    "other",
]

# Field criticality weights for cross-OCR disagreement (Agent 10).
FIELD_CRITICALITY = {
    "designation": 1.0,
    "employer": 1.0,
    "date": 0.9,
    "amount": 0.9,
    "salary": 0.9,
    "name": 0.7,
}
DEFAULT_FIELD_CRITICALITY = 0.3

# Escalation rule thresholds (Step 7).
RULE2_DISAGREEMENT_DELTA = float(os.getenv("RULE2_DISAGREEMENT_DELTA", "0.4"))
RULE2_MIN_AGENTS = int(os.getenv("RULE2_MIN_AGENTS", "2"))

# Upload limits (Step 1).
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(50 * 1024 * 1024)))  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
