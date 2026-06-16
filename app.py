from pathlib import Path
import json
import re
import uuid
import hashlib
import importlib
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from flask import Flask, render_template, request
from PIL import Image
from torchvision import models

try:
    faiss = importlib.import_module("faiss")
except Exception:
    faiss = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "siamese_best.pt"
DB_DIR = BASE_DIR / "fingerprint_db"
SOURCES_FILE = BASE_DIR / "db_sources.json"
EMBED_INDEX_FILE = BASE_DIR / "embedding_index.npz"
EMBED_META_FILE = BASE_DIR / "embedding_index_meta.json"
DEVICE = "cpu"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INDEX_CACHE = {
    "signature": "",
    "embeddings": None,
    "person_ids": [],
    "files": [],
    "sources": [],
    "paths": [],
    "faiss_index": None,
}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.mean((2, 3))
        scaled = self.fc(weights)
        return x * scaled.unsqueeze(-1).unsqueeze(-1)


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.se(features)
        features = self.pool(features)
        embedding = self.head(features)
        return nn.functional.normalize(embedding, dim=1)


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        d = self.dist(e1, e2)
        return e1, e2, d


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)


def load_model(model_path: Path) -> SiameseNet:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = SiameseNet().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


MODEL = load_model(MODEL_PATH)
DB_DIR.mkdir(parents=True, exist_ok=True)
if not SOURCES_FILE.exists():
    SOURCES_FILE.write_text("[]", encoding="utf-8")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def preprocess_from_upload(file_obj) -> Image.Image:
    return Image.open(file_obj.stream).convert("L")


def embed_image(image: Image.Image) -> torch.Tensor:
    with torch.no_grad():
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        return MODEL.encoder(tensor)


def image_embedding_numpy(image: Image.Image) -> np.ndarray:
    emb = embed_image(image).detach().cpu().numpy()[0].astype("float32")
    return emb


def sanitize_person_id(raw_value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_value.strip())
    return cleaned[:80]


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def load_external_sources() -> list:
    try:
        payload = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(Path(p)) for p in payload]
    except Exception:
        pass
    return []


def save_external_sources(sources: list) -> None:
    dedup = sorted(set(str(Path(p)) for p in sources))
    SOURCES_FILE.write_text(json.dumps(dedup, indent=2), encoding="utf-8")


def add_external_source(path_str: str) -> bool:
    src = str(Path(path_str).expanduser().resolve())
    existing = load_external_sources()
    if src in existing:
        return False
    existing.append(src)
    save_external_sources(existing)
    return True


def extract_person_id_from_path(file_path: Path) -> str:
    match = re.match(r"(\d+)__", file_path.name)
    if match:
        return match.group(1)
    if file_path.parent.name.lower() in {"real", "altered", "easy", "medium", "hard"}:
        return file_path.parent.parent.name if file_path.parent.parent else file_path.parent.name
    return file_path.parent.name


def list_enrolled_people() -> list:
    return sorted([p.name for p in DB_DIR.iterdir() if p.is_dir()])


def list_enrolled_images() -> list:
    rows = []
    for person_dir in sorted([p for p in DB_DIR.iterdir() if p.is_dir()]):
        for file in sorted(person_dir.glob("*")):
            if is_image_path(file):
                rows.append({"person_id": person_dir.name, "file": file.name})
    return rows


def list_external_images() -> list:
    rows = []
    for source_root in load_external_sources():
        root = Path(source_root)
        if not root.exists() or not root.is_dir():
            continue
        for file in root.rglob("*"):
            if not file.is_file() or not is_image_path(file):
                continue
            pid = extract_person_id_from_path(file)
            rows.append(
                {
                    "person_id": pid,
                    "file": file.name,
                    "source": str(root),
                    "path": str(file),
                }
            )
    return rows


def save_enrollment_images(person_id: str, files: list) -> int:
    person_dir = DB_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for file in files:
        if not file or not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower() or ".png"
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        target = person_dir / safe_name
        image = preprocess_from_upload(file)
        image.save(target)
        saved += 1
    return saved


def build_database_records() -> list:
    records = []
    for person_dir in sorted([p for p in DB_DIR.iterdir() if p.is_dir()]):
        for file in sorted(person_dir.glob("*")):
            if not is_image_path(file):
                continue
            records.append(
                {
                    "person_id": person_dir.name,
                    "file": file.name,
                    "path": file,
                    "source": "local",
                }
            )

    for ext in list_external_images():
        records.append(
            {
                "person_id": ext["person_id"],
                "file": ext["file"],
                "path": Path(ext["path"]),
                "source": "external",
            }
        )

    return records


def summarize_database() -> dict:
    records = build_database_records()
    people = sorted({r["person_id"] for r in records})
    index_info = load_embedding_index_info()
    return {
        "total_people": len(people),
        "total_fingerprints": len(records),
        "external_sources": load_external_sources(),
        "index_size": index_info.get("size", 0),
        "index_signature": index_info.get("signature", ""),
        "index_ready": index_info.get("ready", False),
        "index_backend": index_info.get("backend", "numpy"),
    }


def clear_index_cache() -> None:
    INDEX_CACHE["signature"] = ""
    INDEX_CACHE["embeddings"] = None
    INDEX_CACHE["person_ids"] = []
    INDEX_CACHE["files"] = []
    INDEX_CACHE["sources"] = []
    INDEX_CACHE["paths"] = []
    INDEX_CACHE["faiss_index"] = None


def clear_local_database() -> int:
    deleted = 0
    for child in DB_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            deleted += 1
        elif child.is_file():
            child.unlink(missing_ok=True)
            deleted += 1
    return deleted


def clear_external_sources() -> None:
    save_external_sources([])


def clear_embedding_artifacts() -> int:
    removed = 0
    for fp in [EMBED_INDEX_FILE, EMBED_META_FILE]:
        if fp.exists():
            fp.unlink(missing_ok=True)
            removed += 1
    clear_index_cache()
    return removed


def database_signature(records: list) -> str:
    if not records:
        return "empty"
    payload = []
    for rec in sorted(records, key=lambda r: str(r["path"])):
        try:
            st = rec["path"].stat()
            payload.append(f"{rec['person_id']}|{rec['file']}|{rec['source']}|{rec['path']}|{int(st.st_mtime)}|{st.st_size}")
        except Exception:
            continue
    joined = "\n".join(payload)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_embedding_index_info() -> dict:
    if not EMBED_META_FILE.exists() or not EMBED_INDEX_FILE.exists():
        return {"ready": False, "size": 0, "signature": "", "backend": "numpy"}
    try:
        meta = json.loads(EMBED_META_FILE.read_text(encoding="utf-8"))
        return {
            "ready": True,
            "size": int(meta.get("size", 0)),
            "signature": str(meta.get("signature", "")),
            "backend": "faiss-ip" if faiss is not None else "numpy-ip",
        }
    except Exception:
        return {"ready": False, "size": 0, "signature": "", "backend": "numpy"}


def write_embedding_index(records: list) -> int:
    vectors = []
    person_ids = []
    files = []
    sources = []
    paths = []

    for rec in records:
        try:
            image = Image.open(rec["path"]).convert("L")
            emb = image_embedding_numpy(image)
            vectors.append(emb)
            person_ids.append(rec["person_id"])
            files.append(rec["file"])
            sources.append(rec["source"])
            paths.append(str(rec["path"]))
        except Exception:
            continue

    if not vectors:
        np.savez_compressed(
            EMBED_INDEX_FILE,
            embeddings=np.zeros((0, 128), dtype="float32"),
            person_ids=np.array([], dtype=object),
            files=np.array([], dtype=object),
            sources=np.array([], dtype=object),
            paths=np.array([], dtype=object),
        )
        EMBED_META_FILE.write_text(
            json.dumps({"size": 0, "signature": "empty"}, indent=2),
            encoding="utf-8",
        )
        return 0

    embeddings = np.vstack(vectors).astype("float32")
    np.savez_compressed(
        EMBED_INDEX_FILE,
        embeddings=embeddings,
        person_ids=np.array(person_ids, dtype=object),
        files=np.array(files, dtype=object),
        sources=np.array(sources, dtype=object),
        paths=np.array(paths, dtype=object),
    )

    signature = database_signature(records)
    EMBED_META_FILE.write_text(
        json.dumps({"size": int(embeddings.shape[0]), "signature": signature}, indent=2),
        encoding="utf-8",
    )
    clear_index_cache()
    return int(embeddings.shape[0])


def ensure_embedding_index(force_rebuild: bool = False) -> dict:
    records = build_database_records()
    current_signature = database_signature(records)
    info = load_embedding_index_info()

    if force_rebuild or (not info["ready"]) or info["signature"] != current_signature:
        size = write_embedding_index(records)
        clear_index_cache()
        return {
            "rebuilt": True,
            "size": size,
            "signature": current_signature,
        }
    return {
        "rebuilt": False,
        "size": info["size"],
        "signature": info["signature"],
    }


def read_embedding_index() -> dict:
    blob = np.load(EMBED_INDEX_FILE, allow_pickle=True)
    return {
        "embeddings": blob["embeddings"].astype("float32"),
        "person_ids": blob["person_ids"].tolist(),
        "files": blob["files"].tolist(),
        "sources": blob["sources"].tolist(),
        "paths": blob["paths"].tolist(),
    }


def get_cached_embedding_index() -> dict:
    info = load_embedding_index_info()
    if not info.get("ready"):
        return {
            "embeddings": np.zeros((0, 128), dtype="float32"),
            "person_ids": [],
            "files": [],
            "sources": [],
            "paths": [],
            "faiss_index": None,
        }

    if INDEX_CACHE["signature"] == info["signature"] and INDEX_CACHE["embeddings"] is not None:
        return {
            "embeddings": INDEX_CACHE["embeddings"],
            "person_ids": INDEX_CACHE["person_ids"],
            "files": INDEX_CACHE["files"],
            "sources": INDEX_CACHE["sources"],
            "paths": INDEX_CACHE["paths"],
            "faiss_index": INDEX_CACHE["faiss_index"],
        }

    blob = read_embedding_index()
    embeddings = blob["embeddings"]
    faiss_index = None
    if faiss is not None and embeddings.shape[0] > 0:
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(embeddings)

    INDEX_CACHE["signature"] = info["signature"]
    INDEX_CACHE["embeddings"] = embeddings
    INDEX_CACHE["person_ids"] = blob["person_ids"]
    INDEX_CACHE["files"] = blob["files"]
    INDEX_CACHE["sources"] = blob["sources"]
    INDEX_CACHE["paths"] = blob["paths"]
    INDEX_CACHE["faiss_index"] = faiss_index

    return {
        "embeddings": embeddings,
        "person_ids": blob["person_ids"],
        "files": blob["files"],
        "sources": blob["sources"],
        "paths": blob["paths"],
        "faiss_index": faiss_index,
    }


def ip_search(query_vec: np.ndarray, embeddings: np.ndarray, top_k: int = 200, faiss_index=None):
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    k = min(top_k, embeddings.shape[0])
    if faiss_index is not None:
        sims, idxs = faiss_index.search(query_vec[None, :].astype("float32"), k)
        return idxs[0], sims[0]

    sims = embeddings @ query_vec
    order = np.argsort(-sims)[:k]
    return order.astype(np.int64), sims[order].astype("float32")


def normalized_ip_to_score(ip_value: float) -> float:
    clipped = max(-1.0, min(1.0, float(ip_value)))
    return round((clipped + 1.0) / 2.0, 4)


def search_query_against_database(query_image: Image.Image, threshold: float, min_margin: float):
    ensure_embedding_index(force_rebuild=False)
    index_blob = get_cached_embedding_index()
    embeddings = index_blob["embeddings"]
    if embeddings.shape[0] == 0:
        return None

    query_vec = image_embedding_numpy(query_image)
    idxs, ips = ip_search(query_vec, embeddings, top_k=200, faiss_index=index_blob["faiss_index"])

    fingerprint_scores = []
    for rank_idx, db_idx in enumerate(idxs):
        ip_value = float(ips[rank_idx])
        score = normalized_ip_to_score(ip_value)
        fingerprint_scores.append(
            {
                "person_id": index_blob["person_ids"][int(db_idx)],
                "file": index_blob["files"][int(db_idx)],
                "score": score,
                "source": index_blob["sources"][int(db_idx)],
                "ip": round(ip_value, 4),
            }
        )

    if not fingerprint_scores:
        return None

    ranked_fingerprints = sorted(fingerprint_scores, key=lambda row: row["score"], reverse=True)
    person_map = {}
    for row in ranked_fingerprints:
        pid = row["person_id"]
        person_map.setdefault(pid, []).append(row)

    person_ranked = []
    for pid, matches in person_map.items():
        best = matches[0]
        top_scores = [m["score"] for m in matches[:3]]
        avg_top = float(np.mean(top_scores))
        person_score = round((0.75 * best["score"]) + (0.25 * avg_top), 4)
        person_ranked.append(
            {
                "person_id": pid,
                "person_score": person_score,
                "best_fingerprint": best["file"],
                "best_fingerprint_score": best["score"],
                "matched_fingerprints": len(matches),
            }
        )

    person_ranked.sort(key=lambda row: row["person_score"], reverse=True)
    top_person = person_ranked[0]
    second_score = person_ranked[1]["person_score"] if len(person_ranked) > 1 else 0.0
    margin = round(top_person["person_score"] - second_score, 4)
    is_known = top_person["person_score"] >= threshold and margin >= min_margin

    confidence_pct = round(top_person["person_score"] * 100, 2)
    if is_known:
        decision = "KNOWN PERSON FOUND"
    elif top_person["person_score"] >= threshold and margin < min_margin:
        decision = "AMBIGUOUS MATCH"
    else:
        decision = "UNKNOWN / NOT IN DATABASE"

    return {
        "threshold": threshold,
        "min_margin": min_margin,
        "decision": decision,
        "best_person": top_person["person_id"],
        "best_file": top_person["best_fingerprint"],
        "best_score": top_person["best_fingerprint_score"],
        "person_score": top_person["person_score"],
        "margin": margin,
        "confidence_pct": confidence_pct,
        "top_persons": person_ranked[:5],
        "top_fingerprints": ranked_fingerprints[:5],
        "search_backend": "faiss-ip" if faiss is not None else "numpy-ip",
    }


def quality_warning(image: Image.Image) -> str:
    arr = np.array(image, dtype=np.float32)
    std = float(arr.std())
    dynamic_range = float(arr.max() - arr.min())
    if std < 18 or dynamic_range < 60:
        return "Possible low-quality/smudged fingerprint (low contrast)."
    return "Fingerprint quality looks usable."


@app.route("/", methods=["GET", "POST"])
def index():
    enroll_result = None
    source_result = None
    rebuild_result = None
    maintenance_result = None
    search_result = None
    error = None
    people = list_enrolled_people()
    images = list_enrolled_images()
    summary = summarize_database()

    if request.method == "POST":
        try:
            action = request.form.get("action")

            if action == "enroll":
                person_id = sanitize_person_id(request.form.get("person_id", ""))
                gallery_files = request.files.getlist("gallery")
                valid_gallery = [file for file in gallery_files if file and file.filename]

                if not person_id:
                    raise ValueError("Please enter a person ID (example: person_01).")
                if len(valid_gallery) < 1:
                    raise ValueError("Please upload at least 1 fingerprint for enrollment.")
                if len(valid_gallery) > 10:
                    raise ValueError("Please upload at most 10 fingerprints at a time.")

                saved_count = save_enrollment_images(person_id, valid_gallery)
                enroll_result = {
                    "person_id": person_id,
                    "saved_count": saved_count,
                }
                rebuild_result = ensure_embedding_index(force_rebuild=True)

            elif action == "index_folder":
                folder_path = request.form.get("dataset_path", "").strip()
                if not folder_path:
                    raise ValueError("Please provide a dataset folder path.")

                resolved = Path(folder_path).expanduser().resolve()
                if not resolved.exists() or not resolved.is_dir():
                    raise ValueError("Dataset folder path not found.")

                added = add_external_source(str(resolved))
                images_count = 0
                for file in resolved.rglob("*"):
                    if file.is_file() and is_image_path(file):
                        images_count += 1

                source_result = {
                    "path": str(resolved),
                    "added": added,
                    "images_count": images_count,
                }
                rebuild_result = ensure_embedding_index(force_rebuild=True)

            elif action == "rebuild_index":
                rebuild_result = ensure_embedding_index(force_rebuild=True)

            elif action == "clear_embeddings":
                removed = clear_embedding_artifacts()
                maintenance_result = {"message": f"Embedding artifacts cleared ({removed} files removed)."}

            elif action == "clear_sources":
                clear_external_sources()
                clear_embedding_artifacts()
                maintenance_result = {"message": "External indexed sources cleared."}

            elif action == "clear_local_db":
                removed_items = clear_local_database()
                clear_embedding_artifacts()
                maintenance_result = {"message": f"Local fingerprint DB cleared ({removed_items} folders/files removed)."}

            elif action == "clear_all":
                removed_items = clear_local_database()
                clear_external_sources()
                removed_artifacts = clear_embedding_artifacts()
                maintenance_result = {
                    "message": f"Clean reset complete. Local DB removed: {removed_items}, embedding files removed: {removed_artifacts}."
                }

            elif action == "search":
                threshold = float(request.form.get("threshold", "0.72"))
                min_margin = float(request.form.get("min_margin", "0.02"))
                query_file = request.files.get("query")

                if not query_file or not query_file.filename:
                    raise ValueError("Please upload one query fingerprint image for testing.")

                query_image = preprocess_from_upload(query_file)
                quality_note = quality_warning(query_image)
                search_result = search_query_against_database(query_image, threshold, min_margin)
                if search_result is None:
                    raise ValueError("No enrolled fingerprints found. Enroll first.")
                search_result["quality_note"] = quality_note
            else:
                raise ValueError("Invalid action.")

            people = list_enrolled_people()
            images = list_enrolled_images()
            summary = summarize_database()
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        enroll_result=enroll_result,
        source_result=source_result,
        rebuild_result=rebuild_result,
        maintenance_result=maintenance_result,
        search_result=search_result,
        people=people,
        images=images,
        summary=summary,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
