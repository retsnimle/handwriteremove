# Agent Operating Guidelines: `handwriteremove`

This document is the **single source of truth for autonomous AI agents** (Antigravity, Claude Code, Cursor, Devin, etc.) operating in or extending this repository. When prompted to inspect, run inference, retrain, or debug this project, strictly follow the architectural contracts and command conventions defined herein.

---

## 1. Architecture & System Contract

The codebase implements a **Dual-Scale Coarse-to-Fine 300 DPI Pipeline** engineered specifically for scanned educational examination papers (A4/B4):

```
Raw Input PDF
   │
   ├─► Coarse Stream @ 150 DPI
   │     ├─► ResNet34-UNet (weights/best_exam_remover.pth) ──► P_hw [0, 1]
   │     └─► DocLayout-YOLO (weights/doclayout_yolo...pt)   ──► M_diag [0, 1]
   │
   └─► Fine Stream @ 300 DPI
         │
         ▼
     5-Channel Fusion: [RGB (3ch), Bilinear_2x(P_hw), Bilinear_2x(M_diag)]
         │
         ▼
     DocUnfoldV2 (weights/best_docunfold_v2_dual_300dpi.pth)
     7-Stage Deep Gradient Unfolding Network (DGUNet)
     - Sliding window: 512x512 patches, 384 stride (64*6 integer grid)
     - Full geometric padding to 64px multiple
         │
         ▼
Cleaned 300 DPI Restored PDF
```

### Channel Specification for `DocUnfoldV2`
Every patch fed to `DocUnfoldV2` has tensor shape `[B, 5, H, W]`:
- **Channels 0..2**: RGB image normalized to $[0, 1]$.
- **Channel 3**: Coarse handwriting probability map $P_{hw} \in [0, 1]$ (bilinear 2x upsampled from 150 DPI).
- **Channel 4**: Diagram & layout protection mask $M_{diag} \in [0, 1]$ (1.0 inside figures, tables, formulas, and geometric axes; 0.0 elsewhere).

---

## 2. Deterministic Agent Action Playbook

### A. Verify System Integrity (Run First)
Always execute the automated self-diagnostic suite before and after making any modifications:
```bash
uv run python -m tools.self_diagnostic
```
**Expected Output**: All 12 checks return `[PASS]`, exit code `0`. Tests 6 canvas aspect ratios, 5-channel tensor dimensions, 4-edge pixel black border algorithm, weight file sizes, and loss stability.

### B. Production Inference (Single File or Batch Folder)
Execute handwriting removal using `tools/colab_batch_worker_300dpi.py`:
```bash
# Single PDF exam:
uv run python -m tools.colab_batch_worker_300dpi --input_dir /path/to/exam.pdf --output_dir /path/to/output_dir

# Entire folder of PDF exams:
uv run python -m tools.colab_batch_worker_300dpi --input_dir /path/to/dirty_pdfs/ --output_dir /path/to/cleaned_pdfs/

# Optional custom device & weight paths:
uv run python -m tools.colab_batch_worker_300dpi --input_dir /path/to/exam.pdf --output_dir /path/to/output_dir --device cuda
```

### C. Cloud Training & Fine-Tuning Orchestration
To fine-tune `DocUnfoldV2` on fresh assets via Google Colab:
```bash
uv run python tools/orchestrate_colab_300dpi.py --session <colab_session_id>
```
This automatically packages assets into 35MB chunks, uploads them to Colab, executes `train_colab_docunfold_dual_300dpi.py` (1,500 steps, AdamW, CosineAnnealingLR, hard-patch sampling), validates weights, and downloads the trained checkpoint to `weights/best_docunfold_v2_dual_300dpi.pth`.

### D. Handwriting Asset Extraction
To re-extract or append new 300 DPI handwriting layers and stamps from annotated exams:
```bash
uv run python -m tools.extract_handwriting_assets_300dpi
```

---

## 3. Strict Invariants & Anti-Patterns

Any agent operating in this repository **MUST NEVER violate** the following rules:

1. **Native 300 DPI Output Invariant**:
   - The final restored document MUST remain at 300 DPI. Never downsample the output PDF to 150 DPI to save compute.
2. **Geometric 64px Grid Alignment Invariant**:
   - Any padded dimension $\text{dim}_{\text{pad}}$ MUST satisfy:
     $$\text{dim}_{\text{pad}} = 64 \times \left\lceil \frac{\text{dim}}{64} \right\rceil$$
   - This guarantees that stride ($384 = 64 \times 6$) and patch size ($512 = 64 \times 8$) tile the padded canvas with 100% boundary coverage, eliminating perimeter dead zones.
3. **Four-Edge Pixel Quality SOP**:
   - Outer perimeter pixels (first and last columns/rows) must be verified: any pixel with RGB brightness $< 30$ is flagged as black edge.
   - The edge black ratio must remain strictly $< 0.05\%$ (target: $0.00\%$).
4. **Prohibited Implementations (Anti-Patterns)**:
   - **DO NOT** re-introduce LaMa inpainting, OpenCV contour clustering, or threshold-based bounding-box erosion.
   - **DO NOT** revert `DocUnfold` to 3-channel blind unrolling. The 5-channel conditional injection is mandatory.
   - **DO NOT** add dirty/annotated exam PDFs into `scanned_pairs/`. That directory is reserved exclusively for clean ground-truth baseline PDFs matching `assets/scanned_catalog.json`.

---

## 4. Directory & File Organization Contract

```
.
├── AGENTS.md                                 # This agent operating manual
├── README.md                                 # Human & agent overview
├── pyproject.toml / requirements.txt         # Dependencies (uv / pip)
├── docs/
│   └── PRODUCTION_AND_TRAINING_PIPELINE.md   # Complete mathematical & pipeline specification
├── models/
│   ├── DocUnfold.py                          # DGUNet unrolling backbone (FP32 numerical protection)
│   ├── DocUnfoldV2.py                        # 5-Channel conditional deep unrolling model
│   └── __init__.py
├── tools/
│   ├── colab_batch_worker_300dpi.py          # Dual-scale inference worker (CLI: single/batch)
│   ├── train_colab_docunfold_dual_300dpi.py  # 300 DPI dual-scale fine-tuner
│   ├── orchestrate_colab_300dpi.py           # Colab remote lifecycle orchestrator
│   ├── extract_handwriting_assets_300dpi.py  # Asset extraction engine
│   └── self_diagnostic.py                    # 12-check automated verification suite
├── assets/
│   ├── hw_layers_300dpi/                     # 54 real handwriting layer masks
│   ├── hw_marks_300dpi/                      # 429 individual marks/stamps + marks_meta.json
│   ├── scanned_catalog.json                  # 166-page ground-truth clean page index
│   └── exam_anchors_catalog_points.json      # Optical anchor catalog
├── scanned_pairs/                            # 10 clean high-resolution baseline exam PDFs
└── weights/                                  # 4 core weights (.gitignored)
    ├── best_exam_remover.pth                 # ResNet34-UNet (93.4 MB)
    ├── doclayout_yolo_docstructbench...pt    # DocLayout-YOLOv10 (38.8 MB)
    ├── best_docunfold_v2_dual_300dpi.pth     # DocUnfoldV2 production inference (46.7 MB)
    └── best_docunfold_v2_5ch.pth             # DocUnfoldV2 training base checkpoint (46.6 MB)
```
