# Open-source release checklist

- [x] Standalone `PicoRet` class; no framework-specific registry required.
- [x] Strict loading of the released 817-parameter checkpoints.
- [x] Full-resolution inference and PSNR/SSIM/DeltaE00 evaluation.
- [x] Single-GPU 100K training recipe with internal-validation selection.
- [x] Fixed LOL-v1 and LOL-v2 Synthetic train/validation manifests.
- [x] Public paired and no-reference dataset download helper.
- [x] Dataset files excluded from the repository.
- [x] Checkpoint hashes included.
- [x] Architecture smoke test passes.
- [x] LOL-v1 and LOL-v2 Synthetic released weights independently reproduced.
- [x] Project-facing package, class, commands, configs and weights use PicoRet naming.
- [x] License retained in `LICENSE`; analytical-prior provenance is documented
      next to the corresponding implementation in `picoret/model.py`.

Before publishing, add the final paper citation, author list, contact address,
repository URL, and an optional model-download mirror.
