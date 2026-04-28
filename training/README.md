# Training

Retraining workflows. **Not yet active** — populated once a meaningful labeled dataset
has accumulated through the labeling workflow.

Planned components:

- Training data assembly from Label Studio exports
- Train/validation split (test set is separate, see `../test_set/`)
- DETR fine-tuning pipeline
- Per-version evaluation against the frozen test set
- Model version log (`../docs/model_versions.md`)
