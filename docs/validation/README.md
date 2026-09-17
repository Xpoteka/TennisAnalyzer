# Validation results

This folder holds the validation reports that spec section 12 requires. Each one is added when its milestone is done:

- `M2_contacts.md`: precision and recall of contact detection against `labels/contacts_<session>.csv`, within ±40 ms
- `M5_classifier.md`: confusion matrix and accuracy against `labels/strokes_<session>.csv`
- `M8_labels.md`: match rate for spoken label words

The tools write these files themselves; pass `--report docs/validation/<name>.md`:

```bash
uv run tennis eval-contacts   <id> --labels ... --report docs/validation/M2_contacts.md
uv run tennis eval-classifier <id> --labels ... --report docs/validation/M5_classifier.md
uv run tennis eval-labels     <id> --labels ... --report docs/validation/M8_labels.md
```

`M5_classifier.md` and `M8_labels.md` are still missing: both need a session that has been labelled by hand. See `docs/HANDOFF.md` §6.
