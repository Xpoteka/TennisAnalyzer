# Validation results

This folder holds the validation reports that spec section 12 requires. Each one is added when its milestone is done:

- `M2_contacts.md`: precision and recall of contact detection against `labels/contacts_<session>.csv`, within ±40 ms
- `M5_classifier.md`: confusion matrix and accuracy against `labels/strokes_<session>.csv`
- `M8_labels.md`: match rate for spoken label words
