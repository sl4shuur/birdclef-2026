# Submission

- `build.build_submission(predictions, template)` aligns explicit IDs and species to a template.
- `format.validate_submission(submission, template)` checks exact order, IDs and probabilities.
- `write.write_submission(submission, path, template)` validates, then writes the assembled CSV.

No clipping or filling. Hidden inference templates must use actual soundscape IDs; visible sample rows are placeholders. See the root README for an example.
