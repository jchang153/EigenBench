# Omitted samples in published runs

When `scripts/upload_results.py` stages an Inspect run, it compares the judgment
samples in the published log with `evaluations.jsonl`. It writes
`collection_report.json` and references it through
`meta.inspect.collection_report_file` for ValueArena's coverage view.

The report includes logged/exported/omitted counts and each omitted sample's
scenario, judge, model names, and error message. A missing export row without a
sample error is distinguished from a failed sample. Stack traces are not included.

The report covers only judgment samples present in the named log. It does not
certify unstarted samples, other logs, or overall plan completeness. Repeated
judgment keys such as multiple epochs are ambiguous; no verified report is
produced in that case. A report-generation error is surfaced as an upload warning
and does not become a zero-failure count. Existing publications need restaging and
re-uploading to gain the report; old exports alone cannot reconstruct errors.
