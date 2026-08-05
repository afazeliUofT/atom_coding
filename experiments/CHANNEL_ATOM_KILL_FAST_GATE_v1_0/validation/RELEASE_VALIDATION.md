# Release Validation

The release candidate was validated from the package source with the `standard` profile.

- Unit and implementation tests: **13 passed**.
- Exactness audit: **passed**.
- Standard-profile wall time in the release environment: approximately **18 seconds**.
- Maximum resident memory in the release environment: approximately **219 MB**.
- Reference scientific classification: `PIVOT_TO_REVISED_TRACK`.
- Authoritative reference command: `STOP_ORIGINAL_BROAD_CHANNEL_ATOM_PROGRAM`.

The reference result is included to expose the complete release behavior before the user's independent WSL reproduction. It is not a universal impossibility theorem. The runtime `results/` directory is intentionally reset so the launcher produces a clean, environment-specific result set.

The PDF was inspected, preflighted, and rendered page-by-page. Representative page renders are retained under `pdf_validation/representative_pages/`.
