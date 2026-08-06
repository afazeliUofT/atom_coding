# Optional measured post-front-end traces

The flagship theory/algorithm gate does not require a measured trace. A real-system or field-defining-impact claim does.

Place one or more CSV files in this directory before running the package. Required columns:

- `frame_id`: unique frame identifier;
- `blocklength`: transmitted coded block length;
- `edit_count`: number of post-front-end insertion/deletion/duplication/slip events represented as edits;
- `substitution_count`: number of survivor hard-decision substitutions.

Recommended additional columns are `event_type`, `edit_positions`, `snr_db`, `receiver_configuration`, `source_capture`, and `split` (`train` or `test`). The data must come from a measured receiver front end or a recorded hardware/software modem trace; synthetic Monte Carlo output must not be labelled measured.
