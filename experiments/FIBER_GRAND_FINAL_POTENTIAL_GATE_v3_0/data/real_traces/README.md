# Optional measured post-front-end traces

Place CSV files here before running the gate to activate the measured-trace audit.
Each row represents one received frame and must contain:

- `n`: transmitted blocklength;
- `deleted_position`: integer in `[0,n-1]`;
- `substitution_weight`: nonnegative integer on the `n-1` surviving decisions;
- optionally `source`, `snr_db`, and `frame_id`.

The default package contains no measured trace and never fabricates one.  In its absence, the gate may confirm high scientific potential but must retain `REAL_TRACE_REQUIRED` in the classification.
