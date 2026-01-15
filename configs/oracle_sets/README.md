# Oracle fixed-set diagnostics

This directory contains deterministic **oracle-slice state sets** used for fixed-set oracle diagnostics.

## File format

Each set is stored as `configs/oracle_sets/<id>.json` with a JSON array of objects:

```json
{
  "avail_mask": 32767,
  "upper_total_cap": 0,
  "dice_sorted": [1, 2, 3, 4, 5],
  "rerolls_left": 2
}
```

## Generation

Sets are intended to be generated once (deterministic, random-but-stratified) and committed.

