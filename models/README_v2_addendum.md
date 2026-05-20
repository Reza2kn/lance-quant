## v2 update — group_size 128 → 64

Re-quantized with `--group_size 64` (was 128 in v1). Same AWQ calibration data,
same scale-fusion recipe. Storage: ~6.15 GB (was 6.02 GB); the +2.5% size is
the cost of 2× more per-group scales.

**Quality jumped substantially on Lance's bundled x2t_image bench**:

| variant | exact-match | char similarity | difflib ratio | word Jaccard |
|---|---|---|---|---|
| v1 (group_size=128) | 33.3 % | 60.4 % | 53.7 % | 55.3 % |
| **v2 (group_size=64)** | **50.0 %** | **69.8 %** | **62.1 %** | **66.3 %** |

The biggest win is on **case 4** ("$ spent on promotional events 1998") — v1
hallucinated entities ("Scott Levin and his family") around the correct number;
v2 produces the **exact baseline output**:

> "According to the data from the proprietary market research, the total amount
>  spent on the promotional meetings and events during 1998 was approximately
>  $1.3 billion."

The smaller group size reduces the per-group outlier impact in `o_proj` and
`down_proj` (the linears we can't fuse AWQ scales into), which were responsible
for the long-form generation drift.

Recipe & eval at: https://github.com/Reza2kn/lance-quant#v2-group_size-64
