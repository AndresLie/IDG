# V4.2f Reproducibility Rerun

## Result

The cache-hit rerun reproduced the complete V4.2f result byte-for-byte.

| Artifact | Initial SHA-256 | Rerun SHA-256 | Match |
| --- | --- | --- | :---: |
| Candidate cache | `21184761905958b5a3a572b51391dafaa029e306358fc9c34e6aac816756598b` | `21184761905958b5a3a572b51391dafaa029e306358fc9c34e6aac816756598b` | Yes |
| Candidate model | `c5a6c9de23a7720c79ec4740bdfab70aa897b33203eadd966f78ce48f5d0a2ad` | `c5a6c9de23a7720c79ec4740bdfab70aa897b33203eadd966f78ce48f5d0a2ad` | Yes |
| Metrics JSON | `06717691fc2ec0bd94866fbbbf8b17e01c656f5661a5cc9b8532ca6fd3b0e4c8` | `06717691fc2ec0bd94866fbbbf8b17e01c656f5661a5cc9b8532ca6fd3b0e4c8` | Yes |
| Markdown report | `146e27be800264f96448a3a091d843663f1d14c807b7431ff44d3499a29df539` | `146e27be800264f96448a3a091d843663f1d14c807b7431ff44d3499a29df539` | Yes |

The rerun also reproduced all provenance identities:

- architecture core fingerprint;
- package-code fingerprint and Python file count;
- models fingerprint;
- config file and effective-config fingerprints;
- split-spec fingerprint;
- candidate-row cache fingerprint.

## Runtime

| Run | Cache state | Elapsed time |
| --- | --- | ---: |
| Initial | miss | `2,502.87 s` |
| Rerun | hit | `2,072.77 s` |

The cache-hit rerun is `430.10 s` faster (`17.18%`, `1.21x`). The remaining
runtime is dominated by nested source-jackknife and all-pairs list-ranking model
fitting rather than candidate reconstruction.

## Decision

Reproducibility passes. The scientific promotion decision does not change:
V4.2f remains default-off because the primary VisA oracle endpoint and selected
deployment guard fail.
