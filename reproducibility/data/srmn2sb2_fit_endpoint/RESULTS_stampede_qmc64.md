# Initialization and Normalization Comparison

The reported result is the center005 / per-scan endpoint below. All four
calculations used official pySpinW Python intensities, 64 QMC resolution
samples per point, a single in-plane spin orientation, fixed scan-local
backgrounds, and the documented legacy resolution scenario.

| Start | Normalization | nllf | Reduced chi-squared | SJ1 | SJ2 | SJc | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all102 | per scan | 750.4483 | 1.937 | 12.0520 | 1.8656 | 0.3159 | 0.0120 |
| all102 | shared zone/config | 906.8844 | 2.296 | 10.5957 | 1.6198 | 0.4396 | 0.0065 |
| center005 | per scan | 683.5699 | 1.764 | 10.3245 | 1.6113 | 0.3699 | 0.0667 |
| center005 | shared zone/config | 786.3537 | 1.991 | 8.2664 | 1.1538 | 0.4020 | 0.0709 |

Energies are in meV. The normalization policies have different parameter
counts. This comparison measures sensitivity to initialization and
normalization; its spread is not a parameter uncertainty interval.

The complete reported endpoint, 20 fitted profiles, and resolution samples
are supplied in this directory. The saved export reproduces
nllf = 683.569864454994 and reduced chi-squared = 1.7640512631096619.
