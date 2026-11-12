# SrMn2Sb2 Spin-Wave Case

The paper reports a joint resolution-convolved intensity fit to 20 BT-7
scans from experiment 24858, with 799 retained points. Each scan has its own
multiplicative intensity scale and a fixed background from its preceding
scan-local fit. The reported endpoint has reduced chi-squared 1.764 for
775 nominal degrees of freedom.

## Reproduce the Reported Result

- [Figure replay](../../reproducibility/README.md) regenerates Figures 8-11
  using the included numerical inputs, without scientific plugins.
- [Saved endpoint](../../reproducibility/data/srmn2sb2_fit_endpoint/README.md)
  identifies the fitted parameters, backgrounds, predictions, and resolution
  samples used for Figure 9.
- [Endpoint recalculation](../../reproducibility/ENDPOINT_RECALCULATION.md)
  documents the separate scientific runtime and command for recomputing
  those predictions at the saved parameters.

The model uses the official pySpinW Python intensity backend, a single
in-plane spin orientation, the documented legacy resolution scenario, and
64 quasi-Monte Carlo resolution samples per measured point. Position-only
fits supplied initial values; they are not the reported intensity fit.
Per-scan normalization is an explicit part of the reported model.

Raw measurements remain in the public NCNR archive:
[201804/24858](https://ncnr.nist.gov/pub/ncnrdata/bt7/201804/24858/) and
[201806/24858](https://ncnr.nist.gov/pub/ncnrdata/bt7/201806/24858/).
The included derived inputs preserve scan filenames and measured
uncertainties. No literature PDFs or private corpus are distributed.
