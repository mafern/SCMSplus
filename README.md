# SCMSplus
Python code used in (add arxiv link when ready).


Takes in an [MP-Gadget](https://github.com/sbird/MP-Gadget3) *PIG* snapshot and identifies filamentary structure in the DM halos. To use with a different input, threshold.py would need to be adjusted, however the other two steps could remain unchanged. Assumes the simulation box has periodic boundaries. The code is meant to run in the following order:

  1. **threshold.py:** reads in all halos, removes those in low density regions, and saves file with retained halos. i.e. "python threshold.py path/to/snapshot"

  2. **scms.py:** reads in halos from threshold step, runs SCMS algorithm until converged, saves file with the filaments. i.e. "python scms.py path/to/threshold_output"

  3. **separate.py:** reads in filaments, halos, and cutoff length, then separates into filaments, calculates filament lengths and masses, then saves into two files (filaments, and filament properties). i.e. "python separate.py path/to/scms&threshold/output cutoff_value"
