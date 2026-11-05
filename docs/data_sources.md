
# Disease data extension to 2025 (Sunaina)

- Raw disease CSVs extended to 2025 (dengue and chikungunya 2021-2025, malaria 2010-2025, from NCVBDC PDFs).
- Dengue has no data for any state in 2014, 2015 and 2016 (no source found). The model treats these years as missing,
  so 2014-2017 evaluation folds have fewer scored pairs.
- Values for 2021 and 2022 were revised in the new source (identical for 25-94% of pairs depending on disease; median
  log ratio about 0). Nothing before 2021 changed. The 2021 fold is the single largest advantage for the model; without
  it the mean difference from persistence is +0.007 (with it, -0.011). Both intervals include 0.
- A true holdout (v1 model trained through 2022, tested on 2023-2025) is not clean because 2022 values were revised.
- Zeros: an annual total of exactly 0 is treated as missing when the series has ever reached 100 (297 values). Rerun with
  zeros kept: see the zeros_kept result in the comparison table.
- Andhra Pradesh values before about 2010 are whole numbers times 0.5861509524794264 (cause unconfirmed).
- Whether 2025 is a full year: [fill in].
