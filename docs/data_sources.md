
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

# Dengue 2014-2016 gap filled (Sunaina)
- 2014: state-wise dengue cases and deaths from Lok Sabha unstarred question 1110 (answered 4 Dec 2015), Annexure-II.
  National total 40,571 reproduced.
- 2015 and 2016: "Dengue/DHF situation in India" table, 2015 and 2016 columns [record the URL and date accessed].
  Totals 99,913 (2015) and 129,166 (2016) reproduced; deaths totals 220 and 245 also match.
- The 2015 column of the Lok Sabha answer is provisional (83,021), so it was not used.
- Dadra and Nagar Haveli and Daman and Diu: the two rows were added together. Jammu and Kashmir includes Ladakh in these years.
- Lakshadweep and Ladakh are not in these tables and stay missing.
- Zeros for Manipur, Meghalaya and Nagaland in 2014 are as reported in the source.
- Provenance file: data/raw/extra_annual/dengue_2014_2016.csv.
