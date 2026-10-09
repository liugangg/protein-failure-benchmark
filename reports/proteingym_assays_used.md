# ProteinGym assays used by this work, and their original experimental papers

ProteinGym's own README requires that the original experimental paper for each deep
mutational scanning assay be cited in addition to the ProteinGym benchmark paper. This file
records exactly which assays this work used and which papers they come from, so that the
requirement can be met without ambiguity.

**33 assays, derived from 13 original papers.** Note that the assay
count and the paper count are not the same number: several papers contribute more than one
assay. Earlier drafts of our documentation said "33 original assay papers", which conflated
the two; the correct statement is 33 assays from 13 papers.

Selection criterion, as recorded in `reports/paper_data_methods.md`: only assays with
`DMS_binarization_method == "manual"` are used, and only those whose
`coarse_selection_type` maps onto one of our stages (Stability, Expression, Binding).
Assays binarized by the dataset median are excluded, because a median cut labels 50% of
variants as failures by construction rather than by observation.

Author, year and title below are copied verbatim from ProteinGym's
`reference_files/DMS_substitutions.csv`. Journal and DOI are deliberately not filled in here:
they are not present in that file and we do not supply bibliographic details we have not
verified. ProteinGym ships `assays.bib` with the complete entries.

## Original experimental papers

| # | First author | Year | Title |
|---|---|---|---|
| 1 | McLaughlin | 2012 | The spatial architecture of protein function and adaptation |
| 2 | Olson | 2014 | A comprehensive biophysical description of pairwise epistasis throughout an entire protein domain |
| 3 | Matreyek | 2018 | Multiplex Assessment of Protein Variant Abundance by Massively Parallel Sequencing |
| 4 | Klesmith | 2019 | Retargeting CD19 Chimeric Antigen Receptor T Cells via Engineered CD19-Fusion Proteins |
| 5 | Starr | 2020 | Deep Mutational Scanning of SARS-CoV-2 Receptor Binding Domain Reveals Constraints on Folding and ACE2 Binding |
| 6 | Suiter | 2020 | Massively parallel variant characterization identifies NUDT15 alleles associated with thiopurine toxicity |
| 7 | Seuma | 2022 | An atlas of amyloid aggregation: the impact of substitutions, insertions, deletions and truncations on amyloid beta fibril nucleation |
| 8 | Clausen | 2023 | A mutational atlas for Parkin proteostasis |
| 9 | Gersing | 2023 | Characterizing glucokinase variant mechanisms using a multiplexed abundance assay |
| 10 | Muhammad | 2023 | High-throughput functional mapping of variants in an arrhythmia gene, KCNE1, reveals novel biology |
| 11 | Tsuboyama | 2023 | Mega-scale experimental analysis of protein folding stability in biology and design |
| 12 | Vanella | 2023 | Understanding Activity-Stability Tradeoffs in Biocatalysts by Enzyme Proximity Sequencing |
| 13 | Yee | 2023 | The full spectrum of OCT1 (SLC22A1) mutations bridges transporter biophysics to drug pharmacogenomics |

## Assays used, by stage

| Stage | ProteinGym DMS_id | Selection assay | Source paper |
|---|---|---|---|
| Binding | `CD19_HUMAN_Klesmith_2019_FMC_singles` | Binding affinity | Klesmith 2019 |
| Binding | `DLG4_RAT_McLaughlin_2012` | peptide binding - natural ligand | McLaughlin 2012 |
| Binding | `SPG1_STRSG_Olson_2014` | Binding (IgG) | Olson 2014 |
| Binding | `SPIKE_SARS2_Starr_2020_binding` | ACE2 binding | Starr 2020 |
| Expression | `HXK4_HUMAN_Gersing_2023_abundance` | abundance | Gersing 2023 |
| Expression | `KCNE1_HUMAN_Muhammad_2023_expression` | cell surface expression | Muhammad 2023 |
| Expression | `NUD15_HUMAN_Suiter_2020` | None | Suiter 2020 |
| Expression | `OXDA_RHOTO_Vanella_2023_expression` | cell surface expression | Vanella 2023 |
| Expression | `PRKN_HUMAN_Clausen_2023` | protein stability | Clausen 2023 |
| Expression | `S22A1_HUMAN_Yee_2023_abundance` | abundance | Yee 2023 |
| Expression | `SPIKE_SARS2_Starr_2020_expression` | ACE2 binding | Starr 2020 |
| Expression | `TPMT_HUMAN_Matreyek_2018` | Protein abundance (FACS sorting for abundance of GFP-fused target) | Matreyek 2018 |
| Stability | `A4_HUMAN_Seuma_2022` | aggregation | Seuma 2022 |
| Stability | `CSN4_MOUSE_Tsuboyama_2023_1UFM` | Stability | Tsuboyama 2023 |
| Stability | `HECD1_HUMAN_Tsuboyama_2023_3DKM` | Stability | Tsuboyama 2023 |
| Stability | `ILF3_HUMAN_Tsuboyama_2023_2L33` | Stability | Tsuboyama 2023 |
| Stability | `MAFG_MOUSE_Tsuboyama_2023_1K1V` | Stability | Tsuboyama 2023 |
| Stability | `MYO3_YEAST_Tsuboyama_2023_2BTT` | Stability | Tsuboyama 2023 |
| Stability | `NKX31_HUMAN_Tsuboyama_2023_2L9R` | Stability | Tsuboyama 2023 |
| Stability | `NUSG_MYCTU_Tsuboyama_2023_2MI6` | Stability | Tsuboyama 2023 |
| Stability | `OBSCN_HUMAN_Tsuboyama_2023_1V1C` | Stability | Tsuboyama 2023 |
| Stability | `OTU7A_HUMAN_Tsuboyama_2023_2L2D` | Stability | Tsuboyama 2023 |
| Stability | `PKN1_HUMAN_Tsuboyama_2023_1URF` | Stability | Tsuboyama 2023 |
| Stability | `POLG_PESV_Tsuboyama_2023_2MXD` | Stability | Tsuboyama 2023 |
| Stability | `PSAE_PICP2_Tsuboyama_2023_1PSE` | Stability | Tsuboyama 2023 |
| Stability | `RL20_AQUAE_Tsuboyama_2023_1GYZ` | Stability | Tsuboyama 2023 |
| Stability | `SDA_BACSU_Tsuboyama_2023_1PV0` | Stability | Tsuboyama 2023 |
| Stability | `TCRG1_MOUSE_Tsuboyama_2023_1E0L` | Stability | Tsuboyama 2023 |
| Stability | `THO1_YEAST_Tsuboyama_2023_2WQG` | Stability | Tsuboyama 2023 |
| Stability | `UBE4B_HUMAN_Tsuboyama_2023_3L1X` | Stability | Tsuboyama 2023 |
| Stability | `VILI_CHICK_Tsuboyama_2023_1YU5` | Stability | Tsuboyama 2023 |
| Stability | `VRPI_BPT7_Tsuboyama_2023_2WNM` | Stability | Tsuboyama 2023 |
| Stability | `YNZC_BACSU_Tsuboyama_2023_2JVD` | Stability | Tsuboyama 2023 |

