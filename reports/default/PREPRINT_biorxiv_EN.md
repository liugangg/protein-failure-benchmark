# Label artifacts in public protein-failure data originate from laboratory identity, and are removed by neither tag stripping nor adversarial debiasing

**— A cross-center evaluation benchmark, and why it cannot be skipped**

---

**Ganggang Liu**<sup>1,2,\*</sup>

<sup>1</sup> Open Intelligence Hub, 218 Sangtian Street, Suzhou Industrial Park, Suzhou 215123, China

<sup>2</sup> School of Life Sciences, Suzhou Medical College, Soochow University, 199 Ren'ai Road, Suzhou 215123, China

<sup>\*</sup> Correspondence: ganggliu@oihtech.com; ggliu@stu.suda.edu.cn

ORCID: [0009-0007-6982-201X](https://orcid.org/0009-0007-6982-201X)

> Preprint license: CC BY 4.0 — this applies to the manuscript text; the derived data are
> licensed separately and not uniformly, see §7.

---

## Abstract

Prediction of experimental failure in protein production lacks a common benchmark. We unify real experimental failure records from four public sources into a six-stage label schema, yielding 2,380,297 records; at 30% sequence identity these collapse to only 98,617 independent clusters. That gap is itself the first result of this paper: counting negative records overstates the effective sample size available for cross-protein tasks, by between less than one and almost four orders of magnitude depending on the source (largest shrinkage ÷6,661, i.e. 3.8 orders of magnitude; see §1.1).

**Sequence-level label artifacts in this kind of data have already been reported.** NetSolP (Thumuluri et al., 2022) found that 11,602 of 69,420 training sequences carried the N-terminal His-tag `MGSDKIHHHHHH` and that ~99% of them were insoluble, concluding that "the trained models focus more on the His-tag instead of the wild-type sequence"; SoluProt (Hon et al., 2021) deliberately balanced the sequence-length distribution so that length alone would not dominate prediction; and by the time PLM_Sol (Zhang et al., 2024) assembled UESolDS, stripping His-tag fragments had become a standard construction step. **This paper does not claim to have discovered these artifacts.**

What we address is the next question: **where the artifacts come from, and what remains once they are removed.** We attribute them to a single underlying level — **laboratory identity**. The presence and position of affinity tags, sequence length, and even whether negative labels exist at all are construction and record-keeping conventions of the individual structural genomics centers. The apparent predictive power of length largely vanishes within centers, and for two stages the within-center and cross-center directions are exactly opposite (Simpson's reversal), indicating that it measures convention differences between centers rather than biology. To our knowledge, none of the three works above performs any center-level comparison or stratification (verified against their full texts).

**The key result is that the artifacts cannot be removed.** After stripping construct residues following current standard practice, and simply not using length as a feature, cross-center generalization under held-out-center evaluation shows a **mixture of four regimes**: across 4 held-out center groups, one shows true signal, one weak but resolvable signal, one is indistinguishable from random, and one is reversed (regimes assigned from homology-cluster-stratified bootstrap 95% intervals, with the interval determining direction and the point estimate determining magnitude; a gradient-boosted tree using **only 20-dimensional amino-acid composition, without length** has intervals that overlap or exceed those of ESM-2 650M, indicating that representational capacity is not the bottleneck). **Both direction and magnitude are inconsistent across centers**, and on one held-out center group the same set of sequences flips the prediction direction entirely when the failure stage is changed (interval [0.77, 0.80] on `express` versus [1.03, 1.07] on `soluble_expression`).

**Conclusion: direction and magnitude are both inconsistent across centers, and at least one instance shows the same set of centers reversing direction between tasks. No cheap a priori criterion predicts whether a given deployment will land in the useful, harmful, or noise regime; each center and each task must be measured directly. This is both the cost and the necessity of a center-stratified evaluation protocol.**

We release the unified labels, four frozen splits with explicit leakage verification, and all evaluation code.

## 1 Data and scale

| Quantity | Value |
|---|---|
| Records | 2,380,297 |
| Unique sequences (DMS variants collapsed to their parents) | 354,019 |
| Clusters at 30% identity | **98,617** |

### 1.1 First result: the number of negative records is not the scale

| Stage | Negative records | Clusters at 30% | Shrinkage |
|---|---|---|---|
| `clone` | 18,398 | **5,663** | ÷3.2 |
| `express` | 94,165 | **22,771** | ÷4.1 |
| `soluble` | 9,995 | **4,640** | ÷2.2 |
| `purify` | 29,854 | **9,001** | ÷3.3 |
| `stable` | 153,858 | **3,766** | ÷41 |
| `bind` | 163,950 | **2,161** | ÷76 |

The table above aggregates by **stage**, which masks large differences between sources. Aggregating by **source** reveals the cause (the 32 and 220 below are the origin of the statement that "a hundred thousand negatives are equivalent to a few dozen independent samples"):

| Source | Negative records | Clusters at 30% | Shrinkage | Why |
|---|---|---|---|---|
| ds1_targettrack | 146,678 | **30,438** | ÷4.8 | Full-pipeline records; each entry is an independent target — smallest shrinkage |
| ds2_tsuboyama | 106,886 | **220** | ÷486 | Many single-point mutants of one set of parent domains (cDNA display proteolysis) |
| ds3_proteingym | 213,151 | **32** | ÷6,661 | Deep mutational scanning (DMS) of one set of parent proteins |
| ds5_dtu_binder | 3,275 | **1,998** | ÷1.6 | De novo designed binders; few targets, but designs differ substantially from one another |
| ds5_adaptyv_egfr | 230 | **40** | ÷5.8 | Single-target (EGFR) competition designs |

> Rounding rule for the "shrinkage" column in both tables: **one decimal place below 10×, integers at 10× and above.** The "less than one order of magnitude" in the abstract refers to the smallest value here, ÷1.6 (log₁₀ = 0.21); "almost four orders of magnitude" refers to the largest, ÷6,661 (log₁₀ = 3.82).

**What this measures is effective diversity for cross-protein generalization, not data quality.** For the purpose DMS data are made for — predicting variant effects within a single protein — these are excellent datasets. But for the task of predicting whether an unseen protein will fail, the number of negatives a source contains matters little; what matters is how many independent protein families it covers. The per-stage table and the per-source table do not reconcile precisely because the negatives for `stable` and `bind` come almost entirely from DMS-type sources.

## 2 Related work and the increment contributed here

### 2.1 The label artifacts were not discovered by this paper

We place this first so that the boundary of our contribution is not misread.

**Thumuluri et al., 2022** (NetSolP) reported affinity-tag artifacts in *E. coli* solubility data:

> "An example of this is that 11 602 out of 69 420 sequences of the training set and 344 out of 2001 sequences of the test set have the N-terminal His-tag 'MGSDKIHHHHHH' with ~99% and ~97% of them being insoluble, respectively."

> "The consequence of this is that the trained models focus more on the His-tag instead of the wild-type sequence."

They also relayed the inconsistency of the labels themselves (this figure comes from a comparison by Hon et al., 2021 against Price et al., 2011, and is not a NetSolP measurement):

> "[Hon et al. (2021)] compared the labels of sequences from this dataset with another dataset whose solubility was provided separately ([Price et al., 2011]) and found that around 18.6% of labels were different, even with 100% identical sequences."

**Hon et al., 2021** (SoluProt) addressed the length confound at dataset-construction time: "we balanced the sequence length distribution so that length alone would not play a dominant role in the predictions."

**Zhang et al., 2024** (PLM_Sol / UESolDS) made tag stripping a fixed step of dataset construction:

> "The following His tag fragments in proteins were excluded due to an uneven distribution of these tags between Insol and Sol proteins revealed by NetSolP: "MGSDKIHHHHHH", "MGSSHHHHHH", "MHHHHHHS", "MRGSHHHHHH", "MAHHHHHH", "MGHHHHHH", "MGGSHHHHHH", "HHHHHHH" and "AHHHHHHH"."

**Thus stripping affinity tags and preventing length from dominating are standard practice established in this field between 2021 and 2024. This paper neither claims to have discovered these artifacts nor counts tag stripping as a contribution.**

### 2.2 The closest prior work: the crystallization-propensity / multi-stage line

The three works above predict **solubility alone**. A separate line of work is closer to our framework — it uses **the same TargetTrack / PepcDB data and likewise predicts several consecutive experimental stages**. It is the specific object of our critique and must be treated separately.

The representative work is **Wang et al., 2014** (PredPPCrys):

> "We downloaded the most recent datasets from the PepcDB database comprising 108,933 targets and 979,645 experimental trials."

It predicts cloning / production / purification / crystallization / diffraction-quality crystallization (five consecutive stages), which is nearly the same problem as our six-stage labels. Redundancy was reduced as follows:

> "We reduced sequence homology in the datasets by removing sequences with >=40% sequence identity using CD-HIT within each class."

> "We applied BLAST to further reduce the sequence redundancy between the training and independent test datasets using a cutoff of 25% sequence identity."

**This is not an isolated paper but a line spanning a decade.** To confirm we had not missed comparable work, we consulted **Wang et al., 2018**, a systematic evaluation of predictors in this line:

> "The data sets used to develop these tools were derived from a number of relevant public databases, including TargetDB, PepcDB and TargetTrack."

> "Three tools (PPCPred, PredPPCrys and Crysalis) also predict the propensity for successfully completing some of the crucial steps during the crystallization process."

| Work | Year | Stages covered | Splitting | Center hold-out |
|---|---|---|---|---|
| **PPCpred** (Mizianty et al., 2011) | 2011 | production / purification / crystallization | sequence-homology reduction only | **none** |
| **PredPPCrys** (Wang et al., 2014) | 2014 | cloning / production / purification / crystallization / diffraction-quality crystallization | sequence-homology reduction only | **none** |
| **Crysalis** (Wang et al., 2016) | 2016 | multi-stage (per the review) | sequence-homology reduction only | **none** |
| **fDETECT** (Meng et al., 2017) | 2017 | production / purification / crystallization | sequence-homology reduction only | **none** |
| **deep-cascade forest** (Zhu et al., 2021) | 2021 | multi-stage | sequence-homology reduction only | **none** |
| This paper | 2026 | clone / express / soluble / purify / stable / bind | homology transitive closure **+ held-out centers** | **yes** |

> **Verification level of each cell, stated explicitly.** The thresholds and the "no center stratification" entry for the PredPPCrys row were obtained by **checking that paper's full text item by item** (quotations above). For the other four papers, "sequence-homology reduction only / no center hold-out" is a family-level conclusion taken from the systematic re-evaluation in Wang et al., 2018 — that review describes the uniform redundancy-reduction protocol used across this line (25% sequence identity within and between classes) and contains no stratification or evaluation split by laboratory or center anywhere in its text. **We did not re-read the full texts of those four papers**, so if any of them contains a center-level analysis not recorded by the review, that cell of this table will be wrong. This is an explicitly stated verification boundary, not a verified assertion.

**There is only one key difference, but it determines what the reported scores mean.** Splitting in this line is uniformly sequence-homology reduction alone (CD-HIT 40% within class, BLAST 25% between training and test, or the review's uniform 25% in its re-evaluation), and **not one of these works performs a laboratory hold-out or center-level stratification** — while the review itself states that TargetTrack data come from ">40 structural genomics centers worldwide". By the measurement in our §5.2, within-center and cross-center training differ by 1.85 versus 1.01 **on the identical set of test items**. **Homology-based redundancy reduction is therefore insufficient to exclude the laboratory-identity channel**, and the scores reported in this line may be inflated by it.

**The wording here must remain restrained.** We did not re-run these models, so we cannot say that their reported scores are false. What we can say is that their evaluation protocols do not separate "predicting whether a protein is difficult to produce" from "recognizing which center a sequence came from", and that this paper shows the latter is in fact learnable on the same data source and sufficient to account for most of the apparent performance. The only point in this line that touches the issue is the pipeline difference noted in the review:

> "a portion of crystal structures deposited in PDB was determined traditionally by structural biologists using specialized equipment, unique protocols and laborious trial-and-error efforts."

That is, the review recognizes that high-throughput pipelines and traditional case-by-case determination yield structures with different properties, but does not carry the observation down to the center level: it neither stratifies evaluation by center nor tests whether center identity is itself predictable from sequence.

On splitting leakage, two related works make distinct claims and we separate them here (citing them jointly would be incorrect): **Bushuiev et al., 2024b** ("Revealing Data Leakage in Protein Interaction Benchmarks", ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design) is the dedicated demonstration that "commonly used splitting strategies for protein complexes, based on protein sequence or metadata similarity, introduce major data leakage"; **Bushuiev et al., 2024a** (ICLR 2024 main conference) is the source of the PPIRef dataset and the iDist near-duplicate detection algorithm, and constructs leakage-free splits on that basis. The split-group construction in our §4 follows the same direction as both; it adopts rather than reinvents their approach.

### 2.3 What this paper adds, and why it is not a repetition of the above

The three solubility papers treat the artifact as a **data-cleaning problem**: identify a sequence motif (the His-tag) or a surface statistic (length), remove or balance it, and continue modelling. The crystallization-propensity line does not reach even that step; what it addresses is **homology redundancy**. This paper asks the question neither line asks: once the visible artifacts are cleaned and homology leakage is excluded, what remains across laboratories? The answer is negative.

The "not covered by prior work" column below refers to **all eight works** listed in §2.1 and §2.2 (three on solubility, five on crystallization propensity). **Verification levels follow the footnote in §2.2 and are not re-argued here**: the three solubility papers and PredPPCrys were verified against their full texts; the remaining four rest on the family-level conclusion of Wang et al., 2018, whose boundary that footnote states.

| # | What this paper does | Why prior work does not cover it |
|---|---|---|
| 1 | **Center-level attribution**: tag usage rate (0.02%→91.8%), tag position (N- versus C-terminal, nearly dichotomous), and whether negative labels exist at all are all center fingerprints | **None of the eight performs any comparison or stratification by laboratory / center** (verification level per the §2.2 footnote); NetSolP treats the His-tag as a sequence motif and does not ask why one set of sequences carries tags and another does not; the crystallization-propensity line uses TargetTrack / PepcDB directly (data from >40 centers) and still does not split by center |
| 2 | **Simpson's reversal**: the predictive power of length vanishes within centers, and for two stages the within-center and cross-center directions are exactly opposite | SoluProt balanced the length distribution so that length would not dominate, but did not test whether the predictive power of length itself arises from between-center differences |
| 3 | **Same-fold within / cross comparison**: the test set is identical item by item; only the source of the training data changes | None of the eight has a held-out-center setting, so this comparison does not exist in them |
| 4 | **Four regimes + homology-cluster-stratified bootstrap + seed robustness**: separating "is there cross-center signal" into a question of direction and a question of magnitude, and requiring conclusions to be insensitive to implementation choices | All eight report aggregate metrics on a single test set (accuracy / MCC / AUC and similar) without group-stratified intervals, so they cannot separate "is the direction resolvable" from "is the magnitude sufficient" |
| 5 | **Quantified effective diversity**: negative record counts and 30%-identity cluster counts differ by up to ÷6,661 (3.8 orders of magnitude) by source | Dataset scale in all of the above is reported as sequence or target counts (e.g. the 108,933 targets / 979,645 trials of PredPPCrys) |
| 6 | **Exclusionary result from adversarial debiasing**: center separability is suppressed to near majority-class guessing, and the cross-center reversal is unchanged | None of the eight attempts to remove center identity — because none identifies it as a confound (the solubility papers identify tags / length; the crystallization line identifies homology redundancy) |
| 7 | **Separating homology reduction from laboratory hold-out**: showing the former does not entail the latter (same-fold within 1.85 versus cross 1.01) | The crystallization-propensity line (Wang et al., 2014 and four others) treats CD-HIT 40% within class plus BLAST 25% between classes as sufficient, and the review (Wang et al., 2018) likewise uses only a 25% homology threshold in its uniform re-evaluation; no work in the line performs a center hold-out |

Put differently, this paper adds one step to each of two lines. To the three solubility papers: **prior work showed that models are looking at the tag; this paper shows that the tag is only a visible proxy for center identity, and that removing the proxy — or even actively erasing center separability by adversarial training — still leaves cross-center generalization unestablished.** To the crystallization-propensity line: **prior work treats homology reduction as sufficient for generalization assessment; this paper shows it is not — after homology reduction, laboratory identity still independently accounts for most of the apparent performance.** Together these form a negative result whose use is to price the evaluation protocol: cross-center hold-out is expensive (the test set shrinks to a few thousand items, and every center and every task must be measured separately), but it cannot be skipped.

## 3 Laboratory identity is the source of the artifacts

### 3.1 The failure records themselves are a center convention

Among the 17 centers that attempted expression and have ≥1,000 records, the expression failure rate ranges from 73.7% down to **0.0%**: three centers have a failure rate of exactly zero, totalling 367,452 records (the largest, JCSG, accounts for **350,702**). Tens of thousands of experiments without a single failure is not biologically coherent — **whether negative labels exist at all is itself strongly associated with which center performed the work.**

> **Note on denominators** (the JCSG figures here and in §3.2 differ; this is not a typographical error). The 350,702 records here are those JCSG records that **reached the expression step and therefore have an `express` label other than −1**; the 378,364 in §3.2 are **all** JCSG records in DS1 (including those that reached only cloning, or for which the step was not observed). The difference of 27,662 records are those with `express` unobserved, recorded as −1 under our three-state label rule and excluded from both loss and evaluation. Every count in this paper states its denominator.

### 3.2 Both the presence and the position of tags are center fingerprints

| Fact | Measured |
|---|---|
| Records containing `HHHHHH` (all of DS1, n=939,665) | 21.1% |
| Range of usage across the 12 large centers (n≥10,000) | SSGCID **91.8%** → NYCOMPS **0.02%** |

Tag **position** is equally a fingerprint (proportion of tagged records with the tag at the N- or C-terminus):

| Center | Records | His6 usage | N-terminal | C-terminal |
|---|---|---|---|---|
| NYSGXRC | 44,727 | 54.5% | 2% | 98% |
| NESG | 133,249 | 69.4% | 41% | 59% |
| SGPP | 20,856 | 55.3% | 100% | 3% |
| SSGCID | 21,715 | 91.8% | 99% | 1% |
| SECSG | 14,816 | 16.5% | 99% | 0% |
| JCSG | 378,364 | 11.5% | 100% | 0% |

The same tag sequence appears almost exclusively at the N-terminus in one center and almost exclusively at the C-terminus in another. **Both the presence and the position of tags are center fingerprints, and together they provide the model with a sequence-visible laboratory-identity channel** — the model need not learn what makes a protein hard to express, only which center submitted the sequence.

### 3.3 Within-center stratification: the predictive power of length is a Simpson's reversal

| Stage | Cross-center rank AUC | Within-center median | Within-center range |
|---|---|---|---|
| `clone` | 0.621 | 0.547 | [0.5158, 0.6608] |
| `express` | 0.577 | 0.533 | [0.4926, 0.6751] |
| `soluble` | 0.427 | 0.441 | [0.4209, 0.5706] |
| `purify` | 0.476 | 0.545 | [0.4703, 0.703] |
| `stable` | 0.412 | 0.548 | [0.5043, 0.5916] |

For `purify` and `stable` the **cross-center and within-center directions are exactly opposite** (0.476 versus 0.545; 0.412 versus 0.548), while for `clone` and `express` the predictive power falls from 0.62 / 0.58 to 0.55 / 0.53, close to uninformative. **The apparent predictive power of length comes mainly from convention differences between centers.**

### 3.4 Asymmetry of the ablation

After removing construct residues (21 pattern classes: His / FLAG / MYC / HA / Strep / T7 tags, TEV / thrombin / Factor Xa / enterokinase sites, GS / GGGGS linkers, vector cloning-site remnants) and dropping the length feature:

- **Sequence split (training and test from the same centers)**: all 6 tasks **decline**, mean **−4.8%**
- **Laboratory split (cross-center)**: 3 of 4 tasks decline, largest change in the opposite direction +9.8%, mean **−1.5%**

**Were this a genuine biological effect, both splits should decline equally.** In fact the same-center side declines in 6 of 6 tasks, by roughly 3.2 times the cross-center magnitude, while the cross-center side declines only slightly and inconsistently in direction (one task improves) — indicating that these two channels are usable when training and test share centers and fail across centers, i.e. that they serve an identifying function. **Note, however, that the statistical strength of this comparison is limited**: only 4 tasks on the cross-center side have sufficient negatives to evaluate, and a ±10% fluctuation in a single task would change the mean. This evidence therefore stands **alongside** the within-center stratification test of §3.3 as corroboration, and does not carry the conclusion on its own.

## 4 Methods: four frozen splits with verified leakage

The unit of splitting is neither the sequence nor the set-cover cluster, but the **split group** — the connected components of the 30%-identity transitive closure obtained from an all-versus-all search over all unique sequences.

| Procedure | Sequence pairs >30% between train and test |
|---|---|
| Whole-group assignment by 30% set-cover cluster | 10,841 (max fident 1.00) |
| plus: merging after pairwise comparison of cluster representatives | 2,302 |
| plus: all-versus-all transitive closure over all sequences | 16 / 1 / 8 |
| plus: iterative repair | **0 / 0 / 0** |

**Splitting leakage is a problem that has been studied specifically; this paper adopts rather than reinvents the approach.** Bushuiev et al., 2024b demonstrated in protein-interaction data that "commonly used splitting strategies for protein complexes, based on protein sequence or metadata similarity, introduce major data leakage", with the consequence that this "may result in overoptimistic evaluation of generalization, as well as unfair benchmarking of the models, biased towards assessing their overfitting capacity rather than practical utility"; the companion main-conference work, Bushuiev et al., 2024a, provides the non-redundant PPIRef dataset and the iDist near-duplicate detection algorithm, and constructs leakage-free splits by 3D interface similarity on that basis. Our task is sequence-level rather than interface-level, so we use the transitive closure of sequence homology as the splitting unit, but the diagnostic approach is the same: **assume your own split is leaking, then measure it explicitly.** The table above is that measurement.

A point related to §2.2 is worth noting here: the crystallization-propensity line treats homology reduction (CD-HIT / BLAST thresholds) as sufficient for generalization assessment, whereas these two works in the PPI domain and this paper in the sequence domain provide counterexamples in the same direction — **removing homology redundancy is not the same as removing leakage**; it removes only one kind.

The cascaded clustering of MMseqs2 is a greedy set cover: it guarantees only that members satisfy the threshold with respect to their own cluster representative, and **not that members of different clusters fall below it** — splitting by cluster therefore leaks. Iterative repair remains necessary at the end because group construction and verification are two independent searches, with heuristic k-mer matching as the prefilter.

Taking the transitive closure at 30% identity and 80% coverage, the target space forms a single giant connected component (containing 28,646 set-cover clusters). The cause is not short-fragment bridging (verified) but multi-domain proteins bridging between families. Groups accounting for more than 1% are forced to the training side; **the cost is that the test set systematically excludes proteins that sit within large homology networks.**

| Split | What it tests | Test records | Leakage check |
|---|---|---|---|
| sequence | basic generalization | 73,219 | **PASS** (0 pairs) |
| lab | whether laboratory bias has been learned | 5,843 | **PASS** (0 pairs) |
| time | temporal drift | 606 | **PASS** (0 pairs) |
| bind_target | binder prediction for unseen targets | 300 | **PASS** (0 pairs) |

**Evaluation convention.** The positive class is failure. The primary metric is **PR-AUC/base** (= PR-AUC divided by the positive-class prevalence). lift@k is reported alongside as an interpretability reference ("how many hits among a fixed budget of k wet-lab slots") but is not used for adjudication: cluster-stratified bootstrap shows that on a fold with n≈2,500 the lift@100 interval is 0.63 wide (using only the top 100 = 4% of the sample) whereas PR-AUC/base is 0.17 wide, and the two once gave opposite verdicts on one fold. The timing and impact of this rule change are recorded in the changelog in the supplementary materials.

## 5 Results

### 5.1 Both existing tools and the trivial baseline are close to random under cross-center evaluation

Against the composite label matching SoluProt's definition (`express` successful and `soluble` successful):

| Method | Sequence split PR-AUC/base | Laboratory split PR-AUC/base |
|---|---|---|
| SoluProt (full set) | 1.47 | 1.16 |
| SoluProt (decontaminated subset) | 1.44 | 1.15 |
| Amino-acid composition GBDT (default configuration: tags stripped + **no length**) | 1.60 | 1.17 |

**We do not describe this as outperforming SoluProt.** SoluProt's `ecoli_usearch_identity` feature (maximum identity to *E. coli* PDB sequences) cannot be computed for **6,730/16,435 = 41% of the sequences** in our sequence-split test set (no usearch hit; SoluProt falls back to the training-set mean), against 4/21 = 19% in the test examples shipped with the tool. It was trained and designed for naturally occurring proteins heterologously expressed in *E. coli*, whereas our test set contains designed proteins, membrane proteins, and many sequences with no significant PDB homolog — we are applying it outside its domain of applicability, **so the SoluProt figures above are a lower bound for it.** The accurate statement is: **on this dataset, a GBDT using only 20-dimensional amino-acid composition (no length, construct residues stripped) matches or exceeds a published tool, while SoluProt's core feature degrades severely out of domain.**

**This comparison has two asymmetries, both unfavourable to SoluProt, and we state them:**

1. **Out of domain versus in domain.** SoluProt was trained on its own dataset and transferred by us to this one for inference; our GBDT is trained directly on this dataset. Cross-domain use necessarily lowers its score.
2. **Feature coverage.** Its PDB-identity feature cannot be computed for 41% of our test set (19% in its own test examples), falling back to the training-set mean.

**One further point we examined and concluded is not an asymmetry: length.** Hon et al., 2021 handled it by **balancing the length distribution at dataset-construction time** ("we balanced the sequence length distribution so that length alone would not play a dominant role in the predictions."); our handling is **not to use length as a feature at all**. These are two different operations, not two descriptions of the same one: the former changes the distribution of the training data, the latter changes the feature set. But their effect in this comparison runs in the same direction — **neither party gains from length** — so there is no unfairness of the form "we quietly used length and it did not".

One point must nevertheless be made on SoluProt's behalf, or the comparison remains unfair: **our test set is not length-balanced.** SoluProt's model was fitted on training data whose length distribution had been balanced, and is now transferred to a test set whose length distribution has not been, which may itself disadvantage it (the decision function it learned assumes a different marginal distribution over length). We have not quantified this — doing so would require retraining SoluProt, which is outside the scope of this paper. We therefore record it as **a known but unquantified residual asymmetry in our favour.**

A final caution about configurations: under the length-retaining ablation configuration (`reports/ablation_keeptags_keeplen/`) our GBDT scores higher (see §3.4), and **that figure must not be placed alongside SoluProt's** — what it gains from is precisely the portion this paper identifies as the laboratory-identity channel.

The use we make of this GBDT is therefore: **as a lower-bound reference point for a "trivial baseline", not as a claim that our method is better.** Its function is to show how small the gap is on this dataset between existing tools and trivial features.

Contamination check (SoluProt states that its training set derives from TargetTrack, the same source as our DS1): contamination at 30% identity is 18.7% for the sequence split and 9.9% for the laboratory split, so every figure is given both for the full set and for the decontaminated subset. The laboratory split is essentially unchanged after decontamination — **contamination did not inflate it** — which is a point in its favour and we record it as such.

### 5.2 Positive control: within-center far exceeds cross-center on the identical test fold

This is the comparison that weighs the contribution of the identity channel directly. **The test sets on the two sides are identical**; only the source of the training data changes: `within` draws training data from **other homology groups inside the held-out center**, `cross` from the other centers. The difference between within and cross therefore contains no test-set difference, only the single variable of whether the training data share a center with the test data.

| Training data source | PR-AUC/base median (L0 composition GBDT) | PR-AUC/base median (L1 frozen PLM) |
|---|---|---|
| **within**: inside the held-out center | **1.58** | **1.85** |
| **cross**: the other centers | 0.97 | 1.01 |
| Random control (measured) | 1.03 | 1.03 |

**To close off the most natural objection first — could `within` simply be overfitting a small dataset?** It could not, and the direction is in fact the reverse: **`cross` has far more training data.**

| Fold | Held-out centers | within training clusters | cross training clusters | cross / within |
|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 1,650 | 29,284 | **17.7×** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,768 | 23,107 | **4.0×** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,472 | 25,051 | **5.6×** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 4,932 | 24,360 | **4.9×** |

After redundancy reduction this dataset retains one record per `split_group`, so the counts above **are homology-cluster counts**, not record counts inflated by homologous redundancy.

The `cross` side has **4.0–17.7 times** as many training clusters as `within`, yet falls from 1.85 to 1.01 on the identical set of test items. **More data, worse performance** — so the gap cannot be overfitting due to insufficient sample size on the `within` side, nor underfitting on the `cross` side. The only variable that changes with it is whether the training data come from the same center as the test data.

**With same-center training the model is effective (L1 1.85); switching to cross-center training collapses it to the random level (1.01 against a random control of 1.03).** This positively establishes that there is something learnable, but that it **does not transfer across centers** — the other face of the identity-channel evidence in §3. The random control falls near 1.0, showing that the evaluation is calibrated and that the low cross-center values are not a computational error.

> One implementation trap in this comparison must be stated, because we fell into it ourselves. We initially defined the `within` training set as "the part of the held-out center not belonging to the test group", but the test fold is **all** of that center's groups, so the mask was identically empty and the `within` column could never obtain a value. The correct procedure is to divide the held-out center's homology groups into a further K=3 parts, use the i-th inner part as test and the remainder as `within` training data, and evaluate `cross` on that same part, so that the test sets on the two sides are identical item by item. The bug was found before any GPU run by checking with numpy that the mask was non-empty; otherwise a full round would have been wasted and would have produced the erroneous conclusion that "within has no data".

### 5.3 The trivial baseline is not surpassed by the language model

On the same held-out centers and the same test folds, L0 (**a GBDT using only 20-dimensional amino-acid composition, without the length feature**) compared against L1 / L2 (ESM-2 650M frozen / LoRA fine-tuned):

| Fold | Base rate | L0 GBDT | L1 frozen | L2 LoRA |
|---|---|---|---|---|
| 1 | 0.306 | 1.33 [1.25, 1.42] · true signal | 1.20 [1.13, 1.28] · true signal | 1.28 [1.21, 1.37] · true signal |
| 2 | 0.032 | 0.86 [0.81, 0.94] · reversed | 0.97 [0.90, 1.07] · indistinguishable | 1.13 [1.0199, 1.3076] · borderline |
| 3 | 0.413 | 0.77 [0.76, 0.78] · reversed | 0.79 [0.77, 0.80] · reversed | 0.84 [0.83, 0.86] · reversed |
| 4 | 0.522 | 1.02 [0.9957, 1.0399] · indistinguishable | 1.05 [1.03, 1.07] · weak but resolvable | 1.04 [1.0139, 1.0582] · weak but resolvable |

**On the only fold with stable true signal (fold 1), L0's PR-AUC/base of 1.33 [1.25, 1.42] exceeds L1's 1.20 and overlaps L2's interval of 1.28 [1.21, 1.37].** That is: substituting a 650M-parameter protein language model buys nothing beyond **20-dimensional amino-acid composition** in the cross-center setting. This is the same fact as the GBDT matching or exceeding published tools in §5.1, not an independent second result.

L0 is **reversed** on fold 2 while L1 / L2 are not, which indicates that the PLM representation at least does not make that fold worse; but this does not constitute evidence that the PLM is better, because L2 on fold 2 is itself a borderline case (see §5.6).

### 5.4 A mixture of four regimes: direction and magnitude both inconsistent across centers, with one instance of reversed direction between tasks

Adjudication rules (`configs/stage3_train.yaml` changelog `[2026-10-02]`, implemented in `src/eval/bootstrap_ci.py`): bootstrap stratified by `split_group`, 2,000 resamples. **The interval determines whether the direction is resolvable; the point estimate determines whether the magnitude is sufficient.**

| Verdict | Condition |
|---|---|
| True signal | interval does not cross 1.0, point estimate ≥ 1.10, and the adversarial version's interval likewise does not cross 1.0 |
| Borderline | formally meets the true-signal criteria, but **the adversarial version's interval crosses 1.0** — a robust signal does not vanish on the addition of an adversarial head |
| Weak but resolvable | interval does not cross 1.0 but point estimate < 1.10 (statistically resolvable, magnitude only a few percent) |
| Indistinguishable | interval crosses 1.0 |
| Reversed | interval lies entirely below 1.0 |
| *(prerequisite)* Direction robustness | the direction verdict above must agree across **5 independent bootstrap seeds**; where it does not, the conservative direction is taken and the cell is recorded as "indistinguishable" |

**Why the seed-robustness rule was added.** During final checking, two cells both displayed as `[1.00, 1.04]` at two decimal places yet received opposite verdicts (lower bound 0.9957 crossing 1.0 versus 1.0010 not crossing), a difference of 0.001. Direct testing showed that one of them had a direction verdict of 4 positive and 1 indeterminate across 5 seeds — **the conclusion changes with the seed**; raising the number of resamples from 2,000 to 20,000 still gave 1.0007 and did not resolve it, because the uncertainty lies in the data rather than in the resampling count. A direction that a random seed can decide is not a conclusion. This follows the same discipline as "a signal that vanishes on the addition of an adversarial head is not a true signal": conclusions must be insensitive to implementation choices unrelated to the scientific question. We did not take a majority vote, because 4:1 should not be treated as settled either. Across all 40 fold-level results, **exactly one was reclassified by this rule** (L2 `soluble_expression` fold 3, from "weak but resolvable" to "indistinguishable"). Where an interval endpoint falls within ±0.02 of 1.0, this paper reports four decimal places throughout.

> Cells marked ᵇ in the tables are of this kind: the direction verdict disagrees across the 5 seeds, so even where the printed interval endpoints do not cross 1.0, the conservative direction is taken and recorded as "indistinguishable", with no directional conclusion drawn.

Separating "resolvable" from "sufficient" is necessary: on this paper's largest fold (n=7,399), even a +5% improvement has an interval that does not cross 1.0. Looking only at the interval, it would be reported as a true signal; looking only at the point estimate, genuine moderate effects on small folds would be missed.

**`express` (primary task) · L1 frozen**

| Fold | Held-out centers | n | Base rate | PR-AUC/base [95% CI] | Verdict |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,475 | 0.306 | 1.20 [1.13, 1.28] | **true signal** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 8,652 | 0.032 | 0.97 [0.90, 1.07] | **indistinguishable** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 6,708 | 0.413 | 0.79 [0.77, 0.80] | **reversed** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,399 | 0.522 | 1.05 [1.03, 1.07] | **weak but resolvable** |

**`express` · L2 LoRA**

| Fold | Held-out centers | n | Base rate | PR-AUC/base [95% CI] | Verdict |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,475 | 0.306 | 1.28 [1.21, 1.37] | **true signal** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 8,652 | 0.032 | 1.13 [1.0199, 1.3076] | **borderline** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 6,708 | 0.413 | 0.84 [0.83, 0.86] | **reversed** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,399 | 0.522 | 1.04 [1.0139, 1.0582] | **weak but resolvable** |

**`soluble_expression` (control) · L1 frozen**

| Fold | Held-out centers | n | Base rate | PR-AUC/base [95% CI] | Verdict |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,539 | 0.404 | 1.28 [1.22, 1.35] | **true signal** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,388 | 0.080 | 0.99 [0.94, 1.07] | **indistinguishable** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,693 | 0.640 | 1.05 [1.03, 1.07] | **weak but resolvable** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,147 | 0.611 | 1.12 [1.10, 1.14] | **true signal** |

**`soluble_expression` (control) · L2 LoRA**

| Fold | Held-out centers | n | Base rate | PR-AUC/base [95% CI] | Verdict |
|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | 2,539 | 0.404 | 1.32 [1.26, 1.40] | **true signal** |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | 5,388 | 0.080 | 1.01 [0.95, 1.08] | **indistinguishable** |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | 4,693 | 0.640 | 1.02 [1.0010, 1.0415] | **indistinguishable ᵇ** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | 7,147 | 0.611 | 1.13 [1.11, 1.15] | **true signal** |

**Cross-task comparison** (same held-out centers, same L1 architecture, direction and magnitude read separately):

| Fold | Held-out centers | `express` direction | `express` magnitude | `soluble_expression` direction | magnitude | Directions agree |
|---|---|---|---|---|---|---|
| 1 | NESG, RSGI, NATPRO | **positive** | 1.20 (+20%) | **positive** | 1.28 (+28%) | yes |
| 2 | MCSG, SGPP, SGX, CSMP, MPP, TRANSPORTPDB | indeterminate | 0.97 (−3%) | indeterminate | 0.99 (−1%) | yes |
| 3 | NYSGRC, NYCOMPS, SSGCID, SECSG, BSGI, BSGC | **negative** | 0.79 (−21%) | **positive** | 1.05 (+5%) | **opposite** |
| 4 | CSGID, NYSGXRC, EFI, CESG, TB, SPINE | **positive** | 1.05 (+5%) | **positive** | 1.12 (+12%) | yes |

**Two statements are required here, of different strength.**

1. **Cross-center inconsistency — supported by all 4 folds.** On `express` the directions are positive, indeterminate, negative, positive; magnitudes range from −21% to +20%. Even restricting attention to the 2 folds with positive direction, the magnitudes differ fourfold (+5% versus +20%). This is the main conclusion of the paper.
2. **Reversed direction between tasks — a single instance, reported as a single observation.** On fold 3 the `express` interval lies entirely below 1.0 while the `soluble_expression` interval lies entirely above 1.0, so the directions are indeed opposite; **the other three folds agree in direction** (folds 1 and 4 both positive, fold 2 both indeterminate) and differ only in magnitude. We therefore write that reversed direction was observed on fold 3 and not on the other folds, and **do not present cross-task inconsistency as a general rule** — one instance cannot support a general statement.

**Independent check on L2 (LoRA fine-tuning) — the result does not replicate, and we report it as such.**

| Fold | `express` direction / magnitude | `soluble_expression` direction / magnitude | Directions |
|---|---|---|---|
| 1 | **positive** / 1.28 (+28%) | **positive** / 1.32 (+32%) | agree |
| 2 | **positive** / 1.13 (+13%) | indeterminate / 1.01 (+1%) | not comparable |
| 3 | **negative** / 0.84 (−16%) | indeterminate ᵇ / 1.02 (+2%) | not comparable |
| 4 | **positive** / 1.04 (+4%) | **positive** / 1.13 (+13%) | agree |

On L2 the `express` side of fold 3 remains reversed, but the direction on the `soluble_expression` side **cannot be determined** — the 5 bootstrap seeds give ['pos', 'cross', 'pos', 'pos', 'pos'], which disagree, so the conservative direction is taken and it is recorded as indistinguishable (point estimate 1.02, same sign as L1 but not significant). **The reversed direction on fold 3 therefore does not replicate on L2 as an adjudicable comparison.** This does not constitute evidence against the L1 result (the two point estimates have the same sign and no fold reverses direction), but neither does it provide independent corroboration.

We therefore state this only as far as it will carry: **on L1, fold 3 shows opposite directions between the two tasks (both sides seed-stable); on L2 the comparison does not replicate because one side's direction cannot be adjudicated; no other fold shows reversed direction.** This is a single observation on a single center group, not a rule.

Together these two statements close off the two most natural remedies: one cannot rely on "this center performed well before" to transfer (direction and magnitude are both inconsistent across centers), nor assume that "no problem on one task means no problem on another" (there is at least one counterexample). No a priori quantity — center identity, base failure rate, sample size, or task — predicts which regime the next deployment will fall into.

**The boundary of the reversal must be stated clearly.** Reversal appears on only 1 held-out center group of the `express` task, consistently across L1 / L2 / the adversarial version ([0.77, 0.80] / [0.83, 0.86] / [0.78, 0.81]). We **do not describe cross-center reversal as a general phenomenon of this kind of data** — it is one of the four regimes.

### 5.5 Exclusions: neither insufficient model capacity nor residual center identity

| What is excluded | Evidence |
|---|---|
| Not insufficient model capacity | After fine-tuning 5.41M LoRA parameters (0.82%) for 3 epochs, the fold-3 reversal persists (PR-AUC/base 0.84 [0.83, 0.86], interval far from 1.0) |
| Not residual center identity | The adversarial head's center-discrimination accuracy falls to 0.429–0.507 (≈ majority-class guessing), i.e. center separability has been removed from the representation; yet the fold-3 reversal is unchanged (0.79 → 0.80) |

Both routes are excluded and the reversal persists, **pointing to the same open question**: the reversal may originate in a label-generating mechanism that varies by center rather than in the sequence representation itself. The present data cannot adjudicate this, and we do not speculate further.

### 5.6 A borderline case (we do not clear the bar ourselves either)

L2 on `express` fold 2 (base rate 0.032) gives PR-AUC/base = 1.13 [1.0199, 1.3076], an interval that formally does not cross 1.0; but with the adversarial head added it returns to [0.96, 1.22], which does cross 1.0, and the lift@100 point estimate of 0.62 points in the opposite direction. **The direct basis for the borderline verdict: a robust signal does not vanish on the addition of an adversarial head.**

The threshold this paper sets for subsequent work therefore requires **both conditions** (see `reports/paper_data_methods.md` §2.6): **(a)** a cluster-stratified bootstrap 95% interval that does not cross 1.0, and **(b)** a PR-AUC/base point estimate ≥ 1.17 — that is, exceeding the score of this paper's trivial baseline, which uses **only 20-dimensional amino-acid composition and no length**, on the same laboratory hold-out set. Condition (a) alone is insufficient: with large enough n, even +2% is significant, as this paper's own fold 4 shows (1.05 [1.03, 1.07], direction resolvable but magnitude below the trivial baseline). Measured against this two-condition standard, our own results give: **L1 satisfies both (a) and (b) on 1 of 4 held-out center groups, and L2 on 1 of the 4 groups with predictions recorded.** This is not a standard set only for others; it is the result of applying the same measure to ourselves.

## 6 Limitations

1. **Only one stage has usable scale.** After redundancy reduction, only `express` has more than ten thousand genuine negatives; the data required for joint six-stage modelling do not exist in public sources, which is why this paper is positioned as a dataset and benchmark.
2. **The `soluble` stage is unresolved.** 99.9% of its negatives come from an inference rule; the only explicit control subset contains just 10 records, all from a single center (MPSBC) and **all without a His-tag** (n=0 on the His6 side), so the question cannot be tested by design. Within the inferred subset the His6 association reverses direction between centers, which does not support a biological explanation but is not sufficient to settle the matter. In addition, some centers have hundreds of thousands of records on both sides yet zero `soluble` failures, which together with those 10 records supports the view that `soluble` failure is a record-keeping artifact.
3. **The test set systematically excludes proteins within large homology networks** (the same matter as §4, restated here as a limitation because it bounds the applicability of our conclusions). Taking the transitive closure at 30% identity and 80% coverage, the target space forms a single giant connected component (containing 28,646 set-cover clusters); groups accounting for more than 1% are pinned to the training side, as otherwise a single group would consume the entire test set. The cost is that our held-out test sets are biased towards proteins in sparse homology networks, and **this paper cannot assess generalization to multi-domain or large-family proteins.** This affects both absolute scores and possibly the size of the cross-center gap — the direction is unknown and we do not speculate.
4. **A leakage-free temporal hold-out barely exists** (only a few hundred records on the test side) and can serve only as an underpowered weak check.
5. **Group audits along the species dimension cannot yield conclusions.** The 3,194 records on the sequence-split test side are spread over 580 species names, of which only 2 reach the minimum group size of 200 (compare 6 of 32 for the laboratory dimension). Lowering the threshold would manufacture more "groups", but precision@100 on a few dozen samples is noise rather than a finding, so our group audits use laboratory and year as their primary axes and we make no claim about cross-species generalization.
6. **No prospective validation.** All evaluation is retrospective; this work includes no wet-lab experiments.
7. **Our own pipeline data cannot remedy this.** The 4,762 unique designs from the OIH computational center are computational products: task-level failures are job crashes, and treating a structural-confidence threshold as a label would be circular (we measured a per-target maximum F1 of only 0.572 for that metric against real binding).

## 7 Code and data availability

**Repository:** <https://github.com/liugangg/protein-failure-benchmark>

All relative paths cited in this manuscript (`src/…`, `configs/…`, `reports/…`,
`data/processed/splits/…`) are paths within that repository. Two items are deposited
separately rather than in the repository: the unified label table `records.parquet`,
whose size exceeds the per-file limit of common code hosts —
its authoritative size, row count and sha256 are recorded in
`data/processed/records.parquet.prov.json` in the repository, so any copy can be verified
byte-for-byte — and the upstream raw files, which this work does not redistribute at all.
The intermediate artifacts under `data/interim/` are shipped as neither: they are regenerated
by the scripts in `src/`, and only their fingerprints (§9.1) are in the repository, so that a
regenerated copy can be checked against the one used here.

Code is released under **Apache-2.0**. The derived data (unified labels, four frozen splits, center folds) are **not under a single license** — the share-alike terms upstream do not permit us to relicense them uniformly as CC BY 4.0:

| Part | Scope | License |
|---|---|---|
| Code (`src/`, `configs/`) | all | **Apache-2.0** |
| Derived data with share-alike upstream | 946,322 records (39.8%; TargetTrack and Adaptyv), plus the `lab` / `time` splits, `center_folds`, and all labels for the `express` task constructed from them | **CC BY-SA 4.0** |
| Other derived data | 1,433,975 records (60.2%) | **CC BY 4.0** |

**Why this cannot be unified as CC BY 4.0.** PSI TargetTrack is licensed CC BY-SA 4.0, whose share-alike clause requires adapted works to be released under the same or a compatible license. Deriving six-stage labels from its `status` / `stopStatus` fields constitutes adaptation. TargetTrack accounts for 39.7% of the data and is the **sole** source for the `express` primary task and for all cross-center evaluation — the core results of this paper all fall within the share-alike-bound portion. **If the dataset is redistributed as a single work, the strictest upstream terms apply (CC BY-SA 4.0).** Users requiring pure CC BY 4.0 may take the subset excluding TargetTrack, but that subset does not contain the `express` primary task and will not reproduce this paper's main results.

**What this means for downstream users**, stated proactively rather than left to be discovered: if you use the share-alike-bound portion (which contains all labels for the `express` primary task and all cross-center evaluation data, and is unavoidable for reproducing our core results) and redistribute, **your derived work must likewise be released under CC BY-SA 4.0 or a compatible license** — it cannot be changed to CC BY or MIT, cannot be redistributed under a closed license, and mixing in your own data does not exempt the whole. Internal research without redistribution does not trigger share-alike (the attribution clause still applies).

Whether model weights trained on CC BY-SA data constitute an adapted work is legally unsettled; we take no position on it, and do one thing that is actionable instead: **every training and evaluation run machine-records the data sources used** (the `sources` field of `reports/runs/*.json`), from which users can determine whether a given set of weights touched TargetTrack.

**Upstream sources, licenses and citations that must accompany them** (all fields verified against the respective official APIs / READMEs on 2026-10-02, not from memory):

| Source | DOI / address | License |
|---|---|---|
| PSI TargetTrack 2000–2017 | `10.5281/zenodo.821654` | **CC BY-SA 4.0** |
| Tsuboyama et al. 2023 | `10.5281/zenodo.7844779` | CC BY 4.0 |
| ProteinGym v1.3 | github.com/OATML-Markslab/ProteinGym | MIT (code); each assay's copyright remains with the original paper |
| Overath et al. 2025 binder meta-analysis (data deposit) | `10.5281/zenodo.15722219` | CC BY 4.0 |
| Adaptyv Bio EGFR competition | github.com/adaptyvbio/egfr_competition_1 / _2 | ODbL 1.0 (data) + Apache-2.0 (code) |

Two traps are easy to fall into, stated here so that those reproducing this work do not repeat them:

1. **There are two Zenodo records for Tsuboyama.** The same version also exists as record `7992926`, whose `Tsuboyama2023_Dataset2_Dataset3` lacks the `match_aaseq` and `name_original` columns (697,658,024 versus 718,214,782 bytes); reproducing from it yields different labels. This paper uses `7844779` and enforces it with an md5 guard (`src/ingest/verify_raw.py`).
2. **The binder meta-analysis manuscript and its data deposit carry different licenses.** The preprint text (bioRxiv `10.1101/2025.08.14.670059`) is CC BY-NC-ND, while the data deposit (Zenodo `15722219`) is CC BY 4.0. This paper uses only `final_dataset.csv` from the data deposit and is not bound by the NC-ND terms.

The full license layering, the per-item citation entries, and the list of 33 original assay papers that ProteinGym requires to be cited are given in `DATA_AVAILABILITY.md` in the repository. This repository **redistributes no upstream raw file**: `data/raw/` is downloaded from the addresses recorded in `configs/data_sources.yaml`, md5-verified, and used read-only.

**Archival and deposit.** The four frozen splits, `center_folds`, the evaluation code and all reports cited here are released in the companion repository. The unified label table is too large for a code host and will be deposited separately at Zenodo; that deposit's DOI will be added in a revised version of this preprint. Because the deposit will be redistributed as a single work, it will carry CC BY-SA 4.0 in its entirety, for the reason given above; a user requiring a pure CC BY 4.0 subset must reconstruct it by excluding the TargetTrack-derived portion, and that subset does not contain the `express` primary task.

### Funding

This work was funded by Open Intelligence Hub (Suzhou, China). No public or external
grant support was received.

### Competing interests

**The author is the founder and CEO of Open Intelligence Hub**, a commercial
computational drug-discovery company, which funded this work. This is the strongest form
the relationship could take, so the three consequences are stated explicitly rather than
left for the reader to infer:

1. This paper criticizes the evaluation protocols of published methods in a field in
   which the funder operates commercially.
2. Data generated by the funder's own pipeline are used in this work (DS6, 4,762
   designs, used **only** as an unlabeled probe set and contributing to no conclusion;
   see §6, limitation 7).
3. A by-product of this work — the calibration of a structure-confidence threshold
   against designs with real wet-lab outcomes — is directly applicable to the funder's
   pipeline. It is reported in the supplementary material
   (`reports/ds6_oih_inventory.md`) and is **not** used in any conclusion of this paper.

The author declares no other competing interests.

## 8 References

> Titles, authors, and volume/issue/page details for all entries were verified against the Crossref API (first round 2026-10-02; second round 2026-10-05 for the crystallization-propensity line and the separation of the two Bushuiev papers), and all quotations in the text were checked word by word against the publisher's full-text page or the authors' PDF; they are registered in `configs/references.yaml` (including the original wording of each quotation and its verification source). **No entry is written in without verification** — during this process, two DOIs recalled from memory both pointed to entirely unrelated papers (one to a Brazilian biotechnology platform, one to a CHO cell genome), so this discipline is not a formality.

1. Thumuluri V, Martiny HM, Almagro Armenteros JJ, Salomon J, Nielsen H, Johansen AR (2022). NetSolP: predicting protein solubility in Escherichia coli using language models. *Bioinformatics* 38(4): 941–946. doi:10.1093/bioinformatics/btab801
2. Hon J, Marusiak M, Martinek T, Kunka A, Zendulka J, Bednar D, Damborsky J (2021). SoluProt: prediction of soluble protein expression in Escherichia coli. *Bioinformatics* 37(1): 23–28. doi:10.1093/bioinformatics/btaa1102
3. Zhang X, Hu X, Zhang T, Yang L, Liu C, Xu N, Wang H, Sun W (2024). PLM_Sol: predicting protein solubility by benchmarking multiple protein language models with the updated Escherichia coli protein solubility dataset. *Briefings in Bioinformatics* 25(5): bbae404. doi:10.1093/bib/bbae404
4. Price WN, Handelman SK, Everett JK, Tong SN, Bracic A, Luff JD, Naumov V, Acton T, Manor P, Xiao R, Rost B, Montelione GT, Hunt JF (2011). Large-scale experimental studies show unexpected amino acid effects on protein expression and solubility in vivo in E. coli. *Microbial Informatics and Experimentation* 1(1): 6. doi:10.1186/2042-5783-1-6
5. Wang H, Wang M, Tan H, Li Y, Zhang Z, Song J (2014). PredPPCrys: Accurate Prediction of Sequence Cloning, Protein Production, Purification and Crystallization Propensity from Protein Sequences Using Multi-Step Heterogeneous Feature Fusion and Selection. *PLoS ONE* 9(8): e105902. doi:10.1371/journal.pone.0105902
6. Mizianty MJ, Kurgan L (2011). Sequence-based prediction of protein crystallization, purification and production propensity. *Bioinformatics* 27(13): i24–i33. doi:10.1093/bioinformatics/btr229
7. Wang H, Feng L, Zhang Z, Webb GI, Lin D, Song J (2016). Crysalis: an integrated server for computational analysis and design of protein crystallization. *Scientific Reports* 6: 21383. doi:10.1038/srep21383
8. Meng F, Wang C, Kurgan L (2017). fDETECT webserver: fast predictor of propensity for protein production, purification, and crystallization. *BMC Bioinformatics* 18: 580. doi:10.1186/s12859-017-1995-z
9. Zhu Y, Hu J, Ge F, Li F, Song J, Zhang Y, Yu DJ (2021). Accurate multistage prediction of protein crystallization propensity using deep-cascade forest with sequence-based features. *Briefings in Bioinformatics* 22(3): bbaa076. doi:10.1093/bib/bbaa076
10. Wang H, Feng L, Webb GI, Kurgan L, Song J, Lin D (2018). Critical evaluation of bioinformatics tools for the prediction of protein crystallization propensity. *Briefings in Bioinformatics* 19(5): 838–852. doi:10.1093/bib/bbx018
11. Bushuiev A, Bushuiev R, Kouba P, Filkin A, Gabrielova M, Gabriel M, Sedlar J, Pluskal T, Damborsky J, Mazurenko S, Sivic J (2024a). Learning to design protein–protein interactions with enhanced generalization. International Conference on Learning Representations (ICLR) 2024. arXiv:2310.18515
12. Bushuiev A, Bushuiev R, Sedlar J, Pluskal T, Damborsky J, Mazurenko S, Sivic J (2024b). Revealing Data Leakage in Protein Interaction Benchmarks. ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design (GEM)

Citation entries for the data sources (TargetTrack / Tsuboyama / ProteinGym / Overath / Adaptyv) are given in `DATA_AVAILABILITY.md` §5 and `configs/data_sources.yaml`, and were likewise verified.

## 9 Supplementary materials

Every figure quoted in this manuscript is injected programmatically from the frozen artifacts by the generator `src/eval/preprint.py`, which performs cross-checking and leakage verification on each run and refuses to emit the document on any inconsistency; no figure in the text is transcribed by hand.

All of the following are in the repository (<https://github.com/liugangg/protein-failure-benchmark>):

- `reports/gate1_data_inventory.md` — data inventory and bias audit (including the core before/after redundancy-reduction figure)
- `reports/gate2_baselines.md` — baseline comparisons: AF3 ipSAE_min reproduction, SoluProt (with contamination check)
- `reports/splits_and_leakage.md` — construction and leakage verification of the four splits
- `reports/default/stage3_training.md` — L0/L1/L2 training results, adversarial debiasing, bootstrap intervals, methodological traps
- `reports/tag_confound_analysis.json` — all figures from the within-center stratification test
- `reports/soluble_adjudication.json` — the `soluble` adjudication check
- `reports/ds6_oih_inventory.md` — in-house pipeline inventory and threshold calibration
- `reports/paper_data_methods.md` — full Data & Methods
- the `changelog` in `configs/stage3_train.yaml` — timing, rationale and impact of evaluation-rule changes
- `LICENSE` (Apache-2.0) · `LICENSE-DATA-CC-BY-SA-4.0.txt` / `LICENSE-DATA-CC-BY-4.0.txt` · `NOTICE` · `DATA_AVAILABILITY.md` (full account of the license layering)
- `reports/runs/` — data-source list, split hash, hyperparameters and environment versions for every evaluation run
- Zenodo deposit — archived copy of the unified label table (DOI to be added in a revised version of this preprint)

### 9.1 Frozen artifact fingerprints

Every artifact carries a `.prov.json` recording its own `sha256`, byte count and row count, together with **the same three values for all of its upstream inputs**; every downstream script hard-verifies this chain at startup (`require()` in `src/labels/provenance.py`) and exits on any mismatch rather than continuing. This mechanism was forced on us by a real incident: after an upstream rebuild, a downstream script read a stale intermediate artifact and SoluProt's decontaminated subset silently shrank from 2,377 records to 289 — the numbers were produced as usual, with no error raised.

| Artifact | Rows | sha256 (first 16) |
|---|---|---|
| `data/processed/records.parquet` | 2,380,297 | `e887059736b60a98` |
| `data/processed/splits/bind_target_split.parquet` | 3,922 | `5ac1e62546717aa6` |
| `data/processed/splits/center_folds.parquet` | 41 | `7c28d45ac7c2b73a` |
| `data/processed/splits/lab_split.parquet` | 945,718 | `8db25c131e451c88` |
| `data/processed/splits/sequence_split.parquet` | 2,380,297 | `1d5032c5d3f3d4c7` |
| `data/processed/splits/time_split.parquet` | 945,718 | `a08c217a6ed3cb57` |
| `data/interim/pooled_records.parquet` | 2,380,297 | `0056308e0ad5a484` |
| `data/interim/pooled_unique_seqs.fasta` | 708,038 | `d8fc464697129e26` |
| `data/interim/split_groups.parquet` | 354,019 | `ba4ddfbe54fdff87` |

The complete sha256 values and upstream input chains are given in the individual `.prov.json` file of each artifact listed above (9 in total, released with the code; the repository additionally carries fingerprints for intermediate and report artifacts not listed in this table).
