# ArtDossier — Quantitative Evaluation: Analysis and Critical Reflection

_Generated from evaluation run, June 2026. n=50 paintings (25 broad / 25 expert), 1 AI judge pass per painting, 10 human-evaluated paintings._

---

## 1. Overview of Metrics Used

Four quantitative metrics were applied to Panel 3 output (the eight-layer dossier) against a ground truth corpus entry per painting. Each metric captures something different, and their divergences are as informative as their values.

**Token F1** (set-based overlap) computes precision and recall over the unordered bag of lowercased tokens shared between hypothesis and reference. It rewards exact vocabulary match and penalises paraphrase, domain-shifted synonyms, and any structural reordering. A dossier that synthesises source content using its own language will score low even if its claims are entirely accurate. F1 is the bluntest instrument here.

**METEOR** adds stemming and synonym matching via a lexical database, making it more forgiving of morphological variation and near-synonyms. It sits between F1 and semantic metrics — better at capturing meaning-preserving paraphrases, but still sensitive to the specific words used rather than to the truth content of claims.

**BERTScore F1** uses contextual embeddings (RoBERTa-large) to match tokens by meaning rather than by string identity. A dossier sentence can receive high BERTScore alignment even if it shares no words with the reference, as long as the underlying semantic content is similar. This makes it the most appropriate single metric for assessing whether the dossier is _about_ the right subject, without penalising it for having a different authorial register.

**CLIPScore** is categorically different from the above three: it measures alignment between the visual image and the Panel 1 text description, not between the dossier and the corpus. A high CLIPScore indicates that the VLM's initial visual description accurately corresponds to the actual painting — it is a measure of the _perceptual accuracy_ of the first panel, not the discursive quality of the dossier itself. Including it in a quantitative table alongside F1 and BERTScore risks implying they measure the same thing, when in practice they are evaluating different links in the pipeline chain.

---

## 2. Quantitative Results: What the Numbers Show

### 2.1 Token F1 by Corpus Source

Panel 3-only F1 (hypothesis = full dossier text, reference = primary corpus section):

| Source | Mean F1 Precision | Mean F1 Recall | Mean F1 |
|---|---|---|---|
| Leiden Collection | 0.22 | 0.28 | 0.249 |
| Rijksmuseum | 0.07 | 0.10 | 0.067 |
| WGA | 0.10 | 0.14 | 0.121 |

Leiden Collection scores markedly higher. This is not surprising: Leiden entries contain extended scholarly essays with rich art-historical vocabulary — the same register the dossier pipeline is prompted to write in. Rijksmuseum entries tend toward brief institutional descriptions or Dutch-language provenance records, producing a systematic vocabulary mismatch even when the dossier content is substantively accurate. WGA entries vary widely in depth.

The F1 range (0.07–0.25) is low by most NLP benchmarks, but the ceiling for this task is itself low: a dossier that accurately synthesises a source will use different sentence structures, interpretive framings, and connecting arguments. Perfect F1 would imply near-verbatim copying — which would be a worse outcome than the current scores.

### 2.2 BERTScore

BERTScore F1 values ranged from approximately 0.72 to 0.82, clustering tightly around 0.77–0.80. The correlation between token F1 and BERTScore F1 (see Fig. I) is positive but moderate (r ≈ 0.4–0.5), which is expected: two metrics that measure different dimensions of text similarity should not be perfectly correlated. Cases with high BERTScore but low token F1 represent the most interesting region — dossiers that are semantically on-target but use their own language rather than the corpus's. These are, arguably, the best outputs.

### 2.3 CLIPScore

CLIPScore (image–Panel 1 alignment) averaged approximately 62–66 across the evaluation set, with Leiden paintings scoring somewhat higher. These values reflect that the VLM's visual description is reasonably well-aligned with the actual painting's visual content. However, interpreting CLIPScore as evidence of pipeline quality requires care: it only validates the first panel's perceptual step, not the reasoning chain that follows. A painting that is visually simple and easily described (a still life with well-known objects) will score high on CLIPScore regardless of the quality of its dossier.

---

## 3. Retrieval Quality and Its Downstream Effect

### 3.1 Hit Rate

The correct painting title appeared in the top-12 retrieved chunks for **17 out of 50 paintings (34%)**. Expert mode retrieved the correct title for 8/25 paintings; Broad mode for 9/25. The difference is small, but Expert mode's exclusion of WGA (a lower-authority, less specific source) did not substantially improve title-level retrieval — suggesting that retrieval difficulty is primarily a function of title ambiguity and image recognition uncertainty rather than corpus quality.

### 3.2 Retrieval Hit vs Dossier Quality

Paintings where the correct title was retrieved scored significantly higher with the AI judge: **mean 50.8/80 vs 35.9/80** for non-retrieved paintings — a gap of approximately 15 points. This is the single most predictive factor in dossier quality. When retrieval succeeds, the pipeline has access to factually grounded artist biography, provenance, and iconographic context; when it fails, Panel 3 must rely on general parametric knowledge and whatever adjacent sources were retrieved.

This finding has a clear epistemic interpretation: the pipeline's quality is bounded by retrieval accuracy. It is not evidence of a retrieval-dependent architecture failing — it is confirmation that the pipeline does what it is designed to do: synthesise retrieved sources. The question of what happens when retrieval fails (does the dossier fail gracefully or confidently?) is addressed under qualitative findings.

### 3.3 Top RAG Score and Quality

Top RAG chunk scores were overwhelmingly in the "high" bin (≥0.70): 42 of 48 judge-evaluated paintings had a top chunk score above 0.70. This confirms that the reranker is finding highly relevant content — but "relevant" does not mean "correct painting." A chunk from a different painting by the same artist (e.g., another Rembrandt portrait) can score high on semantic similarity without being the specific work depicted. The distinction between _thematically relevant_ and _specifically correct_ retrieval is one the reranker cannot make without image-level grounding.

---

## 4. AI Judge Results (n=48)

### 4.1 Score Distribution

AI judge totals ranged from **4 to 73 out of 80**, with a mean of **41.2** and a median of approximately 43. The distribution is roughly normal with a slight left skew, indicating that most paintings received mid-range scores with a handful of notably poor performers. The minimum of 4 represents a near-total failure case; the maximum of 73 represents a best-case scenario where retrieval succeeded, genre was appropriate, and the pipeline's synthesis was substantively grounded.

### 4.2 Layer-Level Patterns (AI, n=48)

| Layer | Mean (AI, n=48) |
|---|---|
| L1 Socio-Historical | 5.42 |
| L2 Artist's Background | 4.31 |
| L3 Materiality & Techniques | 5.21 |
| L4 Appearance & Critique | 4.71 |
| L5 Symbols & Metaphors | 4.62 |
| L6 Artistic Traditions | 4.85 |
| L7 Horizons of Meaning | **6.00** |
| L8 Ambiguity | **5.98** |

Layer 2 (Artist's Background) is consistently the weakest. This is structurally predictable: without retrieving the correct artist, biographical content must be either fabricated or withheld, and the rubric penalises both. Layer 7 (Horizons) and Layer 8 (Ambiguity) score highest — partly because reflexive meta-commentary and epistemic hedging are things the pipeline generates fluently even without accurate retrieval, and partly because the judge prompt rewards these layers for process rather than content.

### 4.3 Genre Patterns

Still life paintings scored highest on average (mean ≈ 53–62), while history/allegory and single-figure history paintings scored lowest (mean ≈ 14–32). This pattern reflects two compounding factors. First, genre paintings and still lifes have well-documented iconographic conventions that are well-represented in all three corpus sources, making retrieval thematically rich even when specific retrieval fails. Second, history paintings often require very specific knowledge of particular commissions, biblical episodes, or named figures — knowledge that is difficult to derive from adjacent corpus entries.

### 4.4 Broad vs Expert Mode

Broad mode scored higher on average (42.0) than Expert mode (36.4). Broad mode includes WGA — the largest corpus source — which increases the probability of retrieving contextually relevant material. Expert mode's restriction to Leiden and Rijksmuseum sources is an intentional design choice that trades retrieval volume for source authority. The score difference does not mean Broad mode is better; it means that access to more sources (even lower-authority ones) statistically improves average dossier quality when measured by an AI judge trained on general scholarly discourse. Whether this translates to higher epistemic trustworthiness is a different question.

---

## 5. Human vs AI Judge Comparison (n=10)

### 5.1 Systematic Divergence

The most consistent finding across all 10 shared paintings is that the AI judge scores substantially higher than the human evaluator:

| | Mean total (/80) |
|---|---|
| Human | 33.1 |
| AI Judge | 46.3 |

The AI judge awards, on average, **13 more points per painting**. This divergence is not random: across 9 of the 10 paintings, the AI scores higher than the human. The single exception is Battle of Gibraltar in 1607 (H=25, AI=14) — a history painting where the pipeline failed to retrieve relevant material and the AI judge penalised it heavily, while the human was perhaps more lenient about the genre's inherent difficulty.

This systematic gap is worth naming carefully. It does not simply mean the AI judge is "lenient." The AI judge was explicitly prompted to be charitable (to treat dossiers as reference, to avoid penalising for low confidence, to not over-criticise when retrieval fails). The human evaluator, working from the same dossiers and rubric, was harder. This may reflect: (a) the AI judge's difficulty in identifying subtly wrong or ungrounded claims that a domain-experienced reader detects immediately; (b) the AI judge rewarding fluent academic prose even when content is shallow; or (c) the human's more conservative interpretation of what "10" means on a 0–10 rubric. All three explanations are probably partially true.

### 5.2 Layer-Level Divergences

| Layer | Human mean (n=10) | AI mean (n=10) | Difference (H−AI) |
|---|---|---|---|
| L1 Socio-Historical | 3.2 | 5.7 | −2.5 |
| L2 Artist's Background | 4.6 | 4.0 | +0.6 |
| L3 Materiality | 4.6 | 5.6 | −1.0 |
| L4 Appearance | 4.8 | 4.9 | −0.1 |
| L5 Symbols | 4.6 | 5.1 | −0.5 |
| L6 Traditions | 3.7 | 5.2 | −1.5 |
| L7 Horizons | 3.6 | 6.1 | −2.5 |
| L8 Ambiguity | 4.0 | 6.1 | −2.1 |

The largest human–AI gaps are in L7 (Horizons of Meaning) and L8 (Ambiguity), followed by L1 (Socio-Historical). The AI judge awards these meta-critical layers almost routinely at 6–7/10; the human evaluator scores them at 3–4/10. This is a significant methodological finding: an AI judge cannot reliably evaluate whether reflexive commentary is genuine critical distance or performative hedging. The dossier pipeline produces confident-sounding uncertainty language by design — and the AI judge rewards it as though it were substantive.

L2 (Artist's Background) is the only layer where humans score slightly higher than the AI. Human evaluators appear more willing to credit biographical attempts that are partially correct; the AI judge penalises L2 more consistently when the specific artist is not identified.

### 5.3 Highest and Lowest Scoring Paintings (Human)

Highest human total: **Christ on the Mount of Olives** (50/80). This religious subject from the Leiden Collection — where the pipeline retrieved relevant Leiden scholarly sources — received the most generous human assessment across nearly all layers, particularly L5 (Symbols, 9/10) and L2 (Artist, 8/10).

Lowest human totals: **Italian Landscape with Mule Driver** (18/80) and **Still Life with Cheeses** (19/80). Both are paintings where genre provides limited iconographic depth: a landscape and a simple food still life offer few symbolic or historical hooks. The human evaluator found the dossiers thin on substance regardless of retrieval quality.

---

## 6. Critical Reflections

### 6.1 What Does Scoring High Actually Mean?

High token F1 means the dossier shares vocabulary with the reference — but this could result from either accurate synthesis _or_ from over-reliance on the corpus text itself. A pipeline that quotes or closely paraphrases its sources would score well on F1 and poorly on the evaluative criteria the rubric is designed to measure (groundedness, situatedness, reflexivity). F1 is most useful as a lower-bound sanity check: very low F1 across all sources suggests the dossier is operating entirely outside the relevant semantic field.

High BERTScore suggests the dossier is semantically on-target without necessarily copying vocabulary — it is the most epistemically honest of the three text-overlap metrics. But it too cannot distinguish between a dossier that makes accurate claims and one that makes plausible-sounding claims in the same vocabulary field.

High CLIPScore validates the visual perception step but is orthogonal to interpretive quality. A painting of an easily recognisable subject (a Vanitas skull, a figure in period costume) will be described accurately by almost any VLM; a complex crowd scene or unusual mythological subject will not. CLIPScore should be treated as a measure of task difficulty at the visual layer, not a measure of overall pipeline quality.

High AI judge score is the most ambiguous of all metrics. It is produced by a large language model that was trained on scholarly discourse and prompted to be epistemically charitable. It correlates with retrieval success (correct title in RAG → +15 points), which suggests it is measuring something real about pipeline capability. But the systematic 13-point gap above human scores, and the disproportionate reward for L7/L8, suggest the AI judge is at least partially rewarding rhetorical sophistication rather than substantive accuracy. Neither the AI judge nor the human evaluator can be treated as ground truth; their disagreements are themselves data.

### 6.2 What the Metrics Cannot Measure

None of the quantitative metrics above can assess whether a dossier is _useful_ to a museum visitor who has no prior knowledge of the painting. They cannot measure whether a dossier that misidentifies the artist still opens productive interpretive questions. They cannot assess whether a high-scoring dossier on a well-retrieved painting contains subtle anachronisms or category errors. And they cannot capture what the thesis calls "groundedness" and "situatedness" — qualities that require a reader with domain knowledge to evaluate.

This is not a failure of the evaluation design. It is a feature of the research question. The pipeline is not trying to produce a factually verified catalogue entry; it is trying to produce a layered interpretive text that enriches engagement with a painting under conditions of uncertainty. Metrics that punish uncertainty and reward vocabulary overlap with authoritative sources are structurally misaligned with this goal. The qualitative judge — despite its own limitations — is the more appropriate primary instrument.

### 6.3 Ethical Note on Automation Bias

Presenting AI judge scores alongside human scores creates a risk of automation bias: readers of this thesis may weight the AI score more heavily because it is produced at scale (48 paintings vs 10) and presented with apparent precision. The correct interpretation is the reverse: the human score on 10 paintings is more trustworthy precisely _because_ it reflects domain-situated human judgment, and its systematic disagreement with the AI judge on L7 and L8 should prompt scepticism about what the AI judge is actually measuring in those layers.

---

## 7. Summary of Key Findings

1. **Retrieval accuracy is the strongest predictor of dossier quality** — paintings with correct title retrieved by RAG score ~15 points higher on the AI judge (50.8 vs 35.9).

2. **The pipeline generates fluent reflexivity (L7, L8) more reliably than it generates accurate factual grounding (L2, L1)** — the AI judge rewards this; the human evaluator does not.

3. **Human evaluators score consistently lower than the AI judge** (mean gap: 13 points), with the largest divergences on meta-critical layers. This is a methodological finding about AI-as-judge, not a finding about the pipeline itself.

4. **Genre shapes quality independently of retrieval** — still life and group portrait subjects score highest; history and allegory subjects score lowest. Genre difficulty is a real variable that any future evaluation should control for.

5. **Expert mode trades retrieval volume for source authority** — its lower average score reflects fewer total sources, not lower pipeline quality. Whether authoritative-but-sparse or broad-but-noisy retrieval is preferable depends on what the dossier is for.

6. **Token F1, BERTScore, and CLIPScore measure different things** and should not be collapsed into a single "score." Their divergences (especially BERTScore high / F1 low) identify the most interesting dossiers — ones that are semantically on-target but not verbally derivative of sources.

7. **No single metric measures what the thesis most cares about** — the capacity of the dossier to produce critical aesthetic engagement under conditions of epistemic uncertainty. That capacity remains primarily assessable by close reading.

---

_Figures referenced: figA–figJ in evaluation/graphs/_
