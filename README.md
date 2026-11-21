# Candy sentiment drivers

Which product attributes make consumers prefer one candy over another, and what
attribute profile should a new store-brand candy have?

The purchasing team is split between a cookie-based product and a gummy. This
repository answers the question from market research data rather than opinion,
and produces every number and figure used in the accompanying presentation with
one command.

## The data

`data/raw/candy-data.csv` holds 85 market entries. Two of them, "One dime" and
"One quarter", are coins that were included in the survey as decoys; they are
not candies and are excluded, leaving **83 candies**. For each candy the data
gives:

- nine yes/no **attributes**: chocolate, fruity, caramel, peanuty/almondy,
  nougat, crisped rice/wafer/cookie, hard, bar, and pluribus (many per pack);
- **sugar level** and **price level**, each a percentile rank within the
  dataset (0 to 1);
- **win rate**: the share of roughly 269,000 head-to-head matchups the candy
  won, where a respondent saw two random candies and picked one. Win rate
  measures *relative preference*, not sales or satisfaction.

`data/derived/gummy.csv` adds a hand-coded **gummy** attribute (soft, elastic
gummy chew), reviewed and approved before use, with a note per candy and
judgement calls flagged. It is used only as a supplementary check; see
`data/derived/README.md` for the coding rule.

## Approach

1. **Clean.** Drop the two coins, normalise column names, join the gummy
   coding, and validate that attributes are 0/1 and levels lie in 0 to 1.
2. **Main model.** Ordinary least squares of win rate on the nine attributes
   plus sugar and price level (12 parameters including the intercept).
   Coefficients read as *points of win rate*. Inference uses HC3
   heteroskedasticity-robust standard errors and t-based 95% intervals.
3. **Diagnostics.** Variance inflation factors, leave-one-out R² (an honest
   out-of-sample fit), Holm correction across the 11 main effects, and a simple
   with/without comparison per attribute (Welch t-test).
4. **Robustness.** The same model on logit-transformed win rate, case-resampling
   bootstrap intervals (fixed seed), the model refitted with the coins
   included (the largest shift in any coefficient is reported in
   `appendix/data_notes.md`), and a gummy check that adds the gummy attribute.
5. **Interactions.** Three interactions were declared before any results were
   seen: chocolate × peanuty/almondy, chocolate × crisped/wafer and
   chocolate × bar. Each is added on its own and kept only if its 95% interval
   excludes zero and leave-one-out R² does not get worse. Interactions the data
   cannot identify are reported as such.
6. **Recommendation.** Every combination of attributes is enumerated and
   filtered to **supported profiles**: realistic ones (not chocolate and fruity
   together; hard excludes bar; bar excludes pluribus) that differ from at least
   one real candy in at most one attribute. Each is predicted with the final
   model at a fixed sugar level (the median, or the favourable quartile if
   sugar's interval excludes zero) and the median price level, with a 95%
   interval for the average and a 95% prediction interval for a single launch,
   capped to the 0 to 100 scale. The **primary profile** is the best supported
   profile; the **fruit-flavoured profile** is the best one with fruity = 1; the
   gap between them answers the cookie-versus-gummy debate. Each comes with
   **benchmark candies**: the existing candies closest to it.

### Modelling uses only NumPy and SciPy

The brief allows only NumPy and SciPy for modelling. All estimators (OLS, HC3
standard errors, VIFs, leave-one-out R², the logit variant, the bootstrap,
predictions and the Holm adjustment) are written by hand in
`src/candy/estimators.py`, with distributions and t-tests from SciPy. pandas
and matplotlib are used only for data handling and plotting. A test fails if
any other library is imported, and the estimators are tested against known
answers (SciPy's `linregress`, published reference values, brute-force
leave-one-out refits and textbook formulas).

## Setup

Python 3.12 is required. Either route installs the exact versions pinned in the
lockfile.

**With [uv](https://docs.astral.sh/uv/)** (recommended):

```sh
uv sync
```

**With pip:**

```sh
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned versions; also installs this package
```

`requirements.txt` is exported from `uv.lock` (`make requirements`); a test
checks that the two agree.

## Run

```sh
uv run make all     # rebuild the results bundle, figures and appendix
uv run make test    # run the test suite
```

With pip, activate the virtual environment and drop `uv run`. Without `make`,
run `python -m candy --out outputs` and `python -m pytest`.

`python -m candy --help` lists the options: input paths (`--raw`, `--gummy`),
the output folder (`--out`, default `outputs`) and the bootstrap seed
(`--seed`). The same seed always gives an identical results bundle.

## Outputs

Everything lands in `outputs/`. The folder is committed so the results can be
read without running anything; `make all` rebuilds it:

| Path | Contents |
| --- | --- |
| `results.json` | The results bundle: every number the presentation quotes, from sample sizes to the recommendation, plus the run seed |
| `figures/win_rate_distribution.png` | How win rates are spread across the 83 candies |
| `figures/coefficients.png` | Effect of each attribute and level on win rate, Holm survivors highlighted |
| `figures/attribute_comparison.png` | Average win rate of candies with vs without each attribute |
| `figures/robustness.png` | HC3, bootstrap and logit intervals side by side |
| `figures/ranked_profiles.png` | Top supported profiles with intervals |
| `figures/recommendation_gap.png` | Primary vs fruit-flavoured profile and the gap between them |
| `appendix/vif.csv` | Variance inflation factor per predictor |
| `appendix/gummy_coding.csv` | The approved gummy coding with notes |
| `appendix/data_notes.md` | Plain-language notes on exclusions and data limits |

Figures are 1920 × 1080 PNGs, ready for 16:9 slides.

## Repository layout

```
data/raw/            source data and its documentation
data/derived/        hand-coded gummy attribute
src/candy/
  data.py            loading, cleaning, validation, gummy join
  estimators.py      hand-written OLS and inference (NumPy/SciPy only)
  profiles.py        profile enumeration, support, ranking, benchmarks
  figures.py         slide-ready figures
  labels.py          business-facing names and wording
  pipeline.py        end-to-end run: data in, bundle, figures, appendix out
  __main__.py        command-line entry point (python -m candy)
tests/
  test_estimators.py estimators checked against independent known answers
  test_pipeline.py   whole pipeline on the real data, checked for invariants
  test_constraints.py library allowlist and requirements/lockfile agreement
```

## Limits worth knowing

- 83 candies is a small sample, and some attributes are rare (7 candies have
  nougat, 7 have crisped rice/wafer), so many intervals are wide.
- Win rate is relative: a candy's score depends on what it was matched
  against, and it says nothing about sales, price willingness or repeat
  purchase.
- The recommended profiles are predictions from an additive model; profiles
  that stack many attributes, or sit an attribute away from any real candy,
  are extrapolations and are flagged as such in the bundle.
