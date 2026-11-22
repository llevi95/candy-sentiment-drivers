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
judgement calls flagged. It is used to test the gummy effect and to answer the
cookie-versus-gummy debate; see `data/derived/README.md` for the coding rule.

## Approach

This is the summary; [PROCESS_CODE.ipynb](PROCESS_CODE.ipynb) runs each step
as a notebook, with the current results and how to read them
(`uv run --with jupyter jupyter lab PROCESS_CODE.ipynb`).

1. **Clean.** Drop the two coins, normalise column names, join the gummy
   coding, and validate that attributes are 0/1 and levels lie in 0 to 1.
2. **Main model.** Ordinary least squares of win rate on the nine attributes
   plus sugar and price level (12 parameters including the intercept).
   Coefficients read as *points of win rate*. Almost every candy is chocolate
   or fruity, so both coefficients compare with the 9 candies that are
   neither; the bundle also gives chocolate minus fruity directly. Inference uses HC3
   heteroskedasticity-robust standard errors and t-based 95% intervals.
3. **Diagnostics.** Variance inflation factors, leave-one-out R² (an honest
   out-of-sample fit), Holm correction across the 11 main effects, and a simple
   with/without comparison per attribute (Welch t-test).
4. **Robustness.** The same model on logit-transformed win rate, case-resampling
   bootstrap intervals (fixed seed), the model refitted with the coins
   included (the largest shift in any coefficient is reported in
   `appendix/data_notes.md`), and a gummy check that adds the gummy attribute,
   refitted once more with the gummy judgement calls recoded as not gummy, and
   the model refitted without the four Reese's candies, the largest brand
   family among the top candies, to see which effects rest on one brand.
5. **Interactions.** Three interactions were chosen on product grounds
   (chocolate with its most common partners): chocolate × peanuty/almondy,
   chocolate × crisped/wafer and chocolate × bar. Each is added on its own and
   kept only if its 95% interval excludes zero and leave-one-out R² does not
   get worse. Interactions the data cannot identify are reported as such, and
   `appendix/interaction_support.csv` counts the candies behind each one.
6. **Recommendation.** Every combination of attributes is enumerated and
   filtered to **supported profiles**: realistic ones (not chocolate and fruity
   together; hard excludes bar; bar excludes pluribus) that differ from at least
   one real candy in at most one attribute. Each is predicted with the final
   model at the median sugar and price level (neither is a confirmed driver,
   so neither is used as a lever), with a 95%
   interval for the average and a 95% prediction interval for a single launch,
   capped to the 0 to 100 scale.

   The profile with the highest prediction is *not* recommended. It switches
   on every attribute with a positive coefficient, however weak the evidence,
   so it is largely picked by noise. Instead each attribute gets an evidence
   tier: **driver** (survives the Holm correction), **tentative** (its own 95%
   interval excludes zero, but it does not survive) or **open** (the data does
   not decide; a free design choice). The **primary profile** is the best
   supported profile made only of drivers. Every profile whose predicted gap
   to the primary profile has an interval that includes zero is flagged as
   tied with it, and the ties are counted; the rest are marked above or below
   it. These intervals are uncorrected, so "above" is weaker evidence than a
   driver: chocolate with peanut/almond is the one profile above, and that
   lead does not survive dropping the Reese's candies.

   The primary profile comes with a market check on its prediction: the
   count, mean and range of win rate of every existing candy within one
   attribute of it. The **benchmark candies** name three of them, those with
   the lowest, median and highest win rate, so the examples span that range.
7. **Cookie versus gummy.** The debate is answered with direct differences
   from the model that includes the gummy attribute: a chocolate product, and
   a chocolate product with a cookie component, each minus a fruity gummy,
   with a 95% interval. Every cookie candy in the data is chocolate, so a
   non-chocolate cookie cannot be assessed. Because these scores come from the
   gummy model, chocolate scores slightly differently here than as the primary
   profile; compare products within the section only.

### Modelling uses only NumPy and SciPy

The brief allows only NumPy and SciPy for modelling. All estimators (OLS, HC3
standard errors, VIFs, leave-one-out R², the logit variant, the bootstrap,
predictions and the Holm adjustment) are implemented with NumPy in
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
uv run make test    # lint with ruff, then run the test suite
```

With pip, activate the virtual environment and drop `uv run`. Without `make`,
run `python -m candy --out outputs`, `python -m ruff check src tests` and `python -m pytest`.

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
| `figures/ranked_profiles.png` | Top supported profiles against the recommended one, ties highlighted |
| `figures/cookie_vs_gummy.png` | Chocolate, chocolate + cookie and fruity gummy, with each lead over the gummy |
| `appendix/vif.csv` | Variance inflation factor per predictor |
| `appendix/interaction_support.csv` | How many candies have both, or only one, of each interaction's attributes |
| `appendix/gummy_coding.csv` | The approved gummy coding with notes |
| `appendix/data_notes.md` | Plain-language notes on exclusions and data limits |

Figures are 1920 × 1080 PNGs, ready for 16:9 slides.

## Repository layout

```
data/raw/            source data and its documentation
data/derived/        hand-coded gummy attribute
src/candy/
  data.py            loading, cleaning, validation, gummy join
  estimators.py      OLS and inference (NumPy/SciPy only)
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
- Predictions come from an additive model. Profiles that stack many
  attributes, or sit an attribute away from any real candy, are extrapolations
  and are flagged as such in the bundle; this is why the top-ranked profile is
  not the recommendation.
- Neither chocolate product clearly beats a gummy: both leads have intervals
  that include zero. The data favours chocolate but does not settle the
  debate.
- The data confirms one driver (chocolate). Most supported profiles cannot be
  told apart from the recommended one, so the remaining attributes are design
  choices, not findings.
