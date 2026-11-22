# Candy sentiment drivers

Which candy attributes make consumers prefer one candy over another, and what
should a new store-brand candy look like? The analysis uses market research on
83 candies (85 entries minus two joke coins) and their win rate in about
269,000 head-to-head consumer choices. It also settles, as far as the data
allows, the purchasing team's debate between a cookie product and a gummy.

Modelling uses only NumPy and SciPy. One command rebuilds every number and
figure. [PROCESS_CODE.ipynb](PROCESS_CODE.ipynb) walks through the analysis
step by step (`uv run --with jupyter jupyter lab PROCESS_CODE.ipynb`).

## Setup

Python 3.12. Both routes install the versions pinned in `uv.lock`.

```sh
uv sync                            # with uv (recommended)
```

```sh
python3.12 -m venv .venv           # with pip
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```sh
uv run make all     # rebuild outputs/
uv run make test    # ruff lint, then the test suite
```

With pip, drop `uv run`. `python -m candy --help` lists the options.

## Outputs

Everything lands in `outputs/`, committed so it can be read without running
anything.

| Path | Contents |
| --- | --- |
| `results.json` | Every number the presentation quotes |
| `figures/*.png` | Slide-ready charts (1920 × 1080) |
| `appendix/` | VIFs, interaction support, gummy coding, data notes |

## Layout

```
data/raw/            source data and its documentation
data/derived/        hand-coded gummy attribute and its coding rule
src/candy/
  data.py            loading, validation, gummy join
  estimators.py      OLS and inference (NumPy/SciPy only)
  profiles.py        product profiles, ranking, benchmarks
  figures.py         charts
  labels.py          business-facing names
  pipeline.py        end-to-end run
  __main__.py        command line (python -m candy)
tests/               estimators, pipeline and project constraints
PROCESS_CODE.ipynb   step-by-step walk-through
```
