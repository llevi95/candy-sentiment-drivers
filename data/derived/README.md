# Derived data

## gummy.csv

Hand-coded gummy attribute for the 83 candies (coins excluded), drafted
from public product knowledge and reviewed before use. The source data
has no texture attribute, so this supports a supplementary check of the
gummy question alongside the fruity attribute.

| Column    | Meaning                                                         |
|-----------|-----------------------------------------------------------------|
| `candy`   | Candy name, exactly as in `data/raw/candy-data.csv`             |
| `gummy`   | 1 if the candy has a soft, elastic gummy chew, else 0           |
| `unclear` | 1 if the coding is a judgement call (see `note`), else 0        |
| `note`    | Short reason for the coding                                     |

Coding rule: texture decides, not ingredient. Starch-based chews with a
gummy texture (e.g. Sour Patch Kids, Swedish Fish) count as gummy;
taffy, licorice, fruit chews, shelled jelly candies (Mike & Ike, Chewey
Lemonhead), caramel and gum do not, even when the recipe contains
gelatin (e.g. Starburst).
