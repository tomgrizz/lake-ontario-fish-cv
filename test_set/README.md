# Test set

Frozen evaluation sets for measuring model performance across versions.

## Rule

**Once a test set version is frozen, it is never modified.** New labels, corrections,
or additions go into a new version (`test_set_v2`, etc.). This is what makes
cross-version comparisons defensible.

## Structure

Each test set version lives in its own directory with:

- The labeled track records
- A construction manifest (sampling strategy, who reviewed, when)
- Stratification metadata (species × site × season × condition cells)
