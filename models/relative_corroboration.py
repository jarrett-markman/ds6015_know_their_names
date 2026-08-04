# Import packages
import pandas as pd
import numpy as np

# Read in predictions (output of em_model.py)
preds = pd.read_csv("data/predictions.csv")
# Read in assertions data
assertions = pd.read_csv("data/assertions.csv")

# Take a subset of columns in preds (probability and unique ids)
# Theoretically, you only need the probability and the unique ids for boosting
matches = preds[['match_probability', 'unique_id_l', 'unique_id_r']]
# Write out matches to CSV for better storage
matches.to_csv("data/matches.csv", index=False)

# Get all relatives for each mention_id (both directions)
relatives_l = assertions[['subject_id', 'object_id']].rename(columns={'subject_id': 'mention_id', 'object_id': 'relative_id'})
relatives_r = assertions[['object_id', 'subject_id']].rename(columns={'object_id': 'mention_id', 'subject_id': 'relative_id'})
# Combine both directions
all_relatives = pd.concat([relatives_l, relatives_r]).drop_duplicates()
# Build a dict: mention_id -> set of relative ids
relative_sets = all_relatives.groupby('mention_id')['relative_id'].apply(set).to_dict()

# Look at the data
print(matches.head())
print(all_relatives.head())

# Map relative sets onto matches for each side
matches['relatives_l'] = matches['unique_id_l'].map(relative_sets)
matches['relatives_r'] = matches['unique_id_r'].map(relative_sets)

# Build match lookup dict
match_lookup = matches.set_index(['unique_id_l', 'unique_id_r'])['match_probability'].to_dict()

# Build a function to get relatives_match_probability
def get_relatives_match_probability(row):
    # Get the two sets of relatives
    relatives_l = row['relatives_l']
    relatives_r = row['relatives_r']
    # If the left side (1870 census) has no relatives, return nan
    if not isinstance(relatives_l, set) or not isinstance(relatives_r, set): # Also nan if no right relatives (nothing to compare)
        return np.nan
    # Create an empty list to hold best match probabilities
    best_probs = []
    # For each 1870 relative of all 1870 relatives
    for rel_l in relatives_l:
        # Get all non-nan match probabilities
        probs = [
            p for rel_r in relatives_r
            if not np.isnan(p := match_lookup.get((rel_l, rel_r), np.nan))
        ]
        # If no valid matches, append nan
        best_probs.append(max(probs) if probs else np.nan)
    # Return mean of best match probabilities (or nan)
    return np.nanmean(best_probs) if best_probs else np.nan

# Apply get_relatives_match_probability to matches
matches['relatives_match_probability'] = matches.apply(get_relatives_match_probability, axis=1)

# Write matches to a parquet file (helps for storage) w/ relatives_match_probability
matches_subset = matches[['unique_id_l', 'unique_id_r', 'match_probability', 'relatives_match_probability']]
matches_subset.to_parquet("data/boosted_preds.parquet", index=False)

# Remove all NAs for relatives_match_probability
matches_no_nas = matches.copy().dropna(subset=['relatives_match_probability'])

# Display matches with relatives_match_probability
print(matches_no_nas.head(n=20))

# Look at Dabney Johnson
# Filter for his mention id, sort on match_probability then relatives_match_probability (descending), and display top 10
print(
    matches_no_nas[matches_no_nas['unique_id_l'] == 'ALB-CN-1870-1688']
    .sort_values(['match_probability', 'relatives_match_probability'], ascending=False)
    .head(n=10)
)