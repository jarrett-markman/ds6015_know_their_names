# loading important packages
import numpy as np
import pandas as pd
import sys
#!{sys.executable} -m pip install splink
import splink.comparison_library as cl
from splink import comparison_level_library as cll
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

# Load in mentions data
mentions_new = pd.read_csv('data/mentions.csv')

# Filter mentions
mentions = mentions_new[
    (mentions_new['source'] == 'ALB_CN_1870')|
    (mentions_new['source'] == 'ALB_CN_1880')
]

# Rename the mentions_id column to unique_id
mentions = mentions.rename(columns={"mention_id": "unique_id"})

# Remove duplicates
mentions = mentions.drop_duplicates(subset=['unique_id'])

# Drop unused cols.
mentions = mentions.drop(columns=['source', 'confidence', 'race', 'occupation', 'created','full_name','first_name','last_name'])

# Normalize all middle names to be uppercase
mentions['middle_name'] = mentions['middle_name'].str.upper()

# Drop more cols.
mentions = mentions.drop(columns=['maiden_name','death_year','legal_status','is_enslaver','location_id'])

# Separate mentions data into 1870 and 1880 records
mentions_1870 = mentions[mentions['source_year'] == 1870]
mentions_1880 = mentions[mentions['source_year'] == 1880]

#creating custom levels for birth_year
birth_year_comparison = cl.CustomComparison(
    output_column_name = "birth_year",
    comparison_levels = [
        cll.NullLevel("birth_year"),
        cll.CustomLevel(
            "abs(birth_year_l - birth_year_r) = 0",
            label_for_charts = "Exact Birth Year"),
        cll.CustomLevel(
            "abs(birth_year_l - birth_year_r) <= 1",
            label_for_charts = "Within 1 Year"),
        cll.CustomLevel(
            "abs(birth_year_l - birth_year_r) <= 2",
            label_for_charts = "Within 2 Years"),
        cll.CustomLevel(
            "abs(birth_year_l - birth_year_r) <= 5",
            label_for_charts = "Within 5 Years"),
        cll.ElseLevel()
    ]
)

# the ordering of the following matters so john and joan would fall into "within 1 letter" rather than "same inital letter"
first_name_comparison = cl.CustomComparison(
    output_column_name = "norm_first_name",
    comparison_levels = [
        cll.NullLevel("norm_first_name"),
        cll.CustomLevel(
            "levenshtein(norm_first_name_l, norm_first_name_r) = 0",
            label_for_charts = "Exact First Name"),
        cll.CustomLevel(
            "levenshtein(norm_first_name_l, norm_first_name_r) = 1",
            label_for_charts = "Within 1 Letter"),
        cll.CustomLevel(
            "levenshtein(norm_first_name_l, norm_first_name_r) = 2",
            label_for_charts = "Within 2 Letters"),
        cll.CustomLevel(
            "substr(norm_first_name_l, 1, 1) = substr(norm_first_name_r, 1, 1)",
            label_for_charts = "Same First Initial"),
        cll.ElseLevel()
    ]
)

# specifies the linkage model
settings = {
    "link_type": "link_only",

    "blocking_rules_to_generate_predictions": [
        "l.norm_first_name = r.norm_first_name and l.nysiis_last_name = r.nysiis_last_name",
        "l.norm_race = r.norm_race and l.gender = r.gender and abs(l.birth_year - r.birth_year) <= 5",
    ],

    "comparisons": [
        birth_year_comparison,
        cl.ExactMatch("norm_race"),
        cl.ExactMatch("gender"),
        first_name_comparison,
    ]
}

# Apply the model to 1870 and 1880 census mentions
linker = Linker([mentions_1870, mentions_1880], settings, db_api=DuckDBAPI())

deterministic_rules = [
   block_on("gender","norm_race", "norm_first_name")
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.90)
# room for ground truth; look at metrics for the gorund truth data and see how can use this for the recall metric

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

training_blocking_rule = block_on("gender")
training_session_gender_race = (
    linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)
)

training_blocking_rule = block_on("norm_race", "norm_first_name")
training_session_race = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule
)

df_predictions = linker.inference.predict(threshold_match_probability=0.2)

# Export predictions
df_predictions.to_csv('data/predictions.csv', index=False)