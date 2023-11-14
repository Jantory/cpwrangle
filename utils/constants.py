"""Constants."""

import os
from pathlib import Path

DATASET_PATH = os.environ.get("DATASET_PATH", Path("data/datasets").resolve())

DATA2TASK = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "entity_matching",
    f"{DATASET_PATH}/data_imputation/Buy": "data_imputation",
    f"{DATASET_PATH}/data_imputation/Restaurant": "data_imputation",
    f"{DATASET_PATH}/error_detection/Hospital": "error_detection",
}

IMPUTE_COLS = {
    f"{DATASET_PATH}/data_imputation/Buy": "manufacturer",
    f"{DATASET_PATH}/data_imputation/Restaurant": "city",
}

MATCH_PROD_NAME = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "Product",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "Product",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "Product",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "Product",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "Product",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "Song",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "Product",
}

# Dropping happens before renaming
DATA2DROPCOLS = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": [],
    f"{DATASET_PATH}/entity_matching/structured/Beer": ["Style", "ABV"],
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": [],
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": [],
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": [],
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": ["CopyRight"],
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": [
        "category",
        "price",
        "brand",
    ],
    f"{DATASET_PATH}/data_imputation/Buy": [],
    f"{DATASET_PATH}/data_imputation/Restaurant": [],
    f"{DATASET_PATH}/error_detection/Hospital": [],
}

DATA2COLREMAP = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": {},
    f"{DATASET_PATH}/entity_matching/structured/Beer": {
        "id": "id",
        "Beer_Name": "name",
        "Brew_Factory_Name": "factory",
        "Style": "style",
        "ABV": "ABV",
    },
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": {},
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": {},
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": {},
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": {
        "id": "id",
        "Song_Name": "name",
        "Artist_Name": "artist name",
        "Album_Name": "album name",
        "Genre": "genre",
        "Price": "price",
        "CopyRight": "CopyRight",
        "Time": "time",
        "Released": "released",
    },
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": {},
    f"{DATASET_PATH}/data_imputation/Buy": {},
    f"{DATASET_PATH}/data_imputation/Restaurant": {},
    f"{DATASET_PATH}/error_detection/Hospital": {},
}


DATA2INSTRUCT = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "Are Song A and Song B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/data_imputation/Buy": "Who is the manufacturer? apple, sony, lg electronics?",
    f"{DATASET_PATH}/data_imputation/Restaurant": "What is the city? san fransisco, new york, denver?",
    f"{DATASET_PATH}/error_detection/Hospital": "Is there a x spelling error? Yes or No?",
}

DATA2SUFFIX = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": " Are Product A and Product B the same?",
    f"{DATASET_PATH}/entity_matching/structured/Beer": " Are Product A and Product B the same?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": " Are Product A and Product B the same?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": " Are Product A and Product B the same?",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": " Are Product A and Product B the same?",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": " Are Song A and Song B the same Song?",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": " Are A and B the Same?",
    f"{DATASET_PATH}/data_imputation/Buy": " Who is the manufacturer?",
    f"{DATASET_PATH}/data_imputation/Restaurant": " What is the city?",
    f"{DATASET_PATH}/error_detection/Hospital": "?",
}

DATA2EXAMPLE_SUBKEY_ATTR = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "manufacturer_A",
    f"{DATASET_PATH}/entity_matching/structured/Beer": None,
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "venue_A",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "venue_A",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": None,
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": None,
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": None,
    f"{DATASET_PATH}/data_imputation/Buy": None,
    f"{DATASET_PATH}/data_imputation/Restaurant": None,
    f"{DATASET_PATH}/error_detection/Hospital": "col_name",
}
