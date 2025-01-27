import re
import pandas as pd
from operator import itemgetter
from unidecode import unidecode

def find_matches_positions(search_text: str, names_list: list, base_regex: str, **kwargs) -> list:
    """
    Find the start and end positions for all the mateches of the values in `values_list`
    in the `search_text`

    Parameters
    ----------
    search_text : str
        text in which to search for matches
    names_list : list
        list of values to find
    base_regex : str
        regex used to search for the values

    Returns
    -------
    list
        list with the positions of all the matches found
    """
    found_matches_pos = {}
    for value_str in names_list:
        # compile regex for current value 
        value_regex = re.compile(base_regex.format(name_value=value_str))
        
        # add matches start and end
        for match in value_regex.finditer(search_text):
            start_pos = match.start()
            end_pos = match.end()

            add_pos_data = True
            if end_pos in found_matches_pos:
                # found string was already matched by another value

                if start_pos > found_matches_pos[end_pos]["start"]:
                    # found string is smaller
                    add_pos_data = False

            if add_pos_data:
                # add new found match
                # or replace current value with the bigger string
                found_matches_pos[end_pos] = {
                    "start": start_pos,
                    "end": end_pos,
                    "place_name": value_str
                }

    return found_matches_pos.values()


def define_place_type(row: pd.Series, search_text: str) -> pd.Series:
    """
    Define if the found place is an estate or a city and save the corresponding data

    Parameters
    ----------
    row : pd.Series
        row with the matches info for the estate and the citi
    search_text : str
        original text in which the name was found

    Returns
    -------
    pd.Series
        row with the updated data (`type`, `start`, `place_name`)
    """

    if row.start_city < row.start_estate:
        # estate is contained in the city, keep the city name
        row["type"] = "city"
    elif row.start_estate < row.start_city:
        # city is contained in the estate, keep the estate name
        row["type"] = "estate"
    else:
        # both names are the same length, check if there is a "municipios" or "estados"
        # mention in the text before the name
        
        city_match = re.search(f"[mM]unicipios de .*? {row.place_name_city}", search_text[:row.end])
        city_start = -1 if city_match is None else city_match.start()

        estate_match = re.search(f"[eE]stados de .*? {row.place_name_city}",  search_text[:row.end])
        estate_start = -1 if estate_match is None else estate_match.start()

        # city if the "municipios" string is closer, else estate
        row["type"] = "city" if city_start > estate_start else "estate"

    # add correct data
    row["start"] = row[f"start_{row.type}"]
    row["place_name"] = row[f"place_name_{row.type}"]

    return row


def replace_places_by_flag(text: str, inegi_replacements: dict) -> str:
    """
    Replace all the names in the `inegi_replacements` dict by a placeholder

    Parameters
    ----------
    text : str
        text to clean
    inegi_replacements : dict
        dictionary with the needed data for the replacements 
        (list of names, placeholder, regex used for the search)

    Returns
    -------
    str
        text with the placeholders for the names
    """
    if re.search(r"[ ,](([eE]stados?|[mM]unicipios?) de|Ciudad |Distrito )", text) is None:
        # text doesn't contain text places
        return text
    
    # lower case and remove accents from text
    search_text = unidecode(text.lower())
    
    cities_pos = find_matches_positions(search_text, **inegi_replacements["city"])
    cities_df = pd.DataFrame(cities_pos)

    estates_pos = find_matches_positions(search_text, **inegi_replacements["estate"])
    estates_df = pd.DataFrame(estates_pos)

    if cities_df.shape[0] > 0 and estates_df.shape[0] > 0:
        # get replacements that matched both
        duplicate_replacements = cities_df.merge(estates_df, how="inner", on=["end"], suffixes=("_city", "_estate"))

        if duplicate_replacements.shape[0] > 0:

            # define which is the correct replacement
            duplicate_replacements = duplicate_replacements.apply(define_place_type, search_text=search_text, axis=1)

            # keep only the needed columns
            duplicate_replacements = duplicate_replacements[["start", "end", "type"]]

            # get replacements that didn't have a duplicate
            cities_df["type"] = "city"
            unique_cities = cities_df.loc[~cities_df.end.isin(estates_df.end)]

            estates_df["type"] = "estate"
            unique_estates = estates_df.loc[~estates_df.end.isin(cities_df.end)]

            # build final df
            total_replacements = pd.concat([unique_cities, unique_estates, duplicate_replacements])
        else:
            # no duplicates found concat original data frames
            cities_df["type"] = "city"
            estates_df["type"] = "estate"
            
            total_replacements = pd.concat([cities_df, estates_df])

        # cast columns to the correct type
        total_replacements["start"] = total_replacements.start.apply(int)
        total_replacements["end"] = total_replacements.end.apply(int)

    elif cities_df.shape[0] > 0:
        # only city replacements found
        total_replacements = cities_df
        total_replacements["type"] = "city"

    elif estates_df.shape[0] > 0:
        # only estate replacements found
        total_replacements = estates_df
        total_replacements["type"] = "estate"
    
    elif cities_df.shape[0] == 0 and estates_df.shape[0] == 0: 
        # no replacements found
        total_replacements = pd.DataFrame(columns=["start"])

    # sort values
    total_replacements = total_replacements.sort_values(by=["start"])
    
    # apply replacements
    last_end = 0
    clean_text = ""
    for _, row in total_replacements.iterrows():
        upper_case = re.search(r"^[ ,]([eE]stados?|[mM]unicipios?|[A-ZÁÉÍÓÚ])", text[row.start:row.end])
        if upper_case is None:
            # false positive
            # the original text doesn't start with an upper case, keep original text
            clean_text = clean_text + text[last_end:row.end]
        else:
            # replace match for given replacement
            clean_text = clean_text + text[last_end:row.start] + inegi_replacements[row.type]["replacement"]

        last_end = row.end
    
    # add end of text
    clean_text = clean_text + text[last_end:]
        
    return clean_text


def replace_string(text, replacement_regex, replacement):
    
    replacements_pos = []
    # add matches start and end
    for match in re.finditer(replacement_regex, text):
        replacements_pos.append((match.start(), match.end()))
            
    # sort replacements by appereance order
    replacements_pos = sorted(replacements_pos, key=itemgetter(0))
    
    # apply replacements
    last_end = 0
    clean_text = ""
    for start, end in replacements_pos:
        # replace match for given text
        clean_text = clean_text + text[last_end:start] + replacement

        last_end = end
    
    # add end of text
    clean_text = clean_text + text[last_end:]
        
    return clean_text