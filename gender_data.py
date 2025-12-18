# gender_data.py - Static dictionary for common English names and their genders

GENDER_MAP = {
    # Female Names (F)
    "elizabeth": "FEMALE",
    "jane": "FEMALE",
    "lydia": "FEMALE",
    "kitty": "FEMALE",
    "mary": "FEMALE",
    "charlotte": "FEMALE",
    "caroline": "FEMALE",
    "georgiana": "FEMALE",
    "anne": "FEMALE",
    "emma": "FEMALE",
    "catherine": "FEMALE",
    "lucy": "FEMALE",
    "sophia": "FEMALE",
    "maria": "FEMALE",
    "fanny": "FEMALE",
    "harriet": "FEMALE",
    "isabella": "FEMALE",
    "louisa": "FEMALE",
    "augusta": "FEMALE",
    
    # Male Names (M)
    "darcy": "MALE",
    "bingley": "MALE",
    "wickham": "MALE",
    "collins": "MALE",
    "fitzwilliam": "MALE",
    "george": "MALE",
    "charles": "MALE",
    "edward": "MALE",
    "henry": "MALE",
    "william": "MALE",
    "john": "MALE",
    "robert": "MALE",
    "thomas": "MALE",
    "james": "MALE",
    "richard": "MALE",
    "philip": "MALE",
    "edmund": "MALE",
    "frank": "MALE",
    "frederick": "MALE",
    "arthur": "MALE",
}

def get_static_gender_map():
    return GENDER_MAP
