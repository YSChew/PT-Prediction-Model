import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
import string
import re

IDX_REVIEWS = 2

tagger = TextClassifier.load("sentiment-fast")

school_map = {
    "AGNR": ["AGNR", "AGST", "ANSC", "AREC", "ENST", "ENTM", "NFSC", "PLSC", "INAG"],
    "ARCH": ["ARCH", "RDEV", "HISP", "URSP", "CPSS"],
    "ARHU": ["AAAS", "AAST", "CINE", "ARAB", "ARTH", "ARTT", "CLAS", "CHIN", "CMLT", "COMM","DANC", "ENGL", "FREN",
        "GERM", "GERS", "GREK", "HEBR", "IMMR", "ITAL", "JAPN", "JWST", "KORA", "LATN", "LGBT", "LING", 
        "MUED", "MUSC", "MUSP", "PHIL", "PORT", "RELS", "RUSS", "SPAN", "THET", "WGSS", "WMST", "SLLC", "TDPS"],
    "BSOS": ["AASP", "ANTH", "CCJS", "ECON", "GEOG", "GVPT", "PSYC", "SOCY", "BSST", "LACS", "USLT", "SLAA", "SURV", "CPBE", "CPSN"],
    "BMGT": ["BMGT", "BUAC", "BUDT", "BUFN", "BULM", "BUMK", "BUMO", "BUSI", "BUSM", "BUSO", "EMBA", "BDBA", "BMSO", "BMIN"],
    "CMNS": ["AMSC", "ASTR", "BCHM", "BIOL", "BSCI", "CBMG", "CHEM", "CHPH", "CLFS", "CMSC", "GEOL", "MATH", "PHYS", 
             "STAT", "AOSC", "BIOE", "BIOI", "BIOM", "BIPH", "BISI", "NACS", "NEUR", "MOCB", "MSML", "MSQC", "CPMS", "CPSF"],
    "EDUC": ["EDCP", "EDHD", "EDHI", "EDMS", "EDPS", "EDSP", "TLPL", "TLTC", "CHSE"],
    "ENGR": ["ENAE", "ENBC", "ENCE", "ENCH", "ENCO", "ENEB", "ENED", "ENEE", "ENES", "ENFP", "ENMA", "ENME", "ENMT", "ENNU", "ENPM", "ENRE", "ENSE", "MEES"],
    "INFO": ["INST", "INFM", "DATA", "INFO", "CPDJ"],
    "JOUR": ["JOUR"],
    "SPP": ["PLCY"],
    "SPH": ["HLTH", "HLSA", "HLSC", "MIEH", "EPIB", "FMSC", "PHSC", "SPHL"],
    "UGST": ["UNIV", "HONR", "HNUH", "HDCC", "HHUM", "HGLO", "HESI", "LEAD", "FIRE", "IDEA", "MLAW", "CPCV", "CPET", 
             "CPGH", "CPJT", "CPPL", "CPSA", "CPSD", "CPSG", "CPSP", "CRLN", "GEMS", "HBUS", "SMLP", "VIPS", "WEID"],
    "OTHER": ["ARMY", "NAVY", "OURS", "PEER", "XPER", "ABRM", "BSCV", "BSGC", "FGSM", "HACS", "HEIP",
        "IMDM", "ISRL", "MAIT", "MITH", "MLSC", "NIAP", "NIAS", "NIAV", "PHPE", "SMLP", "UMEI", "SUMM", "EXST"]
}

def assign_schl(code):
    for school in school_map:
        if code in school_map[school]:
            return school
    return "OTHER"

def generate_score(review):
    # Function generates sentiment analysis score for review
    parts = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", review) if s.strip()] # Split review into sentences
    rev_len = 1 / len(review) # Weight of each word
    comment_score = 0

    sentences = [Sentence(i) for i in parts]
    tagger.predict(sentences, mini_batch_size=4) 
    # Analysis done in batches due to time and memory constraints
    
    for section in sentences:
        sentiment = -1 if section.labels[0].value == "NEGATIVE" else 1
        # Sum score based on cumulative total of considering each word's score
        comment_score += (len(section.text) * rev_len * section.score * sentiment)

    return comment_score

def convert_grade(grade):
    # Convert expected grade into respective GPA points
    grade_dict = {"A+": 4, "A": 4, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, 
                  "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "D-": 0.7}
    if grade in grade_dict.keys():
        return float(grade_dict[grade])
    else:
        return 0
    
def profreview_data(prof_data, course_data):
    # Function that extracts reviews data and returns a df for each review
    course_data = course_data.set_index('course_id')
    review_data = pd.DataFrame()
    data_rows = []

    for prof in prof_data.itertuples():
        for review in prof[IDX_REVIEWS]:
            if (review[0] == None) or (review[0] not in course_data.index):
                school_value = "PROF"
                if (review[0] == None):
                    school_value = "PROF"
                else:
                    school_value = "OTHER"
                avg_gpa = 0
                course_lvl = 0
                credits = 0
            else:
                target_row = course_data.loc[review[0]]
                school_value = target_row["school"]
                avg_gpa = target_row["average_gpa"]
                course_lvl = target_row["course_lvl_bin"]
                credits = target_row["credits"]

            new_row = {'professor_id': prof[4], 
                        "review_score": review[1],
                        "expected_grade": review[2],
                        "school": school_value,
                        "course_avgGPA": avg_gpa,
                        "course_lvl": course_lvl,
                        "credits": credits
                        }
            data_rows.append(new_row)

    review_data = pd.DataFrame(data_rows)

    print(len(review_data))
    review_data["expected_grade"] = review_data["expected_grade"].map(convert_grade)
    review_data["review_score"] = review_data["review_score"].map(generate_score)
    return review_data

def bin_course(course_id):
    # Course binning function based on level
    course_id = course_id.strip(string.ascii_letters)
    if course_id[0] == "0":
        # No degree courses
        return 0
    elif int(course_id[0]) in range(1,5):
        # UG courses
        return 1 
    elif int(course_id[0]) in range(5,9):
        # Grad courses excl. doctoral courses
        return 2
    elif course_id[0] == "9":
        # Doctoral courses
        return 3
    
def create_courselist(reviews_arr):
    # Create list of courses taught for each professor
    courses = []
    for review in reviews_arr:
        if (review[0] not in courses) and (review[0] != None):
            courses.append(review[0])
    
    return courses

def convert_grademethod(grading_array):
    # Find grading methods for array of reviews
    if "Regular" in grading_array:
        return 2
    elif ("Pass-Fail" in grading_array) or ("Sat-Fail" in grading_array):
        return 1
    else:
        return 0