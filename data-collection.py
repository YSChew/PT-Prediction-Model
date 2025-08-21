import pandas as pd
import requests

PT_PROF_MAX = 140
PT_COURSE_MAX = 168
UMDIO_COURSE_MAX = 899

def process_reviews(review_array):
    '''
    Processes list of reviews received. Removes unneccesary reviews and returns list of cleaned reviews.

    Arguments:
    review_array: Array of reviews received from PlanetTerp API professors/staff
    '''
    cleaned_reviews = list()

    for review in review_array:
        # Reviews from 2020 and 2021 removed due to P/F nature of courses
        review_yr = review['created'][:4]

        if((review_yr not in {'2020','2021'}) and (int(review_yr) >= 2018)):
            cleaned_reviews.append([review['course'],review['review'],review['expected_grade']])
        
    return cleaned_reviews

# PlanetTerp Staff Data
def pterp_staff_data():
    '''
    Retreives and stores staff data from PlanetTerp API then combines and converts data into .csv file

    '''
    df_array = list()

    for i in range(PT_PROF_MAX):
        try:
            response = requests.get("https://planetterp.com/api/v1/professors",params={"reviews":'true',
                                                                                        "limit":100, "offset":i*100})
            data = response.json()

            if(not("error" in data) and data):
                curr_profdf = pd.json_normalize(data)

                curr_profdf["reviews"] = curr_profdf["reviews"].apply(lambda x: process_reviews(x))

                # Filter out staff without reviews or non teaching staff
                curr_profdf = curr_profdf[(curr_profdf["reviews"].map(len) > 0) & (curr_profdf["courses"].map(len) > 0)]

                df_array.append(curr_profdf) # Add back to array
                print(f"Professor Set {i} data retreived")
            elif("error" in data):
                print(f"Error: {response.json()["error"]}")
            elif(not(data)):
                print("Range Exceeded")
        except Exception as e:
            print(f"Error: {e}")

    staff_df = pd.concat(df_array, ignore_index=True)
    staff_df.to_csv("PTstaff_data.csv", index=False)
    print("PlanetTerp staff data retreived and .csv created.")
        
def pterp_course_data():
    '''
    Retreives and stores course data from PlanetTerp API then combines and converts data into .csv file

    '''
    df_array = list()

    for i in range(PT_COURSE_MAX):
        try:
            # Reviews are redundant, already accessed from professors
            response = requests.get("https://planetterp.com/api/v1/courses", params={"reviews":'false', 
                                                                                     "limit":100, "offset":i*100})
            data = response.json()

            if(not("error" in data) and data):
                curr_coursedf = pd.json_normalize(data)
                curr_coursedf["professors"] = curr_coursedf["professors"].apply(lambda x: tuple(set(x))) # Remove duplicate professors names

                df_array.append(curr_coursedf) # Add back to array
                print(f"PT Courses Set {i} data retreived")
            elif("error" in data):
                print(f"Error: {response.json()["error"]}")
            elif(not(data)):
                print("Range Exceeded")

        except Exception as e:
            print(f"Error: {e}")

    course1_df = pd.concat(df_array, ignore_index=True)
    course1_df.to_csv("PTcourse_data.csv", index=False)
    print("PlanetTerp course data retreived and .csv created.")

def umdio_course_data():
    '''
    Retreives and stores course data from UMD.io API then combines and converts data into .csv file

    '''
    df_array = list()

    for i in range(UMDIO_COURSE_MAX):
        try:
            response = requests.get("https://api.umd.io/v1/courses", params={"per_page":100, "sort":"course_id",
                                                                          "page":i, "semester":"201801|geq"})
            data = response.json()

            if(not("error" in data) and data):
                curr_coursedf = pd.json_normalize(data)

                curr_coursedf = curr_coursedf.drop(labels=["name", "relationships.formerly", 
                                                           "relationships.restrictions", 
                                                           "relationships.additional_info", 
                                                           "relationships.also_offered_as"], axis=1)

                df_array.append(curr_coursedf) # Add back to array
                print(f"UMD.io Courses Set {i} data retreived")
            elif("error" in data):
                print(f"Error: {response.json()["error"]}")
            elif(not(data)):
                print("Range Exceeded")

        except Exception as e:
            print(f"Error: {e}")

    course2_df = pd.concat(df_array, ignore_index=True)
    course2_df.to_csv("UMDIOcourse_data.csv", index=False)
    print("UMDIO course data retreived and .csv created.")

umdio_course_data()
pterp_course_data()