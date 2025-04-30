

# importing libraries
import pandas as pd
import numpy as np
import os

df= pd.DataFrame()

def read_file():
    global df
    file_path = input(f"Can you please path the file location ? ").strip()
    if os.path.exists(file_path):
        file_name = file_path.split("\\")[-1] # get the last part of the path 'file.csv'
        print(f"File {file_name} is successufuly read")
        df = pd.read_csv(file_path)
        return df
    else:
        print(f"Please make sure of the file location")



def return_dataframe():
    return df



read_file()
