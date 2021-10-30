import numpy as np
import json
import openpyxl
import math
import os

wb = openpyxl.load_workbook("./Tuberculosis_classify.xlsx")
sheet = wb.worksheets[0]

datapath = "./BenMalData/data/tb_210301_refine/"
file_dir = os.listdir(datapath)


for row in sheet.iter_rows():
    for cell in row:
        if cell.coordinate.split("E")[0] == '':
            cell_value = cell.value
            caseid = cell_value.split("/")[-1]
            print(caseid)
            for file in file_dir:
                if file.split("#")[0] == caseid:
                    nodule = np.load(datapath+file)
                    print(nodule.shape)