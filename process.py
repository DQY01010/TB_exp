from shutil import copyfile,copytree
import numpy as np
import os
basepath = './BenMalData/data/RefineProcess/'
phthisispath = './BenMalData/screenlist/phthisis.npy'
hamartomapath = './BenMalData/screenlist/hamartoma.npy'
inflammatory_pseudopath = './BenMalData/screenlist/inflammatory_pseudo.npy'
infectiouspath = './BenMalData/screenlist/infectious.npy'
chronicTissueInflampath = './BenMalData/screenlist/chronicTissueInflam.npy'

phlst = np.load(phthisispath)
hamalst = np.load(hamartomapath)
infllst = np.load(inflammatory_pseudopath)
infelst = np.load(infectiouspath)
chronlst = np.load(chronicTissueInflampath)

print(len(phlst),len(hamalst),len(infllst),len(infelst),len(chronlst))

for i in phlst:
    #print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = os.listdir(basepath + casepth)
    datanpypath = basepath + casepth
    #print(datapth,casepth,nodepth)
    for j in datapth:
       # print(j)
        #print(j.split("_")[0],j.split("_")[-1])
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth.split("_")[-1]:
            copyfile(os.path.join(datanpypath,j), "./BenMalData/phthisis/" + i)

for i in hamalst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = os.listdir(basepath + casepth)
    datanpypath = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth.split("_")[-1]:
            copyfile(os.path.join(datanpypath,j), "./BenMalData/hamartoma/" + i)
            
            
for i in infllst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = os.listdir(basepath + casepth)
    datanpypath = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth.split("_")[-1]:
            copyfile(os.path.join(datanpypath,j), "./BenMalData/inflammatory_pseudo/" + i)
            
                   
for i in infelst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = os.listdir(basepath + casepth)
    datanpypath = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth.split("_")[-1]:
            copyfile(os.path.join(datanpypath,j), "./BenMalData/infectious/" + i)   
            
for i in chronlst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = os.listdir(basepath + casepth)
    datanpypath = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth.split("_")[-1]:
            copyfile(os.path.join(datanpypath,j), "./BenMalData/chronicTissueInflam/" + i)            
