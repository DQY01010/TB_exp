from shutil import copyfile,copytree
import numpy as np
basepth = './BenMalData/data/'
phthisispath = './BenMalData/screenlist/phthisis.npy'
hamartomapath = './BenMalData/screenlist/hamartoma.npy'
inflammatory_pseudopath = './BenMalData/screenlist/inflammatory_pseudo.npy'
infectiouspath = './BenMalData/screenlist/infectious.npy'
chronicTissueInflampath = './BenMalData/screenlist/chronicTissueInflam.npy'

phlst = np.load(phthisispath)
hamalst = np.load(hamartomapath)
infllst = np.load(inflammatory_pseudopath)
infelst = np.load(infectiouspath)
chronlst = np.load(chronicTissueInflamspath)

print(len(phlst),len(hamalst),len(infllst),len(infelst),len(chronlst))

for i in phlst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth:
            copyfile(datapth + "/" + j, "./BenMalData/phthisis/" + i)

for i in hamalst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth:
            copyfile(datapth + "/" + j, "./BenMalData/hamartoma/" + i)
            
            
for i in infllst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth:
            copyfile(datapth + "/" + j, "./BenMalData/inflammatory_pseudo/" + i)
            
                   
for i in infelst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth:
            copyfile(datapth + "/" + j, "./BenMalData/infectious/" + i)   
            
for i in chronlst:
#     print(i)
    casepth = i.split("#")[0]
    nodepth = i.split("#")[1]
    datapth = basepath + casepth
    for j in datapth:
        if j.split("_")[0] == "DATA" and j.split("_")[-1] == nodepth:
            copyfile(datapth + "/" + j, "./BenMalData/chronicTissueInflam/" + i)            
