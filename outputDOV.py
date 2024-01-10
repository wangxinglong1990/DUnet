import os

def center_mol2(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    coord = []
    for point in points[7:]:
        if point[0] == '@':
            break
        x1, y1, z1 = [float(n) for n in point.split()[2:5]]
        coord.append([round(x1),round(y1),round(z1)])
    coord = tuple(coord)
    return coord

def center_pdb(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    coord = []
    for point in points[2:]:
        if point.split()[0] == 'CONECT':
            break
        x1=float(point[27:38])
        y1=float(point[38:46])
        z1=float(point[46:54])
        coord.append([round(x1),round(y1),round(z1)])
    coord = tuple(coord)
    return coord

def center_pdb_EN(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    coord = []
    for point in points[0:]:
        if point[0] == 'C':
            break
        x1, y1, z1 = [float(n) for n in point.split()[6:9]]
        coord.append([round(x1),round(y1),round(z1)])
    coord = tuple(coord)
    return coord

def resnet(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    coord = []
    for point in points[7:]:
        if point.split()[0] == '@<TRIPOS>BOND':
            break
        x1=float(point[18:26])
        y1=float(point[27:35])
        z1=float(point[38:46])
        coord.append([round(x1),round(y1),round(z1)])
    coord = tuple(coord)
    return coord

def export_dov_DUnet_coach (dov_file, Path1,Path2):
    f=open(dov_file,'w')
    f.close()
    for pocket in os.listdir(Path1):
        if pocket[-4:]=='mol2':
            i = 0
            predict_pocket = center_mol2(Path1+'\\%s'%pocket)
            actual_pocket = center_pdb(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
            for everysite in predict_pocket:
                if everysite in actual_pocket:
                    i += 1
                else:
                    pass
            overlap = i/(len(predict_pocket) + len(actual_pocket) - i)
            f_dvo=open(dov_file,'a+')
            f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
            f_dvo.close()

        if pocket[-4:] == pocket:
            k = 0
            for pocket_sub in os.listdir(Path1+'\\'+pocket):
                k += 1
                if k == 1:
                    predict_pocket = center_mol2(Path1+'\\'+pocket+'\\%s'%pocket_sub)
                    actual_pocket = center_pdb(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
                    h = 0
                    for everysite in predict_pocket:
                        if everysite in actual_pocket:
                            h += 1
                    overlap = h/(len(predict_pocket) + len(actual_pocket) - k)
                    f_dvo=open(dov_file,'a+')
                    f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
                    f_dvo.close()

def export_dov_Resnet_coach (dov_file, Path1,Path2):
    f=open(dov_file,'w')
    f.close()
    for pocket in os.listdir(Path1):
        if pocket[-4:]=='mol2':
            i = 0
            predict_pocket = center_mol2(Path1+'\\%s'%pocket)
            actual_pocket = center_pdb(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
            for everysite in predict_pocket:
                if everysite in actual_pocket:
                    i += 1
                else:
                    pass
            overlap = i/(len(predict_pocket) + len(actual_pocket) - i)
            f_dvo=open(dov_file,'a+')
            f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
            f_dvo.close()

        if pocket[-4:] == pocket:
            k = 0
            for pocket_sub in os.listdir(Path1+'\\'+pocket):
                k += 1
                if k == 1:
                    predict_pocket = resnet(Path1+'\\'+pocket+'\\%s'%pocket_sub)
                    actual_pocket = center_pdb(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
                    h = 0
                    for everysite in predict_pocket:
                        if everysite in actual_pocket:
                            h += 1
                    overlap = h/(len(predict_pocket) + len(actual_pocket) - k)
                    f_dvo=open(dov_file,'a+')
                    f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
                    f_dvo.close()

def export_dov_DUnet_BU (dov_file, Path1,Path2):
    f=open(dov_file,'w')
    f.close()
    for pocket in os.listdir(Path1):
        if pocket[-4:]=='mol2':
            i = 0
            predict_pocket = center_mol2(Path1+'\\%s'%pocket)
            actual_pocket = center_mol2(Path2+'\\%s'%pocket[:4]+'\\ligand.mol2')
            for everysite in predict_pocket:
                if everysite in actual_pocket:
                    i += 1
                else:
                    pass
            overlap = i/(len(predict_pocket) + len(actual_pocket) - i)
            f_dvo=open(dov_file,'a+')
            f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
            f_dvo.close()

        if pocket[-4:] == pocket:
            k = 0
            for pocket_sub in os.listdir(Path1+'\\'+pocket):
                k += 1
                if k == 1:
                    predict_pocket = center_mol2(Path1+'\\'+pocket+'\\%s'%pocket_sub)
                    actual_pocket = center_mol2(Path2+'\\%s'%pocket[:4]+'\\ligand.mol2')
                    h = 0
                    for everysite in predict_pocket:
                        if everysite in actual_pocket:
                            h += 1
                    overlap = h/(len(predict_pocket) + len(actual_pocket) - k)
                    f_dvo=open(dov_file,'a+')
                    f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
                    f_dvo.close()

def export_dov_Resnet_BU (dov_file, Path1,Path2):
    f=open(dov_file,'w')
    f.close()
    for pocket in os.listdir(Path1):
        if pocket[-4:]=='mol2':
            i = 0
            predict_pocket = center_mol2(Path1+'\\%s'%pocket)
            actual_pocket = center_mol2(Path2+'\\%s'%pocket[:4]+'\\ligand.mol2')
            for everysite in predict_pocket:
                if everysite in actual_pocket:
                    i += 1
                else:
                    pass
            overlap = i/(len(predict_pocket) + len(actual_pocket) - i)
            print('i',i,overlap,str(pocket[:4])+' '+str(overlap))
            f_dvo=open(dov_file,'a+')
            f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
            f_dvo.close()

        if pocket[-4:] == pocket:
            k = 0
            for pocket_sub in os.listdir(Path1+'\\'+pocket):
                k += 1
                if k == 1:
                    predict_pocket = center_mol2(Path1+'\\'+pocket+'\\%s'%pocket_sub)
                    actual_pocket = center_mol2(Path2+'\\%s'%pocket[:4]+'\\ligand.mol2')
                    h = 0
                    for everysite in predict_pocket:
                        if everysite in actual_pocket:
                            h += 1
                    overlap = h/(len(predict_pocket) + len(actual_pocket) - k)
                    f_dvo=open(dov_file,'a+')
                    f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
                    f_dvo.close()

def export_dov_EN (dov_file, Path1,Path2):
    f=open(dov_file,'w')
    f.close()
    for pocket in os.listdir(Path1):
        if pocket[-4:]=='mol2':
            i = 0
            predict_pocket = center_mol2(Path1+'\\%s'%pocket)
            actual_pocket = center_pdb_EN(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
            for everysite in predict_pocket:
                if everysite in actual_pocket:
                    i += 1
                else:
                    pass
            overlap = i/(len(predict_pocket) + len(actual_pocket) - i)
            print('i',i,overlap,str(pocket[:4])+' '+str(overlap))
            f_dvo=open(dov_file,'a+')
            f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
            f_dvo.close()

        if pocket[-4:] == pocket:
            k = 0
            for pocket_sub in os.listdir(Path1+'\\'+pocket):
                k += 1
                if k == 1:
                    predict_pocket = center_mol2(Path1+'\\'+pocket+'\\%s'%pocket_sub)
                    actual_pocket = center_pdb_EN(Path2+'\\%s'%pocket[:4]+'\\ligand.pdb')
                    h = 0
                    for everysite in predict_pocket:
                        if everysite in actual_pocket:
                            h += 1
                    overlap = h/(len(predict_pocket) + len(actual_pocket) - k)
                    #print('k',k,overlap,str(pocket[:4])+' '+str(overlap))
                    f_dvo=open(dov_file,'a+')
                    f_dvo.write(str(pocket[:4])+' '+str(overlap)+'\n')
                    f_dvo.close()
                    
if __name__ == "__main__":
    export_dov_DUnet_coach()
    export_dov_Resnet_coach()
    export_dov_DUnet_BU()
    export_dov_Resnet_BU()
    export_dov_EN ()
