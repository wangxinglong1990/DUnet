struct = []
score = []
energy = []
f=open(r'score.sc')
scores = f.readlines()
f.close()

c = 0
for k in scores:
    c += 1
    if c > 2:
        if len(k.split()) > 41:
            score.append(k.split()[4])
            struct.append(k.split()[41])
            energy.append(k.split()[1])

f = open('score.txt','w')
f.close()
f = open('score.txt','a+')
for i,j,k in zip(score,struct,energy):
    f.write('%s %s %s\n'%(i,k,j))
f.close()
