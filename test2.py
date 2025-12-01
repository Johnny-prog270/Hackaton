
minx = 133
miny = 133

def rgb(i):
    if i%3==0:
        return "R"
    elif i%3==1:
        return "G"
    else:
        return "B"

pixels = ["pixel"+str((i//3)+1)+rgb(i) for i in range(0,minx*miny*3)]

print(pixels[0:12])
print(len(pixels))