import os
import numpy as np

folder = os.listdir("Datasets")
data =[]
lable = 0 
lable_map = {}
y_lables = []
for i in folder:
    if i == "x.npy" or i == "y.npy":
        continue
    lable_map[lable] = i
    total_data = 0
    for j in os.listdir("Datasets/"+i):
        numpy_data = np.load("Datasets/"+i+"/"+j)
        shape  = numpy_data.shape
        total_data += shape[0]
        for k in range(0,shape[0]):
            avg_x = numpy_data[k,0,:,0].mean()
            avg_y = numpy_data[k,0,:,1].mean()
            avg_z = numpy_data[k,0,:,2].mean()
            # for t in range(0,shape[1]):
            #     numpy_data[k,t,:,0] -= avg_x
            #     numpy_data[k,t,:,1] -= avg_y
            #     numpy_data[k,t,:,2] -= avg_z
            numpy_data[k,:,:,0] -= avg_x
            numpy_data[k,:,:,1] -= avg_y
            numpy_data[k,:,:,2] -= avg_z
        data.extend(numpy_data)
        y_lables.extend([lable]*shape[0])
        # data.append([numpy_data,lable])
    lable+=1

x = np.array(data)
y = np.array(y_lables)

print(x.shape)
print(y.shape)
print(lable_map)
print(y)
np.save("Datasets/x",x)
np.save("Datasets/y",y)
