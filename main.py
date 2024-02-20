import pandas as pd
from sklearn.model_selection import train_test_split
import nearest_neighbor

dataset1=pd.read_csv("project1_dataset1.txt", header=None, delimiter='\t')
dataset2=pd.read_csv("project1_dataset2.txt", header=None, delimiter='\t')

def preprocessing():
    replacement_dict={'Present':1, 'Absent':0}
    dataset2[4]=dataset2[4].replace(replacement_dict)

preprocessing()

d1_train, d1_test = train_test_split(dataset1, test_size=0.1, random_state=13)

#print(nearest_neighbor.nearest_neighbor(d1_train, d1_test, 3))