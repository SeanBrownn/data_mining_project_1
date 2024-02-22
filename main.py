import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset1=pd.read_csv("project1_dataset1.txt", header=None, delimiter='\t')
dataset2=pd.read_csv("project1_dataset2.txt", header=None, delimiter='\t')

# normalizes columns of dataframe
def normalize(df):
    scaler=MinMaxScaler()
    df[df.columns]=scaler.fit_transform(df[df.columns])


# preprocesses both datasets before we run classification algorithms
def preprocessing():
    replacement_dict={'Present':1, 'Absent':0}
    dataset2[4]=dataset2[4].replace(replacement_dict)

    normalize(dataset1)
    normalize(dataset2)

    dataset1.rename(columns={dataset1.columns[-1]: 'class'}, inplace=True)
    dataset2.rename(columns={dataset2.columns[-1]: 'class'}, inplace=True)



preprocessing()