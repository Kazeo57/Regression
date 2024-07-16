import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

data=pd.read_excel("Insurance-data.xlsx")
#print(data.head(10))

"""
if str(col.dtype)[:5]=="float":
    plt.hist(col)
elif str(col.dtype)[:8]=="category" or str(col.dtype)=="object" or str(col.dtype)=="bool":
    plt.bar(col)
"""
plt.bar(data["region"])


#sns.histplot(data['bmi'])
plt.show()

#print(str(data['bmi'].dtype)[:5])
