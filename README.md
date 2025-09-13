# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import numpy as np
import pandas as pd
data=pd.read_csv("SAMPLEIDS.csv")
data
```


<img width="1030" height="779" alt="image" src="https://github.com/user-attachments/assets/0fa955a9-88d1-497e-827d-69506f116aa8" />


```
data.head(4)
```

<img width="1142" height="184" alt="Screenshot 2025-09-13 205456" src="https://github.com/user-attachments/assets/338bee9b-c4c6-41c8-8b88-0007cff33ae0" />

```
data.tail(7)
```

<img width="1215" height="299" alt="image" src="https://github.com/user-attachments/assets/e364b7f4-0409-48e7-9b6f-bd3d5f7967a1" />

```
data.isnull()
```

<img width="903" height="789" alt="image" src="https://github.com/user-attachments/assets/8e0550a6-d9cd-40cd-9cd1-43c365d12e14" />

```
data.notnull()
```

<img width="912" height="787" alt="image" src="https://github.com/user-attachments/assets/b092760f-39fe-4023-a703-b056ff1fcd4f" />

```
data.isnull().sum()

```

<img width="240" height="511" alt="image" src="https://github.com/user-attachments/assets/1f24533c-f8db-4de2-a9a3-ab4e721c4c8c" />

```
data.isnull().any()
```

<img width="271" height="513" alt="image" src="https://github.com/user-attachments/assets/9b311d2b-ec8d-4bce-a051-b139ff3c8be1" />

```
data.dropna(axis=1)
```

<img width="407" height="790" alt="image" src="https://github.com/user-attachments/assets/17dfe5d4-0cb6-4a3d-b96b-01898681feca" />

```
data.dropna(axis=0)
```

<img width="1092" height="524" alt="image" src="https://github.com/user-attachments/assets/0ce6d7d1-cec0-41dd-b93f-8b44f854b808" />

```
data.fillna(0)
```

<img width="1129" height="787" alt="image" src="https://github.com/user-attachments/assets/c71aa30e-ecd0-4f97-8c1a-3f3b3b011c38" />

```
data.fillna(method="ffill")
```

<img width="1051" height="797" alt="image" src="https://github.com/user-attachments/assets/eeedce89-f78e-49b0-b0e5-56a739dd6c16" />

```
data.fillna(method="bfill")
```

<img width="1021" height="785" alt="image" src="https://github.com/user-attachments/assets/1407d817-7085-4aaf-8f3a-aacfebbe83be" />

```
data.fillna({'REGNO':0,'NAME':'PRAVEEN'})
```

<img width="1024" height="787" alt="image" src="https://github.com/user-attachments/assets/b40a62cc-dfdf-4cc3-b4a3-bf4c05dd5760" />

```
ir=pd.read_csv("iris.csv")
ir
```

<img width="662" height="446" alt="image" src="https://github.com/user-attachments/assets/a6be4a49-ef3a-42c0-b6c2-52e0df86963c" />


```
ir.describe()
```

<img width="718" height="321" alt="image" src="https://github.com/user-attachments/assets/75126bc1-4ba3-4b9c-8f93-429be117c717" />


```
import seaborn as sns
sns.boxplot(x='sepal_length',data=ir)
```

<img width="709" height="508" alt="Screenshot 2025-09-10 162108" src="https://github.com/user-attachments/assets/e4d802a0-d191-49b8-b320-8a50ab7f162f" />


```
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```

<img width="395" height="139" alt="image" src="https://github.com/user-attachments/assets/ae5d9bdb-9e8b-4a2a-9064-6309f1ac485b" />


```
rid=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid['sepal_width']
```

<img width="169" height="232" alt="Screenshot 2025-09-10 162148" src="https://github.com/user-attachments/assets/04f7c328-4255-443a-b42b-289706f1b780" />


```
rid=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid
```

<img width="685" height="454" alt="Screenshot 2025-09-10 162158" src="https://github.com/user-attachments/assets/ab0309af-6e29-4914-96a3-9ed8f258b820" />


```
rid=ir[((ir.sepal_width>(q1-1.5*iqr))&(ir.sepal_width<(q3+1.5*iqr)))]
rid['sepal_width']
```

<img width="193" height="509" alt="Screenshot 2025-09-10 162215" src="https://github.com/user-attachments/assets/842d92ed-715d-45a4-897d-53b94d1fd411" />


```
import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(ir.sepal_width))
z
```

<img width="745" height="618" alt="Screenshot 2025-09-10 162223" src="https://github.com/user-attachments/assets/f73f9f5e-fe39-4307-b765-9e41d42173d5" />


```
df=ir[z<3]
df
```

<img width="865" height="480" alt="Screenshot 2025-09-10 162229" src="https://github.com/user-attachments/assets/ea81f77a-f22c-4035-b079-62ebe671358e" />

# Result
         Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method
