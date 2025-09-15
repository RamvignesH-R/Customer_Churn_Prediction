import numpy as np
import pandas as pd
import ast
import re

def filterDataset(dataset,allergies:list):

    if len(allergies)==0:
        return dataset
    
    allergyPattern='|'.join([re.escape(i.strip().lower()) for i in allergies])
    mask=dataset['RecipeIngredientParts'].str.lower().str.contains(allergyPattern,na=False)
    newdf=dataset[~mask]

    return newdf

def contentBasedRecommend(dataset,nutritionInput,params):

    if dataset.empty:
        return pd.DataFrame()
    
    nutritiondf=dataset.iloc[:,7:16]
    standarddf=nutritiondf

    c=0
    for i in nutritiondf.columns:
        mew=nutritiondf[i].mean()
        sigma=nutritiondf[i].std()
        
        for j in range(len(i)):
            standarddf[i].iloc[j]=(standarddf[i].iloc[j]-mew)/sigma
        nutritionInput[c]=(nutritionInput[c]-mew)/sigma
        c+=1

    val=[]
    nutritionInput=np.array(nutritionInput)

    for i in range(len(standarddf)):
        temp=(standarddf.iloc[i].to_numpy()).T
        num=np.dot(temp,nutritionInput)
        temp1=standarddf.iloc[i].to_numpy()
        norm1=np.sqrt(sum(temp1**2))
        norm2=np.sqrt(sum(nutritionInput**2))
        val.append(num/(norm1*norm2))

    print(dataset['Name'].iloc[val.index(max(val))])

df=pd.read_csv("data.csv")
contentBasedRecommend(df,[110,2.6,2.1,3,250,25,3.6,30,35],1)