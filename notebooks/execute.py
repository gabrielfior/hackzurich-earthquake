import pandas as pd
from fastai.tabular.all import *
from fastai.tabular.data import *
from functools import reduce
from tqdm import tqdm, trange

learn = load_learner('monster_model_10batches.pkl')

df = pd.read_csv('../public_data/train.csv')
test = pd.read_csv('../public_data/test.csv')
build_owner = pd.read_csv('../public_data/building_ownership.csv')
build_struct = pd.read_csv('../public_data/building_structure.csv')
ward_demo = pd.read_csv('../public_data/ward_demographic_data.csv')

# merge everything before predicting
# joining on building_id
dfs = [test, build_owner,build_struct]
col_name='building_id'
df_final = reduce(lambda left,right: pd.merge(left,right,on=col_name), dfs)
#df_final.shape, df_final.columns
df_end = df_final.merge(ward_demo,left_on='ward_id_x',right_on='ward_id')

preds = []
for i in trange(len(df_end)):
    obj = df_end.iloc[i]
    row, _,_ = learn.predict(obj)
    preds.append({'building_id':df_end.iloc[i]['building_id'],
                  'damage_grade':int(row['damage_grade'].tolist()[0])})

df_export = pd.DataFrame(preds)
df_export.to_csv('submission_simple.csv',index=False)