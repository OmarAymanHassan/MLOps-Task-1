
from pckgs import train
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder , StandardScaler ,LabelEncoder
import hydra


df = train.return_dataframe()
"""if df.empty:
    print("Empty df")
else:
    print(df.head(2))
"""


#print (f"Processing Time")


#print(f"DataFrame shape : {df.shape} ")

def splitting_data (cfg,df):
 #   print(f"----getting column types----")
    x = df.drop([cfg.target_col.name,"Name","PassengerId"],axis=1)
    y = df[cfg["target_col"]["name"]]
    x_train ,x_test , y_train , y_test = train_test_split(x , y , random_state=cfg["train_test_split"]["random_state"] , test_size=cfg.train_test_split.test_size)
    return x_train ,x_test ,y_train , y_test


def columns_types(x_train):
  #  print("-----Obtaining the columns------")
    ohe_cols = [ i for i in x_train.columns if x_train[i].dtype == "object" and x_train[i].nunique() < 10 ]
    ordinal_cols = [i for i in x_train.columns if x_train[i].dtype =="object" and x_train[i].nunique() >= 10]
    numerical_cols = [i for i in x_train.columns if x_train[i].dtype != "object"]
    numerical_discrete = [i for i in x_train.columns if x_train[i].dtype != "object" and x_train[i].nunique()>15]
    numerical_continous = [i for i in numerical_cols if i not in numerical_discrete]

    return ohe_cols , ordinal_cols , numerical_discrete , numerical_continous , numerical_cols


def pipeline(cfg,ohe_cols ,ordinal_cols , numerical_cols):
   # print("------Creating ColumnTransformer----------")
    transformer = ColumnTransformer([
        ("ohe_cols", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("ordinal_cols", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=cfg.column_transformer.unknown_value), ordinal_cols),
        ("standard_scaling", StandardScaler(), numerical_cols)],remainder="passthrough")
    

    return transformer


def apply_pipeline(transformer , x_train , x_test):
    #print("------Applying ColumnTransformer----------")

    x_train=transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    return x_train , x_test


def fully_transformation(cfg,df):

    #print("------Fully Transformation----------")

    #df = train.return_dataframe()
    x_train , x_test , y_train , y_test = splitting_data(cfg,df)
    ohe_cols , ordinal_cols , numerical_discrete,numerical_continuous ,numerical_cols = columns_types(x_train)
    transformer = pipeline(cfg,ohe_cols , ordinal_cols , numerical_cols)
    x_train , x_test = apply_pipeline(transformer , x_train , x_test)


    #print(f"y_test after all transformation : {x_train.shape}")
   # print(f"x_train df :\n {x_train[0]}")

    return x_train , x_test, y_train , y_test