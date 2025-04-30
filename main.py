from pckgs import train , process , model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import hydra


@hydra.main(config_name="config" , config_path="conf")
# config_path : folder name that contains the config.yaml
# config_name : file name of .yaml 

def main(cfg):
   # print(cfg)

    # importing the dataframe 
    df = train.return_dataframe()

    # processing the data 

    x_train , x_test, y_train , y_test = process.fully_transformation(cfg,df)




    # Model
    models = {"RandomForest" : RandomForestClassifier(max_depth=cfg.params.max_depth , n_jobs= cfg["params"]["n_jobs"],n_estimators=cfg.params.n_estimators) 
              , "Xgboost" :XGBClassifier(max_depth =cfg["params"]["max_depth"] , n_jobs =cfg.params.n_jobs , n_estimators = cfg["params"]["n_estimators"]) }

    for model_name , real_model in models.items():
        print(f"{model_name}")
        model.fit_model(real_model,x_train , y_train)
        model.predict_model(real_model,x_test , y_test)



main()