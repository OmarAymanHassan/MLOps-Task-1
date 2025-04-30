
from pckgs import process
from pckgs import train


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score



#model = RandomForestClassifier(max_depth=5)

def fit_model(model, x_train , y_train):
  #  print("----Fitting Phase--------")

    model.fit(x_train , y_train)




def predict_model(model ,x_test, y_test):
   # print("-----Predict Phase------")

    y_pred = model.predict(x_test)
    accuracy= accuracy_score(y_test , y_pred)
    report = classification_report(y_test , y_pred)
    print(f"Model Accuracy : {accuracy}")

  #  print(f"\n ----------------- Full Report ------------ \n {report}")




