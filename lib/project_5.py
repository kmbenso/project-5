
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC



def load_data_from_database():
    url = 'postgresql://dsi:correct horse battery staple@joshuacook.me:5432'
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM madelon", con=engine)
    
    return df 
  
    
    
def make_data_dict(df, random_state=None):
    
    y = df['label']
    
    X = []
    for i in df.columns:
        if 'feat' in i:
            X.append(i)
    X = df[X]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
                                          
    return {
        'X_train'     : X_train,
        'X_test'      : X_test,
        'y_train'     : y_train,
        'y_test'      : y_test
    }

def general_transformer(transformer, data_dict): 
    #if transformer == SelectKBest():
    transformer.fit(data_dict['X_train'], data_dict['y_train'])
    #else:
     #   transformer.fit(data_dict['X_train'])
                        
    data_dict['X_train'] = transformer.transform(data_dict['X_train'])
    data_dict['X_test']= transformer.transform(data_dict['X_test'])
    
    return data_dict


def general_model(model, data_dict):
    model.fit(data_dict['X_train'], data_dict['y_train'])
    data_dict['train_score'] = model.score(data_dict['X_train'], data_dict['y_train'])
    data_dict['test_score'] = model.score(data_dict['X_test'], data_dict['y_test'])
    data_dict['model'] = model
    return data_dict


