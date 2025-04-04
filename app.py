import pandas as pd  #handle dataframe
import numpy as np  #processing data 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import dash
from dash import dcc, html,Input,Output,State
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

# Initialize the dash app
app=dash.Dash(__name__)

# Load data and preprocessing data
filename="insurance.csv"
df=pd.read_csv(filename,engine='python')

# Create single/family status
df['single_family'] = np.where(df.children==0, 'single', 'family')

# Create age groups
bins=[18,35,50,65]
labels=['Teen','Young Adult','Adult',]
df['agegroup']=pd.cut(df['age'],bins=bins,labels=labels,right=False)

#Create a BMI group
bmi_bins = [0, 18.5, 25, 30, 55]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight','Obesity']
df['bmigroup'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

# Create Machine Learning Model
# Encode categorical data
encoder = LabelEncoder()
cat_cols = [col for col in df.columns if df[col].dtype in (['O','category'])]
df_model = df.copy()
for col in cat_cols:
    df_model[col] = encoder.fit_transform(df_model[col])

# Train the model
listTrainCols = ['age','bmi', 'sex', 'children','smoker','region']
X = df_model[listTrainCols]
y= df_model['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
model = LinearRegression()
model.fit(X_train, y_train)


app.layout=html.Div([
    html.H1("Insurance Charge Analysis Dashboard",style={'textAlign':'center'}),

    # Dropdown Menu
    dcc.Dropdown(
        id='plot selector',
        options=[
            {'label':'Charges by Family Status and Smoking','value':'family_smoke'},
            {'label':'Charges by Number of Children','value':'children'},
            {'label':'Average Charges by Age Group','value':'age_group'},
        ],
        value='family_smoke'
    ),

    # Graph display
    dcc.Graph(id='main graph'),

    # Prediction Section
    html.Div([
        html.H2("Insurance Charge Prediction"),

        # Input Fields
        html.Div([
            html.Label("Age"),
            dcc.Input(id='age_input',type='number',value=20),

            html.Label('BMI'),
            dcc.Input(id='bmi_input',type='number',value=25),

            html.Label('Children'),
            dcc.Input(id='children_input',type='number',value=0),

            html.Label('Sex'),
            dcc.Dropdown(
                id='sex_input',
                options=[{'label':'Male','value':1},
                         {'label':'Female','value':0}],
                value=1,
            ),

            html.Label('Smoker'),
            dcc.Dropdown(
                id='smoker_input',
                options=[{'label':'Yes','value':1},
                         {'label':'No','value':0}],
                value=1,
            ),

            html.Label('Region'),
            dcc.Dropdown(
                id='region_input',
                options=[{'label':'NE','value':0},
                         {'label':'NW','value':1},
                         {'label':'SE','value':2},
                         {'label':'SW','value':3}],
                value=1,
            ),
        ],style={'display':'flex','flexDirection':'column','gap':'10px'} ),

        html.Button('Predict',id='predict_button',n_clicks=0),

        html.Div(id='prediction_output'),
     ])
    
    
])
# Callback functions
@app.callback(
    Output('main graph','figure'),
    Input('plot selector','value'),
)

def update_graph(selected_plot):
    if selected_plot=='family_smoke':
        fig=px.box(df,y="charges",x="single_family",color='smoker')
    elif selected_plot=='children':
        fig=px.box(df,y="charges",x="children",color='smoker')#edit later
    elif selected_plot == 'age_group':
        age_group_charges = df.groupby('agegroup')['charges'].mean().reset_index()
        fig = px.bar(age_group_charges, x='agegroup', y='charges',
                    title='Average Insurance Charges by Age Group')   
    return fig

@app.callback(
    Output('prediction_output','children'),
    Input('predict_button','n_clicks'),
    [State('age_input','value'),
     State('bmi_input','value'),
     State('children_input','value'),
     State('sex_input','value'),
     State('smoker_input','value'),
     State('region_input','value'),
    ])

def predict_charge(n_clicks,age,bmi,children,sex,smoker,region):
    if n_clicks>0:
        input_data=np.array([[age,bmi,children,sex,smoker,region]])
        prediction=model.predict(input_data)[0]
        return f'Predicted Insurance Charge: ${prediction:,.2f}'
    return ''

if __name__=='__main__':
    app.run_server(debug=True,port=8097)