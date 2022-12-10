import dash
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc 
import base64
import datetime
import io
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#052F37",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "22rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#045D68",
}

PANEL_STYLE = {
    "margin-left": "5rem",
    "margin-right": "35rem",
    "background-color": "#045D68",
}

sidebar = html.Div(
    [
        html.H2("TELCO", className="display-4",style={'color': 'white', 'text-align' : 'center'}),
        html.Hr(),
        html.P(
            "we help you to predict customer churn, to be a better telecommunications company.", className="lead",style={'color': 'white'}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact",style={'color': 'white'}),
                dbc.NavLink("Unique prediction", href="/unique-prediction", active="exact",style={'color': 'white'}),
                dbc.NavLink("Set of predictions", href="/set-of-predictions", active="exact",style={'color': 'white'}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
],style={'color': '#045D68'})


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.Div([
                    html.H1("Customer Churn",style={'color': 'white', 'font-weight': 'bold'}),
                    html.Img(src=app.get_asset_url('logo.png'))],style={'textAlign':'center'})
                ]
    elif pathname == "/unique-prediction":
        return [
                html.Div([
                    html.P("Demographic Information:", className="lead",style={'color': 'white', 'font-weight': 'bold'}),
                    dbc.Label("Gender", html_for="gender",style={'color': 'white'}),
                    dcc.Dropdown(id="gender",options=[{"label": "Male", "value": 0},{"label": "Female", "value": 1}]),
                    html.Br(),
                    dbc.Label("SeniorCitizen ", html_for="seniorCitizen",style={'color': 'white'}),
                    dcc.Dropdown(id="seniorCitizen",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1}]),
                    html.Br(),
                    dbc.Label("Partner ", html_for="partner",style={'color': 'white'}),
                    dcc.Dropdown(id="partner",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1}]),
                    html.Br(),
                    dbc.Label("Dependents", html_for="dependents",style={'color': 'white'}),
                    dcc.Dropdown(id="dependents",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1}]),
                    html.Br(),
                    html.Br(),
                    html.P("Services signed up for:", className="lead",style={'color': 'white', 'font-weight': 'bold'}),
                    dbc.Label("PhoneService ", html_for="phoneService",style={'color': 'white'}),
                    dcc.Dropdown(id="phoneService",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1}]),
                    html.Br(),
                    dbc.Label("MultipleLines  ", html_for="multipleLines",style={'color': 'white'}),
                    dcc.Dropdown(id="multipleLines",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Phone Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("InternetService  ", html_for="internetService",style={'color': 'white'}),
                    dcc.Dropdown(id="internetService",options=[{"label": "DSL", "value": 0},{"label": "Fiber Optic", "value": 1},{"label": "No", "value": 2}]),
                    html.Br(),
                    dbc.Label("OnlineSecurity ", html_for="onlineSecurity",style={'color': 'white'}),
                    dcc.Dropdown(id="onlineSecurity",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("OnlineBackup  ", html_for="onlineBackup",style={'color': 'white'}),
                    dcc.Dropdown(id="onlineBackup",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("DeviceProtection  ", html_for="deviceProtection",style={'color': 'white'}),
                    dcc.Dropdown(id="deviceProtection",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("TechSupport ", html_for="techSupport",style={'color': 'white'}),
                    dcc.Dropdown(id="techSupport",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("StreamingTV  ", html_for="streamingTV",style={'color': 'white'}),
                    dcc.Dropdown(id="streamingTV",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    dbc.Label("StreamingMovies ", html_for="streamingMovies",style={'color': 'white'}),
                    dcc.Dropdown(id="streamingMovies",options=[{"label": "Yes", "value": 0},{"label": "No", "value": 1},{"label": "No Internet Service", "value": 2}]),
                    html.Br(),
                    html.Br(),
                    html.P("Customer account information:", className="lead",style={'color': 'white', 'font-weight': 'bold'}),
                    dbc.Label("Tenure ", html_for="tenure",style={'color': 'white'}),
                    dbc.Input(type="number", id="tenure", placeholder="Number of months"),
                    dbc.FormText("Number of months the customer has stayed with the company",color="secondary"),
                    html.Br(),
                    html.Br(),
                    dbc.Label("Contract  ", html_for="contract",style={'color': 'white'}),
                    dcc.Dropdown(id="contract",options=[{"label": "Month-to-month", "value": 0},{"label": "One year", "value": 1},{"label": "Two years", "value": 2}]),
                    html.Br(),
                    dbc.Label("PaperlessBilling   ", html_for="paperlessBilling",style={'color': 'white'}),
                    dcc.Dropdown(id="paperlessBilling",options=[{"label": "No", "value": 0},{"label": "Yes", "value": 1}]),
                    html.Br(),
                    dbc.Label("PaymentMethod  ", html_for="paymentMethod",style={'color': 'white'}),
                    dcc.Dropdown(id="paymentMethod",options=[{"label": "Electronic check", "value": 0},{"label": "Mailed check", "value": 1},{"label": "Bank transfer", "value": 2},{"label": "Credit card", "value": 3}]),
                    html.Br(),
                    dbc.Label("MonthlyCharges  ", html_for="monthlyCharges",style={'color': 'white'}),
                    dbc.Input(type="number", id="monthlyCharges", placeholder="Enter the amount"),
                    dbc.FormText("The amount charged to the customer monthly",color="secondary"),
                    html.Br(),
                    html.Br(),
                    dbc.Label("TotalCharges ", html_for="totalCharges",style={'color': 'white'}),
                    dbc.Input(type="number", id="totalCharges", placeholder="Enter the amount"),
                    dbc.FormText("The total amount charged to the customer",color="secondary"),
                    html.Br(),
                    html.Br(),
                    html.Div([html.Button('Predict', id='btn-predict', n_clicks=0)],style={'text-align' : 'center'}),
                    html.Br(),
                    html.Br(),
                    html.H4(id='single-output',style={'color': 'white', 'font-weight': 'bold','text-align' : 'center'})
                ],style=PANEL_STYLE)
                ]

    elif pathname == "/set-of-predictions":
        return [
                html.Div([
                html.H1("Set Of Predictions",style={'color': 'white', 'text-align' : 'center','font-weight': 'bold'}),
                html.Br(),
                html.Br(),
                html.P("Please upload the CSV with the users data:", className="lead",style={'color': 'white', 'font-weight': 'bold'}),
                dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',
                html.A('Select Files')]),
                style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'color': 'white'
                },
                # Allow multiple files to be uploaded
                multiple=True
                ),
                html.Br(),
                html.Div(id='output-data-upload'),])

                ]
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    Output(component_id="single-output", component_property="children"),
    Input('btn-predict', 'n_clicks'),
    State(component_id="gender", component_property="value"),
    State(component_id="seniorCitizen", component_property="value"),
    State(component_id="partner", component_property="value"),
    State(component_id="dependents", component_property="value"),
    State(component_id="tenure", component_property="value"),
    State(component_id="phoneService", component_property="value"),
    State(component_id="multipleLines", component_property="value"),
    State(component_id="internetService", component_property="value"),
    State(component_id="onlineSecurity", component_property="value"),
    State(component_id="onlineBackup", component_property="value"),
    State(component_id="deviceProtection", component_property="value"),
    State(component_id="techSupport", component_property="value"),
    State(component_id="streamingTV", component_property="value"),
    State(component_id="streamingMovies", component_property="value"),
    State(component_id="contract", component_property="value"),
    State(component_id="paperlessBilling", component_property="value"),
    State(component_id="paymentMethod", component_property="value"),
    State(component_id="monthlyCharges", component_property="value"),
    State(component_id="totalCharges", component_property="value")
    )

def prediction_output_div(btnPred, gender, seniorcitizen, partner,dependents, tenure, phoneservice, mutliplelines, internetservice,onlinesecurity, onlinebackup, deviceprotection,techsupport, streamingtv,streamingmovies,contract,paperlessbilling,PaymentMethod,monthlycharges,totalcharges):

    data = {
                'gender': gender,
                'SeniorCitizen': seniorcitizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod, 
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges
                }


    singleDf = pd.DataFrame.from_dict([data])

    singleDf.replace('No internet service','No',inplace=True)
    singleDf.replace('No phone service','No',inplace=True)
    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yes_no_columns:
        singleDf[col].replace({'Yes': 1,'No': 0},inplace=True)

    #Remplazemos tambien el genero por 0 para hombre y 1 para mujer
    singleDf['gender'].replace({'Female':1,'Male':0},inplace=True)

    if(singleDf['InternetService'].iloc[0]==0): 
        singleDf['InternetService_DSL']=1
        singleDf['InternetService_Fiber_optic']=0
        singleDf['InternetService_No']=0
    elif(singleDf['InternetService'].iloc[0]==1):
        singleDf['InternetService_DSL']=0
        singleDf['InternetService_Fiber_optic']=1
        singleDf['InternetService_No']=0
    else:
        singleDf['InternetService_DSL']=0
        singleDf['InternetService_Fiber_optic']=0
        singleDf['InternetService_No']=1

    if(singleDf['Contract'].iloc[0]==0): 
        singleDf['Contract_Month-to-month']=1
        singleDf['Contract_One_year']=0
        singleDf['Contract_Two_year']=0
    elif(singleDf['Contract'].iloc[0]==1):
        singleDf['Contract_Month-to-month']=0
        singleDf['Contract_One_year']=1
        singleDf['Contract_Two_year']=0
    else:
        singleDf['Contract_Month-to-month']=0
        singleDf['Contract_One_year']=0
        singleDf['Contract_Two_year']=1

    if(singleDf['PaymentMethod'].iloc[0]==0): 
        singleDf['PaymentMethod_Bank_transfer_(automatic)']=0
        singleDf['PaymentMethod_Credit_card_(automatic)']=0
        singleDf['PaymentMethod_Electronic_check']=1
        singleDf['PaymentMethod_Mailed_check']=0
    elif(singleDf['PaymentMethod'].iloc[0]==1):
        singleDf['PaymentMethod_Bank_transfer_(automatic)']=0
        singleDf['PaymentMethod_Credit_card_(automatic)']=0
        singleDf['PaymentMethod_Electronic_check']=0
        singleDf['PaymentMethod_Mailed_check']=1
    elif(singleDf['PaymentMethod'].iloc[0] ==2):
        singleDf['PaymentMethod_Bank_transfer_(automatic)']=1
        singleDf['PaymentMethod_Credit_card_(automatic)']=0
        singleDf['PaymentMethod_Electronic_check']=0
        singleDf['PaymentMethod_Mailed_check']=0
    else:
        singleDf['PaymentMethod_Bank_transfer_(automatic)']=0
        singleDf['PaymentMethod_Credit_card_(automatic)']=1
        singleDf['PaymentMethod_Electronic_check']=0
        singleDf['PaymentMethod_Mailed_check']=0

    singleDf['gender'] = pd.to_numeric(singleDf['gender'])
    singleDf['SeniorCitizen'] = pd.to_numeric(singleDf['SeniorCitizen'])
    singleDf['Partner'] = pd.to_numeric(singleDf['Partner'])
    singleDf['Dependents'] = pd.to_numeric(singleDf['Dependents'])
    singleDf['tenure'] = pd.to_numeric(singleDf['tenure'])
    singleDf['PhoneService'] = pd.to_numeric(singleDf['PhoneService'])
    singleDf['MultipleLines'] = pd.to_numeric(singleDf['MultipleLines'])
    singleDf['OnlineSecurity'] = pd.to_numeric(singleDf['OnlineSecurity'])
    singleDf['OnlineBackup'] = pd.to_numeric(singleDf['OnlineBackup'])
    singleDf['InternetService'] = pd.to_numeric(singleDf['InternetService'])
    singleDf['OnlineSecurity'] = pd.to_numeric(singleDf['OnlineSecurity'])
    singleDf['OnlineBackup'] = pd.to_numeric(singleDf['OnlineBackup'])
    singleDf['DeviceProtection'] = pd.to_numeric(singleDf['DeviceProtection'])
    singleDf['TechSupport'] = pd.to_numeric(singleDf['TechSupport'])
    singleDf['StreamingTV'] = pd.to_numeric(singleDf['StreamingTV'])
    singleDf['StreamingMovies'] = pd.to_numeric(singleDf['StreamingMovies'])
    singleDf['PaperlessBilling'] = pd.to_numeric(singleDf['PaperlessBilling'])

    singleDf.drop('InternetService',axis='columns',inplace=True)
    singleDf.drop('Contract',axis='columns',inplace=True)
    singleDf.drop('PaymentMethod',axis='columns',inplace=True)

    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

    scaler = MinMaxScaler()
    singleDf[cols_to_scale] = scaler.fit_transform(singleDf[cols_to_scale])

    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model("churnModel.bin")    

    result = loaded_model.predict(singleDf)
    p = singleDf.columns
    if(result==1) : result = "Yes, the customer will terminate the service"
    elif(result==0) : result = "The client will not terminate the service"
    return(result)

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

            dfOrigin = df.copy()
            dfOrigin['SeniorCitizen'].replace(0,'No',inplace=True)
            dfOrigin['SeniorCitizen'].replace(1,'Yes',inplace=True)

            #En este caso vamos a eliminar la columna de costumerID porque no la vamos a utilizar
            df.drop('customerID',axis='columns',inplace=True)

            df.replace('No internet service','No',inplace=True)
            df.replace('No phone service','No',inplace=True)

            yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
            for col in yes_no_columns:
                df[col].replace({'Yes': 1,'No': 0},inplace=True)

            #Remplazemos tambien el genero por 0 para hombre y 1 para mujer
            df['gender'].replace({'Female':1,'Male':0},inplace=True)

            dfG = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod'])
            dfG.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)   

            cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

            scaler = MinMaxScaler()
            dfG[cols_to_scale] = scaler.fit_transform(dfG[cols_to_scale])

            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model("churnModel.bin")    

            prediction = loaded_model.predict(dfG)

            prediction_df = pd.DataFrame(dfOrigin['customerID'], columns=["customerID"])
            prediction_df['Predictions'] = prediction
            prediction_df = prediction_df.replace({1:'   Yes, the customer will terminate the service.', 
                                                    0:'   No, the customer will not terminate the service.'})

            dfOrigin['Churn'] = prediction
            dfOrigin['Churn'].replace(0,'No',inplace=True)
            dfOrigin['Churn'].replace(1,'Yes',inplace=True)
            dfOrigin.replace('No internet service','No',inplace=True)
            dfOrigin.replace('No phone service','No',inplace=True)

            figGender = px.bar(dfOrigin, x=dfOrigin['gender'], color=dfOrigin['Churn'], barmode="group")
            figSeniorCitizen = px.bar(dfOrigin, x=dfOrigin['SeniorCitizen'], color=dfOrigin['Churn'], barmode="group")
            figPartner = px.bar(dfOrigin, x=dfOrigin['Partner'], color=dfOrigin['Churn'], barmode="group")
            figDependents = px.bar(dfOrigin, x=dfOrigin['Dependents'], color=dfOrigin['Churn'], barmode="group")
            figPhoneService = px.bar(dfOrigin, x=dfOrigin['PhoneService'], color=dfOrigin['Churn'], barmode="group")
            figPaperlessBilling = px.bar(dfOrigin, x=dfOrigin['PaperlessBilling'], color=dfOrigin['Churn'], barmode="group")
            figStreamingTV = px.bar(dfOrigin, x=dfOrigin['StreamingTV'], color=dfOrigin['Churn'], barmode="group")
            figStreamingMovies = px.bar(dfOrigin, x=dfOrigin['StreamingMovies'], color=dfOrigin['Churn'], barmode="group")
            figOnlineSecurity = px.bar(dfOrigin, x=dfOrigin['OnlineSecurity'], color=dfOrigin['Churn'], barmode="group")
            figOnlineBackup = px.bar(dfOrigin, x=dfOrigin['OnlineBackup'], color=dfOrigin['Churn'], barmode="group")
            figDeviceProtection = px.bar(dfOrigin, x=dfOrigin['DeviceProtection'], color=dfOrigin['Churn'], barmode="group")
            figTechSupport = px.bar(dfOrigin, x=dfOrigin['TechSupport'], color=dfOrigin['Churn'], barmode="group")

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.',
        ])

    return html.Div([
        html.H5(filename,style={'color': 'white'}),
        html.H6(datetime.datetime.fromtimestamp(date),style={'color': 'white'}),
        html.Div([
            dash_table.DataTable(
            dfOrigin.to_dict('records'),
            [{'name': i, 'id': i} for i in dfOrigin.columns],
            page_size=20,
            style_cell={'textAlign': 'left'},
            style_table={'overflowX': 'auto'},
            ),
            html.Hr(),  # horizontal line
            ],CONTENT_STYLE),
        html.Br(),
        html.H3("Predictions", style={'color': 'white','textAlign':'center','font-weight': 'bold'}),
        html.Br(),
        html.Div([generate_table(prediction_df)]),
        html.Br(),
        html.Div([
            html.H3('Demographic data:', style={'color': 'white','textAlign':'center','font-weight': 'bold'}),
            dcc.Graph(id="figGender", figure=figGender,style={'display': 'inline-block'}),
            dcc.Graph(id="figSeniorCitizen", figure=figSeniorCitizen,style={'display': 'inline-block'}),
            ]),
        html.Div([
            dcc.Graph(id="figPartner", figure=figPartner,style={'display': 'inline-block'}),
            dcc.Graph(id="figDependents", figure=figDependents,style={'display': 'inline-block'}),
            ]),
        html.Br(),
        html.Div([
            html.H3('Customer account information:', style={'color': 'white','textAlign':'center','font-weight': 'bold'}),
            dcc.Graph(id="figPhoneService", figure=figPhoneService,style={'display': 'inline-block'}),
            dcc.Graph(id="figPaperlessBilling", figure=figPaperlessBilling,style={'display': 'inline-block'}),
            ]),
        html.Div([
            dcc.Graph(id="figStreamingTV", figure=figStreamingTV,style={'display': 'inline-block'}),
            dcc.Graph(id="figStreamingMovies", figure=figStreamingMovies,style={'display': 'inline-block'}),
            ]),
        html.Div([
            dcc.Graph(id="figOnlineSecurity", figure=figOnlineSecurity,style={'display': 'inline-block'}),
            dcc.Graph(id="figOnlineBackup", figure=figOnlineBackup,style={'display': 'inline-block'}),
            ]),
        html.Div([
            dcc.Graph(id="figDeviceProtection", figure=figDeviceProtection,style={'display': 'inline-block'}),
            dcc.Graph(id="figTechSupport", figure=figTechSupport,style={'display': 'inline-block'}),
            ]),

        
    ])

def generate_table(dataframe, max_rows=20):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ],style={'color': 'white'})

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__=='__main__':
    app.run_server(debug=True, port=3000)
