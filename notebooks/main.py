import json

from dash import Dash, dcc, html,ctx
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sys
import plotly.graph_objects as go

from lib.utils import load_thrasher_by_index
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
colormap = {
    'Eating':'green',
    'NONE':'black',
    'Smoking':'red',
    'Jogging':'blue',
    'Medication':'yellow',
    'Exercise':'blue',
    'General':'black',
    'PILL':'yellow'
}
def get_data_dot_json(index,dir):
    import os
    files = os.listdir(dir)
    if(os.path.isfile(f'{dir}/{files[index]}/data.json')):
        print('data.json exists')
        with open(f'{dir}/{files[index]}/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        print('data.json does not exist')
        data = {}
        with open(f'{dir}/{files[index]}/data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return data
def write_data_dot_json(data,index,dir):
    import os
    files = os.listdir(dir)
    with open(f'{dir}/{files[index]}/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
df,(times,events),X = load_thrasher_by_index(1,dir='data/thrasher')
data_json = get_data_dot_json(1,dir='data/thrasher')
# fig = px.line(data_frame=df,y=['acc_x','acc_y','acc_z'])
fig = go.Figure()
fig.add_trace(go.Scatter(y=df.iloc[::10]['acc_x'],
                    mode='lines+markers',
                    name='markers'
                    ))
fig.update_traces(marker_size=1)
fig.update_layout(clickmode='event+select')
app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig,
        style={'width': '90vw', 'height': '90vh'}
    ),
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Writing Puffs**

                Click on one data point then write beginning and end of session.
            """),
            html.Button('Write Puff',id='write_puff',n_clicks=0),
            html.Pre(id='selected-data', style=styles['pre'])
    ]),
    ],style={
        'display':'flex',
        'justify-content':'center'
    }),

],style={
    'display':'flex',
    'flex-direction':'column',
    'align-items':'center',
})

@app.callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'),
    Input('write_puff', 'n_clicks'))
def display_selected_data(selectedData,n_click_puif):
    trigger = ctx.triggered_id
    print(f'triggered by {trigger}')
    if(selectedData is None):
        print('selected data is none')
        return json.dumps(selectedData, indent=2)
    print(selectedData)
    if(trigger == 'write_puff'):
        data = get_data_dot_json(1,dir='data/thrasher')
        start = selectedData['points'][0]['x']
        end = selectedData['points'][-1]['x']
        print(start,end)
        if('puffs' in data.keys()):
            print('puffs exists')
        else:
            data['puffs'] = []
        data['puffs'].append({'start':start,'end':end})
        write_data_dot_json(data,1,dir='data/thrasher')
    return json.dumps(selectedData, indent=2)

    
if __name__ == '__main__':
    app.run_server(debug=True,port=5051)