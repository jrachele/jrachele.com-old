import flask
from flask_caching import Cache
import dash
import dash_core_components as dcc
import dash_html_components as html

from flask_babel import Babel

import base64
import io
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.graph_objs as go
# Static site serving
server = flask.Flask(__name__)
server.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(server)

@server.route('/')
def main_page():
    return flask.render_template('index.html')

@babel.localeselector
def get_locale():
    if flask.request.args.get('lang'):
        flask.session['lang'] = flask.request.args.get('lang')
        return flask.session.get('lang', 'en')
    return flask.request.accept_languages.best_match(['en', 'fr'])

    

# Dash
app = dash.Dash(__name__, server=server, url_base_pathname='/research')
app.title = 'Yeast genome visualization'

# Set up caching for data frames
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
}

cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

# Load CSV files for demonstration
@cache.memoize()
def extractDataFromSet(set):
    return dict(
        xyz=pd.read_csv('media/{0}/{0}_xyz.csv'.format(set), header=None).values.tolist(),
        msd=pd.read_csv('media/{0}/{0}_msd.csv'.format(set), header=None).values.tolist(),
        lagtime=pd.read_csv('media/{0}/{0}_lagtime.csv'.format(set), header=None).values.tolist(),
        colormap = pd.read_csv('media/colormap.csv', header=None).values.tolist()
    )

extracted_data = extractDataFromSet('nucleolus')
xyz = extracted_data['xyz']
msd = extracted_data['msd']
lagtime = extracted_data['lagtime']
colormap = extracted_data['colormap']

# Retrieve MSD dataset with conditions
def getMSD(msd_row, msd_range):
    row_array = msd[msd_row][msd_range[0]:msd_range[1]]
    row_array_excluding_zero = [x for x in row_array if x != 0]
    return row_array_excluding_zero
    
# This function will get the appropriate color based on the MSD of a certain column, which is standardized
# based on the MSD of a given ROW
getColor_memo = {}
def getColor(msd_row, msd_column, raw=False):
    # the standardization function goes like this: current MSD cell value - minimum of that row all over the range
    if (msd_row not in getColor_memo):
        row_array = msd[msd_row]
        row_array_excluding_zero = [x for x in row_array if x != 0]
        minVal = min(row_array_excluding_zero)
        maxVal = max(row_array_excluding_zero)
        getColor_memo[msd_row] = (row_array, minVal, maxVal)
    else:
        row_array = getColor_memo[msd_row][0]
        minVal = getColor_memo[msd_row][1]
        maxVal = getColor_memo[msd_row][2]


    standard = int((row_array[msd_column] - minVal) / (maxVal-minVal) * 64)
    if standard < 0: standard = 0 # otherwise any zero column values will produce a negative index
    # now we will get the corresponding color array for the standard
    color = [int(e*255) for e in colormap[standard-1 if standard > 0 else standard]] # here we convert the values to a 1x3 RGB standard
    return "rgb({},{},{})".format(*color) if raw is False else standard # this formats it as a plotly friendly string if raw is False


def clusterData(x_data,y_data,z_data, max_d=500):
    xyz_data = np.asarray([x_data, y_data, z_data]).transpose()
    z_link = linkage(xyz_data, 'ward') # We will use the ward algorithm to cluster the data
    clusters = fcluster(z_link, max_d, criterion='distance')
    return clusters

app.layout = html.Div(children=[
    html.Div([
        html.H1(children='Yeast Genome Visualization'),

        html.H3(
            id='tagline',
            children='Currently loaded set:',
        ),

        html.Div([
            dcc.Dropdown(
                id='set-dropdown',
                options=[
                    {'label': 'Nucleolus', 'value': 'nucleolus'},
                    {'label': 'Keq0p01', 'value': 'Keq0p01'},
                ],
                value='nucleolus'
            ),
        ]),

        html.Div(
            children='''
            A look into the yeast nucleus.
        '''),

        html.Div(
            id='set-current',
            children='nucleolus',
            style={'display': 'none'},
        ),

        html.Div(
            id='boundaries',
            style={'display': 'none'},
        ),

        dcc.Graph(
            id='3d-visualization'
        ),

        html.H5(
            id='lag-time-label',
            children='Show data at selected lag time index'),

        html.Div([
            dcc.Slider(
                id='lag-slider',
                min=0,
                max=len(lagtime[0])-1,
                value=0,
                step=1,
                marks={str(val): str(val) for val in range(len(lagtime[0])-1)[0::(int(len(lagtime[0]) / 10))]},
            ),
        ], style={'marginBottom': 50}),
        

        html.H5(
            id='indices-range-label',
            children='Filter range of indices rendered'),

        html.Div([
            dcc.RangeSlider(
                id='value-range-slider',
                count=1,
                min=0,
                max=len(msd[0])-1,
                step=1,
                value=[0, len(msd[0])-1],
                marks={str(val): str(val) for val in range(len(msd[0])-1)[0::(int(len(msd[0]) / 10))]}
            ),
        ], style={'marginBottom': 50}),
        

        html.H5(
            id='color-range-label', 
            children='Filter range of colors seen'),

        html.Div([
            dcc.RangeSlider(
                id='color-range-slider',
                count=1,
                min=0,
                max=63,
                step=1,
                value=[0,63],
                marks={str(val): str(val) for val in range(63)[0::8]}
            ),
        ], style={'marginBottom': 50}),

        html.H5(children='Filter range of Z-values rendered'),

        html.Div([
            dcc.RangeSlider(
                id='z-range-slider',
                count=1,
                min=int(min(xyz[2])),
                max=int(max(xyz[2])),
                step=1,
                value=[min(xyz[2]),max(xyz[2])],
                marks={str(val): str(val) for val in range(int(min(xyz[2])), int(max(xyz[2])))[0::(int(max(xyz[2]) / 10))]}
            ),
        ], style={'marginBottom': 50}),
        

        html.H5(children='Extra options'),

        html.Div([
            dcc.Checklist(
                id='checklist',
                options=[
                    {'label': 'Show Chain', 'value': 'show_chain'},
                    {'label': 'Consolidate data', 'value': 'consolidate_data'}
                ],
                values=[]
            ),
        ], style={'marginBottom': 50}),
    ], id='main-container'),

    
    
])


@app.callback(
    dash.dependencies.Output('lag-time-label', 'children'),
    [
        dash.dependencies.Input('lag-slider', 'value'),
    ]
)
def update_lagtime(lagSliderValue):
    return "Lag time index: {}, real lag time: {}".format(lagSliderValue, lagtime[0][lagSliderValue])

@app.callback(
    dash.dependencies.Output('indices-range-label', 'children'),
    [
        dash.dependencies.Input('value-range-slider', 'value'),
    ]
)
def update_valuerange(valueSliderValue):
    return "Index Lower boundary: {}, upper boundary: {}".format(valueSliderValue[0], valueSliderValue[1])

@app.callback(
    dash.dependencies.Output('color-range-label', 'children'),
    [
        dash.dependencies.Input('color-range-slider', 'value'),
    ]
)
def update_colorrange(colorSliderValue):
    return "Color Lower boundary: {}, upper boundary: {}".format(colorSliderValue[0], colorSliderValue[1])

@app.callback(
    dash.dependencies.Output('boundaries', 'children'),
    [
        dash.dependencies.Input('set-dropdown', 'value')
    ]
)
def changeBoundaries(dropdownSet):
    extracted_data = extractDataFromSet(dropdownSet)
    # MSD
    msd = extracted_data['msd']
    # Lagtime
    lagtime = extracted_data['lagtime']
    return json.dumps(dict(
        lagtime=[0,len(lagtime[0])-1],
        valueRange=[0,len(msd[0])-1],
    ))

@app.callback(
    dash.dependencies.Output('lag-slider', 'max'),
    [
        dash.dependencies.Input('boundaries', 'children')
    ]
)
def adjustLagMax(boundaries):
    return json.loads(boundaries)['lagtime'][1]

@app.callback(
    dash.dependencies.Output('lag-slider', 'value'),
    [
        dash.dependencies.Input('boundaries', 'children')
    ]
)
def adjustLagValue(boundaries):
    return 0

@app.callback(
    dash.dependencies.Output('value-range-slider', 'max'),
    [
        dash.dependencies.Input('boundaries', 'children')
    ]
)
def adjustValueMax(boundaries):
    return json.loads(boundaries)['valueRange'][1]

@app.callback(
    dash.dependencies.Output('value-range-slider', 'value'),
    [
        dash.dependencies.Input('boundaries', 'children')
    ]
)
def adjustValueRange(boundaries):
    return json.loads(boundaries)['valueRange']

@app.callback(
    dash.dependencies.Output('3d-visualization', 'figure'),
    [
        dash.dependencies.Input('lag-slider', 'value'),
        dash.dependencies.Input('value-range-slider', 'value'),
        dash.dependencies.Input('color-range-slider', 'value'),
        dash.dependencies.Input('z-range-slider', 'value'),
        dash.dependencies.Input('checklist', 'values'),
        dash.dependencies.Input('set-dropdown', 'value')
    ]
)
def update_figure(lagTimeIndex, valueRange, colorRange, zRange, checklist, dropdownSet):
    extracted_data = extractDataFromSet(dropdownSet)
    # Positions
    xyz = extracted_data['xyz']
    # MSD
    msd = extracted_data['msd']
    # Lagtime
    lagtime = extracted_data['lagtime']

    colorRange = [0,63] # disabling custom colors for now
    


    chainVisible = 'show_chain' in checklist
    consolidate = 'consolidate_data' in checklist

    print("lagTimeIndex: {}".format(lagTimeIndex))
    print("valueRange: {}".format(valueRange))
    print("colorRange: {}".format(colorRange))
    print("zRange: {}".format(zRange))
    print("Checklist: {}".format(checklist))

    random_walk = []
    # print("Actual lag time: {}".format(lagtime[0][lagTimeIndex]))
    x_range = []
    y_range = []
    z_range = []
    color_range = []

    msd_dataset = getMSD(lagTimeIndex, valueRange)
    print("min: {}, max:{}".format(min(msd_dataset), max(msd_dataset)))
    msd_tick = int((max(msd_dataset)-min(msd_dataset)) / 8)
    msd_range = [int(min(msd_dataset))+i*msd_tick for i in range(9)]
    tick_text = [str(msd_range[i]) for i in range(9)]
    tick = int(((colorRange[1]-colorRange[0])+1) / 8)
    tick_range = [colorRange[0]+i*tick for i in range(9)]

    print(msd_range)
    print(tick_range)

    # Vetting process depending on parameters
    for i in range(*valueRange):
        if (xyz[2][i] >= zRange[0] and xyz[2][i] <= zRange[1]):
            if (getColor(lagTimeIndex, i, True) >= colorRange[0] and getColor(lagTimeIndex, i, True) <= colorRange[1]): # if the raw color index is above colorLimit we will render the bead
                x_range.append(xyz[0][i])
                y_range.append(xyz[1][i])
                z_range.append(xyz[2][i])
                color_range.append(getColor(lagTimeIndex, i, True))

    if consolidate:
        # Now we will cluster the data
        clustering = clusterData(x_range, y_range, z_range)
        # Based on the clustering, we will group the data into a dictionary for each cluster
        clusters = {}
        for i in range(len(clustering)): # get the index at each cluster
            if clustering[i] not in clusters: # if the particular cluster at index i is empty
                clusters[clustering[i]] = [] # create an empty list
            clusters[clustering[i]].append([x_range[i], y_range[i], z_range[i], color_range[i]]) # append a set of coordinates there
        # Now we have a populated dictionary with all the clusters

        # for every cluster in the dictionary now, we can consolidate them
        x_range_consolidated = []
        y_range_consolidated = []
        z_range_consolidated = []
        size_range_consolidated = []
        color_range_consolidated = []
        for i in range(len(clusters)):
            # The cluster is found by the key at i+1

            # For now, we will consolidate it to the first object in the set
            cluster_coords = clusters[i+1][0]
            x_range_consolidated.append(cluster_coords[0])
            y_range_consolidated.append(cluster_coords[1])
            z_range_consolidated.append(cluster_coords[2])
            size_range_consolidated.append(2*len(clusters[i+1]))
            color_range_consolidated.append(cluster_coords[3])

        random_walk.append( # append the individual beads
            go.Scatter3d(
                x=x_range_consolidated,
                y=y_range_consolidated,
                z=z_range_consolidated,
                mode='markers',
                marker=dict(
                    size=size_range_consolidated,
                    color=color_range_consolidated,
                    colorbar=dict(
                        title='MSD',
                        titleside='top',
                        tickmode= 'array',
                        tickvals = tick_range,
                        ticktext = tick_text,
                        ticks='outside'
                    ),
                    colorscale='Jet',
                    opacity=1
                )
            )

        )
    else:
        random_walk.append( # append the individual beads
            go.Scatter3d(
                x=x_range,
                y=y_range,
                z=z_range,
                mode='markers',
                marker=dict(
                    size=5,
                    color= color_range,
                    colorbar=dict(
                        title='MSD',
                        titleside='top',
                        tickmode= 'array',
                        tickvals = tick_range,
                        ticktext = tick_text,
                        ticks='outside'
                    ),
                    colorscale='Jet',
                    opacity=1
                )
            )

        )




    if (chainVisible):
        random_walk.append( # append chain 1
            go.Scatter3d(
                x=x_range,
                y=y_range,
                z=z_range,
                mode='lines',
                name='Chain {}'.format(1),
                hoverinfo='none',
                line=dict(
                    color='rgba(0,0,0,0.25)',
                )
            )

        )
    data=[*random_walk]
    # Simple layout
    layout=go.Layout(
        title='Random Walk',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            aspectmode="cube",
            aspectratio=dict(
                x=1, y=1, z=1
            ),
            xaxis=dict(
                range=[-1000, 1000]
            ),
            yaxis=dict(
                range=[-1000, 1000]
            ),
            zaxis=dict(
                range=[-1000, 1000]
            )
        ),
        
        
        showlegend=False
    )
    return {
        'data': data,
        'layout': layout
    }



app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "/static/css/dash_style.css"})

