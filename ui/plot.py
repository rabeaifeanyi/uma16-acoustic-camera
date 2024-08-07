from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


DOTCOLOR = 'black'
SHADOWCOLOR = 'orange'
DOTSIZE = 4
DOTALPHA = 1
SHADOWALPHA = 0.8
VIDEOALPHA = 0.75
XMIN = -1.5
XMAX = 1.5
YMIN = -1.5
YMAX = 1.5


def create_plot(frame_width, frame_height):
    # Set up figure dimensions to match room coordinates
    fig = figure(width=frame_width, height=frame_height, 
                 x_range=(XMIN, XMAX), y_range=(YMIN, YMAX),
                 output_backend='webgl')
    
    # Initialize ColumnDataSource for circles
    cds = ColumnDataSource(data=dict(x=[], y=[], s=[]))
    fig.scatter(x='x', y='y', legend_label='Strength of Source', marker='circle', size='s', 
                color=SHADOWCOLOR, alpha=SHADOWALPHA, source=cds)
    fig.scatter(x='x', y='y', legend_label='Sound Source', marker='circle', size=DOTSIZE, 
                color=DOTCOLOR, alpha=DOTALPHA, source=cds)
    
    # Initialize ColumnDataSource for camera image
    cameraCDS = ColumnDataSource({'image_data': []})
    fig.image_rgba(image='image_data', x=XMIN, y=YMIN, dw=(XMAX - XMIN), dh=(YMAX - YMIN), 
                   source=cameraCDS, alpha=VIDEOALPHA)
    
    return fig, cds, cameraCDS

def update_plot(cds, model_data):
    # Update the ColumnDataSource with data from the model
    cds.data = dict(x=model_data['x'], y=model_data['y'], s=model_data['s'])
