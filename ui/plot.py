from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


DOTCOLOR = 'black'
SHADOWCOLOR = 'orange'
DOTSIZE = 4
DOTALPHA = 1
SHADOWALPHA = 0.8
VIDEOALPHA = 0.75


def create_plot(frame_width, frame_height):
    # Create the figure using the provided width and height
    fig = figure(width=frame_width, height=frame_height, output_backend='webgl')
    
    # Initialize ColumnDataSource for circles
    cds = ColumnDataSource(data=dict(x=[], y=[], s=[]))
    fig.scatter(x='x', y='y', legend_label='Strength of Source', marker='circle', size='s', color=SHADOWCOLOR, alpha=SHADOWALPHA, source=cds)
    fig.scatter(x='x', y='y', legend_label='Sound Source',marker='circle', size=DOTSIZE, color=DOTCOLOR, alpha=DOTALPHA, source=cds)
    
    # Initialize ColumnDataSource for camera image
    cameraCDS = ColumnDataSource({'image_data': []})
    fig.image_rgba(image='image_data', x=0, y=0, dw=frame_width, dh=frame_height, source=cameraCDS, alpha=VIDEOALPHA)
    
    # Set the title and axis labels
    fig.legend.title = "Obervations"
    fig.legend.label_text_font = "times"
    fig.legend.label_text_font_style = "italic"
    
    return fig, cds, cameraCDS

def update_plot(cds, model_data):
    # Update the ColumnDataSource with data from the model
    cds.data = dict(x=model_data['x'], y=model_data['y'], s=model_data['s'])
