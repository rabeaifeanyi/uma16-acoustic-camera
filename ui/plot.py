from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


DOTCOLOR = 'blue'
DOTSIZE = 4
DOTALPHA = 0.8
SHADOWCOLOR = 'orange'
SHADOWALPHA = 0.4
VIDEOALPHA = 0.75
MICCOLOR = 'gray'
MICLINECOLOR = 'black' 
MICSIZE = 3
MICALPHA = 1

# Visual Range = Estimation -> TODO make this configurable
XMIN = -2.5
XMAX = 2.5
YMIN = -1.75
YMAX = 1.75
Z = 2.0


def create_plot(frame_width, frame_height, mic_positions):
    """ Create a Bokeh plot with the given frame dimensions.
    
    Args:
        frame_width (int): The width of the frame.
        frame_height (int): The height of the frame.
        mic_positions (np.ndarray): The positions of the microphones.
    """
    # Set up figure dimensions to match room coordinates
    fig = figure(width=frame_width, 
                 height=frame_height, 
                 x_range=(XMIN, XMAX), 
                 y_range=(YMIN, YMAX),
                 output_backend='webgl')
    
    # Initialize ColumnDataSource for camera image
    cameraCDS = ColumnDataSource({'image_data': []})
    
    fig.image_rgba(image='image_data', 
                   x=XMIN, 
                   y=YMIN, 
                   dw=(XMAX-XMIN), 
                   dh=(YMAX-YMIN), 
                   source=cameraCDS, 
                   alpha=VIDEOALPHA)
    
    # Initialize ColumnDataSource for circles
    cds = ColumnDataSource(data=dict(x=[], y=[], s=[]))
    fig.scatter(x='x', 
                y='y', 
                legend_label='Strength of Source', 
                marker='circle', 
                size='s', 
                color=SHADOWCOLOR, 
                alpha=SHADOWALPHA, 
                line_color=None,
                source=cds)
    
    fig.scatter(x='x', 
                y='y', 
                legend_label='Sound Source', 
                marker='circle', 
                size=DOTSIZE, 
                color=DOTCOLOR, 
                alpha=DOTALPHA, 
                source=cds)
    
    # Initialize ColumnDataSource for microphones
    mic_cds = ColumnDataSource(data=dict(x=[], y=[]))
    mic_cds.data = dict(x=mic_positions[0], y=mic_positions[1])
    fig.scatter(x='x', 
                y='y', 
                legend_label='Microphones', 
                marker='circle', 
                size=MICSIZE, 
                color=MICCOLOR, 
                line_color=MICLINECOLOR,
                alpha=MICALPHA, 
                source=mic_cds)
    
    # mic_positions:
    # [[ 0.021  0.063  0.021  0.063  0.021  0.063  0.021  0.063 -0.063 -0.021 -0.063 -0.021 -0.063 -0.021 -0.063 -0.021]
    #  [-0.063 -0.063 -0.021 -0.021  0.021  0.021  0.063  0.063  0.063  0.063  0.021  0.021 -0.021 -0.021 -0.063 -0.063]
    #  [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.   ]]
    
    return fig, cds, mic_cds, cameraCDS

def update_plot(cds, model_data):
    """Update the plot with the given data.

    Args:
        cds (_type_): _description_
        model_data (_type_): _description_
    """
    cds.data = dict(x=model_data['x'], y=model_data['y'], s=model_data['s'])
