import plotly.graph_objects as go


def plot_sphere(points, values, title=None):
    x,y,z = points.reshape(3,-1)
    colorscale = [
        [0, "#214F26"],
        [0.3, "#41B54F"],
        [0.65, "#F0BD35"],
        [0.8, "#DF721A"],
        [1, "#BA1F16"]
    ]
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                    alphahull=1,
                    opacity=1,
                    colorscale=colorscale,
                    intensity=values,
                    color='blue')])
    fig.update_layout(
        title=title
    )
    fig.show()