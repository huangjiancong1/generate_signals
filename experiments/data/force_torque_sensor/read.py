import numpy as np 

sides_3 = np.load('Dataset/3_sides/Data/data.npy')
sides_4 = np.load('Dataset/4_sides/Data/data.npy')
sides_5 = np.load('Dataset/5_sides/Data/data.npy')
sides_6 = np.load('Dataset/6_sides/Data/data.npy')
sides_200 = np.load('Dataset/200_sides/Data/data.npy')


import numpy as np 
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.plotly as py
import plotly
plotly.tools.set_credentials_file(username='huangjiancong1', api_key='0ESI8NK8350cPTVnT6N2')

names = {
    0:"Force x",
    1:"Force y",
    2:"Force z",
    3:"Moment x",
    4:"Moment y",
    5:"Moment z",
    6:"Peg Position x",
    7:"Peg Position y",
    8:"Peg Position z",
    9:"Angle",
    10:"Time",
    11:"Counter",
}

data = []
for i in range(len(sides_3[0])):
    k=0
    samples=[]
    
    for j in range(15853):
        samples.append(sides_3[k][i])
        k+=100
        
    trace0 = go.Scatter(
        x=np.linspace(0, len(samples)-1, num=len(samples), endpoint=True),
        y=samples,
        mode='lines+markers',
        name=names[i],
        hoverinfo='name',
        
        marker=dict(
                size=3.5,
        ),
        line=dict(
            shape='linear',
#             color=colors[i],
             width=0.4,
        )
    )
    data.append(trace0)

layout = dict(
    legend=dict(
        y=0.5,
        traceorder='reversed',
        font=dict(
            size=16
        )
    )
)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='force_dataset_sides_3_all')
