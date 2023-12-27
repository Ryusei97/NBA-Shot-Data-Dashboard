from settings import *
import streamlit as st
from PIL import Image
import os
import time

#########################################################
# Plotting Functions ####################################
#########################################################

@st.cache_data
def load_plots(year):
    if not year:
        year = 'All Data'
    path = os.path.join('figures', year)

    bar_plot = Image.open(os.path.join(path, 'bar_plot.png'))
    accuracy_plot = Image.open(os.path.join(path, 'accuracy_plot.png'))
    accuracy_plot_zoomed = Image.open(os.path.join(path, 'accuracy_plot_zoomed.png'))
    expected_points = Image.open(os.path.join(path, 'expected_points.png'))
    heatmap = Image.open(os.path.join(path, 'heatmap.png'))
    stats_plot = Image.open(os.path.join(path, 'stats_plot.png'))
    trend_plot = Image.open(os.path.join(path, 'trend_plot.png'))

    return bar_plot, accuracy_plot, accuracy_plot_zoomed, expected_points, heatmap, stats_plot, trend_plot


@st.cache_data
def load_bars(paths):
    res = []
    for path in paths:
        img = Image.open(os.path.join('figures', path, 'bar_plot.png'))
        res.append(img)

    return res


#########################################################
# Streamlit App #########################################
#########################################################

st.set_page_config(layout = 'wide')
side, col1, col2 = st.columns([13,33,21])
col3, col4 = col2.columns([1,1])

side.title('NBA Shot Data Dashboard')

year_options = [
    '2001-02',
    '2002-03',
    '2003-04',
    '2004-05',
    '2005-06',
    '2006-07',
    '2007-08',
    '2008-09',
    '2009-10',
    '2010-11',
    '2011-12',
    '2012-13',
    '2013-14',
    '2014-15',
    '2015-16',
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20'
]


year = side.selectbox(
   'Select Season',
   year_options,
   index=None,
   placeholder='Select Season',
)
show_gif = side.toggle('Animation', value=False)
zoom = side.toggle('Zoom', value=False)

bar_plot, accuracy_plot, accuracy_plot_zoomed, expected_points, heatmap, stats_plot, trend_plot = load_plots(year)

if show_gif:
    images = load_bars(year_options)
    placeholder = col1.empty()
    
else:
    col1.image(bar_plot)

if zoom:
    col3.image(accuracy_plot_zoomed)
else:
    col3.image(accuracy_plot)
col4.image(stats_plot)

col2.image(trend_plot)

col3, col4 = col2.columns([1,1])
col3.image(heatmap)
col4.image(expected_points)



while show_gif:
    for img in images:
        placeholder.image(img)
        time.sleep(0.5)
