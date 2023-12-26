from settings import * 
from utils import *
import pandas as pd
import numpy as np
from PIL import Image
import os

# Import Court Image
court = np.asarray(Image.open('data/nba_court.jpg'))
full_court = np.concatenate((court[::-1], court))

if not os.path.exists('data/NBA Shots refined.csv'):

    # Import the data
    data = pd.read_csv('data/NBA Shot Locations 1997 - 2020.csv')

    # Transform the coordinates to match the pixels
    # X: 250, -250, Y: -52, 418
    new_ymax, new_xmax, _ = court.shape

    # Remap the coordinates to pixel coordinates
    data['X'] = (data['X Location'] - XMIN) * new_xmax / (XMAX-XMIN)
    data['Y'] = (data['Y Location'] - YMIN) * new_ymax / (YMAX-YMIN)

    data['Distance'] = np.sign(data['Y Location'])*np.ceil(5*(data['X Location']**2 + data['Y Location']**2)**0.5/10)/5
    data['Points'] = 2*data['2pt Attempts'] + 3*data['3pt Attempts']
    data['Shot Made Flag'] = data['Shot Made Flag'].map({'Made':1, 'Missed':0})
    data = data[['Game Date', 'Player ID', 'Player Name', 'Season',
        'Shot Made Flag', 'Shot Type', 'Team Name',
        'Hex X', 'Hex Y', 'X Location', 'Y Location', 
        'X', 'Y', 'Distance', 'Points']]



    x_grid, y_grid = assign_grid(data['X'], data['Y'], xmin=0, xmax=new_xmax, ymin=0, ymax=2*new_ymax, xbins=XBINS, ybins=YBINS)
    data['x grid'] = x_grid
    data['y grid'] = y_grid

    data.to_csv('data/NBA Shots refined.csv')
    print('Finished Processing')

else:
    data = pd.read_csv('data/NBA Shots refined.csv')

    
for season in data['Season'].unique():
    print(f'Creating figures for {season}')
    make_viz(season, data, full_court)

print(f'Creating figures for All Data')
make_viz(None, data, full_court)