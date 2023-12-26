from PIL import Image
import os

folder_paths = [
    'figures/2001-02',
    'figures/2002-03',
    'figures/2003-04',
    'figures/2004-05',
    'figures/2005-06',
    'figures/2006-07',
    'figures/2007-08',
    'figures/2008-09',
    'figures/2009-10',
    'figures/2010-11',
    'figures/2011-12',
    'figures/2012-13',
    'figures/2013-14',
    'figures/2014-15',
    'figures/2015-16',
    'figures/2016-17',
    'figures/2017-18',
    'figures/2018-19',
    'figures/2019-20'
]


def create_gif(folder_paths, gif_path, duration=100, loop=0):
    images = []
    
    for folder in folder_paths:
        path = os.path.join(folder, 'bar_plot.png')
        img = Image.open(path)
        images.append(img)
    
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=loop)

gif_path = 'data/bar_plot.gif'

create_gif(folder_paths, gif_path, duration=500, loop=0)