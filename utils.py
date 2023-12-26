from settings import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import os


def assign_grid(x_data, y_data, xmin, xmax, ymin, ymax, xbins, ybins):
    '''
    Assign each data point to a grid on the court. 
    '''
    x_grid, y_grid = [], []

    x = np.linspace(xmin+(xmax-xmin)/(2*xbins), xmax-(xmax-xmin)/(2*xbins), xbins)
    y = np.linspace(ymin+(ymax-ymin)/(2*ybins), ymax-(ymax-ymin)/(2*ybins), ybins)
    x_gap = x[1] - x[0]
    y_gap = y[1] - y[0]

    for x_p, y_p in zip(x_data, y_data):
        x_p = np.clip(x_p, xmin, xmax - 1e-10)
        y_p = np.clip(y_p, ymin, ymax - 1e-10)

        i = int((x_p - xmin) // x_gap)
        j = int((y_p - ymin) // y_gap)

        x_grid.append(i)
        y_grid.append(j)
    
    return x_grid, y_grid


def filter_data(year, df):
    df_filtered = df.copy()
    if year:
        df_filtered = df_filtered[df_filtered['Season']==year]

    return df_filtered


def accuracy_distance(df):
    accuracy_df = df.groupby('Distance')[['Shot Made Flag','Points']].mean().reset_index()
    accuracy_df = accuracy_df.rename(columns={'Shot Made Flag': 'Accuracy'})
    accuracy_df['Expected Points'] = accuracy_df['Accuracy'] * accuracy_df['Points']
    return accuracy_df


def get_counts(data, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin+(xmax-xmin)/(2*XBINS), xmax-(xmax-xmin)/(2*XBINS), XBINS)
    y = np.linspace(ymin+(ymax-ymin)/(2*YBINS), ymax-(ymax-ymin)/(2*YBINS), YBINS)
    x_gap = x[1] - x[0]
    y_gap = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(shape=(y.shape[0], x.shape[0]))
    EP = np.zeros(shape=(y.shape[0], x.shape[0]))

    shot_counts = data.groupby(['x grid', 'y grid']).agg(
            counts=pd.NamedAgg(column='Shot Made Flag', aggfunc='size'),
            points=pd.NamedAgg(column='Points', aggfunc='mean'),
            accuracy=pd.NamedAgg(column='Shot Made Flag', aggfunc='mean')
            ).reset_index()
    shot_counts['expected'] = shot_counts['accuracy']*shot_counts['points']

    for x_i, y_i, ct, pt, acc, exp in shot_counts.itertuples(index=False):
        Z[y_i][x_i] = ct
        EP[y_i][x_i] = exp
    
    Z = Z/shot_counts['counts'].sum()
    Z = Z + OFFSET
    x_width = X[0][1] - X[0][0]
    y_width = Y[1][0] - Y[0][0]
    X = X.ravel()
    Y = Y.ravel()
    dz = Z.ravel()

    return X, Y, Z, dz, x_width, y_width, EP


def plot_3d_bar(X, Y, dz, x_width, y_width, full_court, title):
    X = X - x_width*SHRINK/2
    Y = Y - y_width*SHRINK/2
    dx = np.repeat(x_width*SHRINK, X.shape[0])
    dy = np.repeat(y_width*SHRINK, Y.shape[0])
    im_x = list(range(full_court.shape[1]))
    im_y = list(range(full_court.shape[0]))
    im_X, im_Y = np.meshgrid(im_x, im_y)

    cmap = plt.get_cmap(CMAP)
    norm = colors.LogNorm(vmax=CLIP_RANGE)
    z_colors = cmap(norm(dz))

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(im_X, im_Y, np.zeros_like(im_X), facecolors=full_court/255.0, 
                    rstride=10, cstride=10, cmap='Greys', zorder=1)
    ax.bar3d(X, Y, np.zeros(X.shape[0]), dx, dy, np.clip(dz, 0, CLIP_RANGE), alpha=ALPHA_BAR, color=z_colors, zorder=2)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=-45)
    ax.set_aspect('auto')
    ax.set_title(f'Shot Locations: \n{title}', fontsize=45)
    ax.set_box_aspect([1, 2, 0.3])

    fig.text(0.25, 0.2, f'Probability values have been clipped at {round(CLIP_RANGE,4)} for better readability', 
        fontsize=14, color='gray')
    fig.tight_layout()
    return fig


def plot_heatmap(Z, full_court):
    full_court = Image.fromarray(full_court)

    cmap = plt.get_cmap(CMAP)
    norm = colors.LogNorm(vmax=CLIP_RANGE)
    rgba_image = (cmap(norm(Z))*255.0).astype(np.uint8)

    colormap_image = Image.fromarray(rgba_image, 'RGBA')
    colormap_image = colormap_image.resize(full_court.size, Image.NEAREST)
    
    overlay_image = Image.blend(full_court.convert('RGBA'), colormap_image, ALPHA_OVERLAY).rotate(90, expand=True)

    fig, ax = plt.subplots(figsize=(12,9))
    ax.imshow(overlay_image, origin='lower')
    ax.set_title('Shot Locations', fontsize=40)
    ax.set_axis_off()

    # Add colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z)

    cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal')
    
    
    return fig


def plot_expected(EP, full_court):
    full_court = Image.fromarray(full_court)
    EP[EP>2.0] = 0.0

    cmap = plt.get_cmap(CMAP_EXP)
    norm = colors.Normalize()
    rgba_image = (cmap(norm(EP))*255.0).astype(np.uint8)

    colormap_image = Image.fromarray(rgba_image, 'RGBA')
    colormap_image = colormap_image.resize(full_court.size, Image.NEAREST)
    
    overlay_image = Image.blend(full_court.convert('RGBA'), colormap_image, ALPHA_OVERLAY).rotate(90, expand=True)

    fig, ax = plt.subplots(figsize=(12,9))
    ax.imshow(overlay_image, origin='lower')
    ax.set_title('Expected Points per Shot', fontsize=40)
    ax.set_axis_off()
    
    fig.text(0.6, 0.3, f'Outliers have been removed', 
        fontsize=14, color='gray')

    # Add colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(EP)

    cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal')
    
    
    return fig


def plot_accuracy(df, df_filtered, zoom, title):
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(6,8), constrained_layout=True)
    ax1.set_title('Accuracy/Points per Shot vs. Distance (All Data)', fontsize=15)
    ax1.scatter(df['Distance'], df['Accuracy'], s=5, label='Expected Points')
    ax1.scatter(df['Distance'], df['Expected Points'], s=5, label='Shot Accuracy')
    ax1.set_xlabel('Distance from Basket (Feet)')
    ax1.set_ylabel('Points per Shot')
    if zoom:
        ax1.set_xlim(0, 45)
        ax1.set_ylim(0, 2)
    else:
        ax1.axvspan(df['Distance'].min(), 0, alpha=0.3, label='Behind Basket', zorder=0)
    ax1.legend(loc='upper right')

    ax2.set_title(title)
    ax2.scatter(df_filtered['Distance'], df_filtered['Accuracy'], s=5, label='Expected Points')
    ax2.scatter(df_filtered['Distance'], df_filtered['Expected Points'], s=5, label='Shot Accuracy')
    ax2.set_xlabel('Distance from Basket (Feet)')
    ax2.set_ylabel('Points per Shot')
    if zoom:
        ax2.set_xlim(0, 45)
        ax2.set_ylim(0, 2)
    else:
        ax2.axvspan(df_filtered['Distance'].min(), 0, alpha=0.3, label='Behind Basket', zorder=0)
    ax2.legend(loc='upper right')
    return fig



def plot_stats(df, df_filtered, title):
    overall_accuracy = round(df['Shot Made Flag'].mean(), 2)
    two_p_accuracy = round(df[df['Points']==2]['Shot Made Flag'].mean(), 2)
    three_p_accuracy = round(df[df['Points']==3]['Shot Made Flag'].mean(), 2)
    two_rate = round(df[df['Points']==2].size / df.size, 2)
    three_rate = round(df[df['Points']==3].size / df.size, 2)

    overall_accuracy_f = round(df_filtered['Shot Made Flag'].mean(), 2)
    two_p_accuracy_f = round(df_filtered[df_filtered['Points']==2]['Shot Made Flag'].mean(), 2)
    three_p_accuracy_f = round(df_filtered[df_filtered['Points']==3]['Shot Made Flag'].mean(), 2)
    two_rate_f = round(df_filtered[df_filtered['Points']==2].size / df_filtered.size, 2)
    three_rate_f = round(df_filtered[df_filtered['Points']==3].size / df_filtered.size, 2)

    all_stats = [overall_accuracy, two_p_accuracy, three_p_accuracy, two_rate, three_rate]
    filtered_stats = [overall_accuracy_f, two_p_accuracy_f, three_p_accuracy_f, two_rate_f, three_rate_f]
    labels = ['Shot Accuracy', '2 pt Accuracy', '3 pt Accuracy', '2 pt Shot Rate', '3 pt Shot Rate']
    stats_dict = {
        'All Data': all_stats,
        title: filtered_stats
    }

    x = np.arange(len(all_stats))
    width = 0.35
    multiplier = 0
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)

    for attribute, measurement in stats_dict.items():
        offset = width * multiplier
        rects = ax.barh(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=20)
        multiplier += 1

    ax.set_xlabel('Measurement')
    ax.set_title('Statistics Comparison', fontsize=25)
    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(labels, fontsize=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=15)
    ax.set_xlim(0, 1)
    return fig



def plot_trend(df):
    result = df.groupby('Season')['Points'].value_counts(normalize=True).unstack().fillna(0) * 100
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    result.plot(kind='bar', stacked=True, width=0.8, ax=ax)
    
    ax.set_title('2 pt vs. 3 pt Attempts Over Time')
    ax.set_xlabel('Season')
    ax.set_ylabel('Percentage')
    ax.legend(title='Points', loc='lower right', labels=['2 pt Attempt', '3 pt Attempt'])
    fig.tight_layout()

    return fig


def make_viz(year, df, full_court):
    plt.style.use('ggplot')
    plt.style.use('dark_background')

    title = year if year else 'All Data'
    if year:
        df_filtered = filter_data(year, df)
    else:
        df_filtered = df
    accuracy_df = accuracy_distance(df)
    accuracy_df_filtered = accuracy_distance(df_filtered)
    X, Y, Z, dz, x_width, y_width, EP = get_counts(
        df_filtered, 
        xmin=0, xmax=full_court.shape[1], 
        ymin=0, ymax=full_court.shape[0])

    bar_plot = plot_3d_bar(X, Y, dz, x_width, y_width, full_court, title)
    heatmap = plot_heatmap(Z, full_court)
    expected_points = plot_expected(EP, full_court)
    stats_plot = plot_stats(df, df_filtered, title)
    accuracy_plot = plot_accuracy(accuracy_df, accuracy_df_filtered, False, title)
    accuracy_plot_zoomed = plot_accuracy(accuracy_df, accuracy_df_filtered, True, title)
    trend_plot = plot_trend(df)


    if not os.path.exists(f'figures/{title}'):
        os.makedirs(f'figures/{title}')

    bar_plot.savefig(f'figures/{title}/bar_plot.png')
    heatmap.savefig(f'figures/{title}/heatmap.png')
    expected_points.savefig(f'figures/{title}/expected_points.png')
    stats_plot.savefig(f'figures/{title}/stats_plot.png')
    accuracy_plot.savefig(f'figures/{title}/accuracy_plot.png')
    accuracy_plot_zoomed.savefig(f'figures/{title}/accuracy_plot_zoomed.png')
    trend_plot.savefig(f'figures/{title}/trend_plot.png')

    return