from numpy import *
from pylab import *
import seaborn as sns
from matplotlib.patches import Ellipse

axes_labels = ['response', 'stimulus strength']


def make_1Dplot(df, encoding_axes=0):
    sns.tsplot(df[df.encoding_axis == encoding_axes], time='time',
               unit='subject', value='data', condition='condition')
    ylabel(encoding_axes)
    axhline(color='k')


def make_2Dplot(df, colors=None, errors=False, agg=None, time_frame=None):
    if agg is None:
        agg = lambda x: nanmean(x, 0)
    if colors is None:
        colors = sns.color_palette()
    conditions = []
    leg = []
    ax1label, ax2label = unique(df.encoding_axis)

    for i, (cond, df_c) in enumerate(df.groupby('condition', sort=False)):
        ax1 = df_c[df_c.encoding_axis == ax1label].pivot('subject', 'time', 'data').values
        ax2 = df_c[df_c.encoding_axis == ax2label].pivot('subject', 'time', 'data').values
        time = df_c[df_c.encoding_axis == ax1label].pivot('subject', 'time', 'time').values
        time = agg(time)
        if time_frame is not None:
            start, end = time_frame
            start_idx = argmin(abs(time-start))
            end_idx = argmin(abs(time-end))
            end_idx = min(end_idx+1, ax1.shape[1])
            ax1 = ax1[:, start_idx:end_idx]
            ax2 = ax2[:, start_idx:end_idx]
            time = time[start_idx:end_idx]
        x, y = agg(ax1), agg(ax2)

        if not errors:
            #scatter(x, y, c=time)
            leg += [plot(x,  y, '-', color=colors[i])[0]]
        else:
            sem1 = (nanstd(ax1, 0)/(ax1.shape[0])**.5)
            sem2 = (nanstd(ax2, 0)/(ax2.shape[0])**.5)
            for xx, yy, sx, sy in zip(nanmean(ax1, 0), nanmean(ax2, 0), sem1, sem2):
                gca().add_artist(Ellipse((xx, yy), sx, sy, facecolor=colors[i], alpha=0.1))
            #errorbar(nanmean(ax1, 0), nanmean(ax2, 0), xerr=sem1, yerr=sem2, color=colors[i])
            leg += [plot(x, y, '-', color=colors[i])[0]]
        if len(x.shape) == 1:
            x = x[:, newaxis]
            y = y[:, newaxis]
        plot(x[0, :], y[0, :], color=colors[i], marker='s', linestyle='None')
        plot(x[-1, :], y[-1, :], color=colors[i], marker='>', linestyle='None')
        conditions.append(cond)
    axvline(color='k')
    axhline(color='k')
    legend(leg, conditions)
    xlabel(ax1label)
    ylabel(ax2label)
