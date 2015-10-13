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
    axvline(color='k')
    axhline(color='k')
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

    legend(leg, conditions)
    xlabel(ax1label)
    ylabel(ax2label)



def plot_population_activity(data, factors, axis1, axis2,
                             legend=False, epochs=None):
    import seaborn as sns
    conditions = list(dict_product(factors))
    colors = sns.color_palette('muted', len(conditions))
    symbols = ['-', '--', '-.', ':']
    leg = []
    for i, condition in enumerate(conditions):
        population_activity = condition_matrix(data, [condition])
        assert population_activity.shape[1] == data.data.shape[1]
        vax1 = dot(axis1, population_activity)
        vax2 = dot(axis2, population_activity)
        if epochs is None:
            h = plt.plot(vax1, vax2, color=colors[i])[0]
            plt.plot(vax1[0], vax2[0], 'o', color=colors[i])
            plt.plot(vax1[-1], vax2[-1], 'D', color=colors[i])
        else:
            for j, (low, high) in enumerate(epochs):
                h = plt.plot(vax1[low:high],
                             vax2[low:high], ls=symbols[j % len(symbols)],
                             color=colors[i])[0]
                plt.plot(vax1[low], vax2[low], 'o', color=colors[i])
                plt.plot(vax1[high], vax2[high], 'D', color=colors[i])

        leg.append((h, str(condition)))
    if legend:
        plt.legend([l[0] for l in leg], [l[1] for l in leg])
    return [l[0] for l in leg], [l[1] for l in leg]


def trellis_plot(data, Q, labels, condition, Bmax=None):
    import matplotlib
    gs = matplotlib.gridspec.GridSpec(3, 3)
    for k in range(1, Q.shape[1]):
        for j in range(Q.shape[1]-1):
            plt.subplot(gs[k-1, j])
            if k <= j:
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('')
                plt.ylabel('')
                continue
            leg = plot_population_activity(
                data,
                condition,
                Q[:, k], Q[:, j])

            plt.xlabel('%s' % labels[k])
            plt.ylabel('%s' % labels[j])
            if Bmax is not None:
                a = dot(Q[:, [k, j]].T, Bmax[:, [k, j]])
                plt.plot([-a[1, 0], a[1, 0]], [-a[1, 1], a[1, 1]], 'k--')
                plt.plot([-a[0, 0], a[0, 0]], [-a[0, 1], a[0, 1]], 'k--')
                plt.axis('equal')
    plt.subplot(gs[0, 2])
    plt.legend(leg[0], leg[1])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')


def characterize_population(data, bnt, factors, Q):
    conditions = list(dict_product(factors))
    gs = matplotlib.gridspec.GridSpec(
            len(conditions) + 1, len(unique(data.unit)))
    for i, cond in enumerate(conditions):
        d = dict_filter(data, cond)
        for iu, u in enumerate(d.by_field('unit')):
            plt.subplot(gs[i, iu])
            plt.plot(u.data.T.mean(0), 'k', alpha=0.5)
            plt.ylim([-max(absolute(plt.ylim())), max(absolute(plt.ylim()))])
            plt.plot(plt.xlim(), [0, 0], 'k--')
            plt.xticks([])
            plt.yticks([])
            if iu == 0:
                plt.ylabel(str(cond))
    for iu, unit in enumerate(unique(data.unit)):
        plt.subplot(gs[len(conditions), iu])
        bs = [bnt[unit, t] for t in range(bnt.shape[1])]
        plt.plot(array(bs)[:, (0, 2)])
        plt.legend(factors.keys())
