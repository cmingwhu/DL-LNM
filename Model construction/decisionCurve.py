import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, lw = 1, color = 'royalblue', label = 'ResNet-SVM')
    ax.plot(thresh_group, net_benefit_all,lw = 1, color = 'darkgray',label = 'All')
    ax.plot((0, 1), (0, 0), lw = 1, color = 'black', label = 'NONE')



    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    # ax.grid('major')

    ax.grid(linestyle="--", linewidth=0.5,
            color=".25", zorder=-10)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc = 'upper right',frameon=True,framealpha= 1)

    return ax

def plot_DCAs(ax, thresh_group, net_models, net_alls, colors, names):
    for (net_benefit_model, net_benefit_all, color, name) in zip(net_models, net_alls, colors, names):
    #Plot
        ax.plot(thresh_group, net_benefit_model, lw = 1, color = color, label = name)
    ax.plot(thresh_group, net_benefit_all,lw = 1, color = 'darkgray',label = 'All')
    ax.plot((0, 1), (0, 0), lw = 1, color = 'black', label = 'NONE')


    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'fontsize': 12}#{'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'fontsize': 12}#{'family': 'Times New Roman', 'fontsize': 15}
        )
    # ax.grid('major')

    ax.grid(linestyle="--", linewidth=0.5,
            color="gainsboro", zorder=-10)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc = 'upper right',frameon=True,framealpha= 1)

    return ax

if __name__ == '__main__':

    y_pred_score = np.arange(0, 1, 0.001)
    y_label = np.array([1]*25 + [0]*25 + [0]*450 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*50 + [1]*125)

    thresh_group = np.arange(0,1,0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)

    plt.show()
