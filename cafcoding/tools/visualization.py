import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation(corr, colormap="YlGnBu", title=None,figsize=(12, 10), filename=None):
    if colormap is None:
        colormap= "YlGnBu"
    
    if title is None:
        title=""
        
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=figsize)
    plt.title(title, y=1.05, size=16)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
                linewidths=.1, cmap=colormap, cbar_kws={"shrink": .8})
    
    if filename:
        plt.savefig(filename, format='jpeg')
    plt.show()

# def plot_differences(y_test, y_pred, plot_percent=None,title=None,label=None, normalize=True, absolute= False, figsize=(12, 10),filename=None):
#     if plot_percent is None:
#         plot_percent=1.0
    
#     if title is None:
#         title = 'Diferencias normalizadas real/prediccion'
        
#     if label is None:
#         label = 'real-pred'
    
#     LIMIT_PLOT = int(len(y_test)*plot_percent)
#     data = y_test[:LIMIT_PLOT] - y_pred[:LIMIT_PLOT]
    
#     if normalize:
#         data/=max(max(y_test[:LIMIT_PLOT]),max(y_pred[:LIMIT_PLOT]))
    
#     if absolute:
#         data = np.absolute(data)

#     plt.figure(figsize=figsize)
#     plt.plot(range(len(data)),data, label= label)
#     plt.title(title)
#     plt.legend()
#     if filename:
#         plt.savefig(filename, format='jpeg')
#     plt.show()
    


# def plot_two_series(serie1, serie2,label1, label2, title=None, plot_percent=None, normalize=True, figsize=(12, 10),filename=None):
#     if plot_percent is None:
#         plot_percent=1.0
    
#     if title is None:
#         title=""
    
    
#     LIMIT_PLOT = int(len(serie1)*plot_percent)
#     data1 = serie1[:LIMIT_PLOT].copy()
#     data2 = serie2[:LIMIT_PLOT]
    
#     if normalize:
#         max_value=max(max(data1),max(data2))
#         data1/=max_value
#         data2/=max_value
    
#     plt.figure(figsize=figsize)
#     plt.plot(range(len(data1)),data1,label=label1)
#     plt.plot(range(len(data2)),data2,label=label2)
#     plt.title(title)
#     plt.legend()
    
#     if filename:
#         plt.savefig(filename, format='jpeg')
#     plt.show()
    

def plot_multiple_series(series, labels, title=None, plot_percent=None, normalize=True, absolute=False, figsize=(14, 8),filename=None):
    if plot_percent is None:
        plot_percent=1.0
    
    if title is None:
        title=""
    
    
    LIMIT_PLOT = int(len(series[0])*plot_percent)
    data_series=[]
    for serie in series:
        data_series.append(serie[:LIMIT_PLOT].copy())
    
    if normalize:
        max_value = max([max(data) for data in data_series])
        data_series = [data/max_value for data in data_series]
    
    if absolute:
        data_series = np.absolute(data_series)
        
    plt.figure(figsize=figsize)
    for i, data in enumerate(data_series):
        plt.plot(range(len(data)),data,label=labels[i])
    plt.title(title)
    plt.legend()
    
    if filename:
        plt.savefig(filename, format='jpeg')
    plt.show()

def plot_one_series(serie1, label1, title=None, plot_percent=None, normalize=True,absolute= False, figsize=(12, 10),filename=None):
    plot_multiple_series([serie1],[label1],title=title,plot_percent=plot_percent, normalize=normalize,absolute=absolute,figsize=figsize, filename=filename)
    
def plot_two_series(serie1, serie2,label1, label2, title=None, plot_percent=None, normalize=True,absolute= False, figsize=(12, 10),filename=None):
    plot_multiple_series([serie1, serie2],[label1,label2],title=title,plot_percent=plot_percent, normalize=normalize,absolute=absolute,figsize=figsize, filename=filename)

def plot_differences(serie1, serie2,label=None, plot_percent=None,title=None,normalize=True, absolute= False, figsize=(12, 10),filename=None):    
    if label is None:
        label = "Diff"
    plot_multiple_series([serie1-serie2],[label],title=title,plot_percent=plot_percent, normalize=normalize,absolute=absolute,figsize=figsize, filename=filename)



def plot_history_model(history, metrics, title, figsize=(12, 10),  filename=None):
    plt.figure(figsize=figsize)
    for key in metrics:
        plt.plot(history[key])    
        
    plt.title(title)
    
    plt.ylabel('value')
    plt.xlabel('epoch')    
    plt.legend(metrics, loc='upper left')
    if filename:
        plt.savefig(filename, format='jpeg')
    
    plt.show()