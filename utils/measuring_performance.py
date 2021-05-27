import itertools
import numpy as np
from bokeh.io import export_svgs
from bokeh.plotting import figure, show
from bokeh.models import Band, ColumnDataSource, HoverTool, LinearColorMapper, NumeralTickFormatter
from bokeh.models.annotations import Label
from collections.abc import Iterable


def get_prediction(score, threshold=0.5):
    return np.where(score >= threshold, 1, 0)


def get_histogram(score, bins=30):
    hist, edges = np.histogram(score, bins=bins)
    percent = list(map(lambda x: x, hist / hist.sum()))
    alpha = hist / hist.sum() + 0.5 * (1.0 - np.max(hist) / hist.sum())

    histogram = dict(count=hist, percent=percent, left=edges[:-1], right=edges[1:], alpha=alpha)
    histogram['interval'] = ['{0:.2f} to {1:.2f}'.format(left, right)
                             for left, right in zip(histogram['left'], histogram['right'])]
    return histogram


def plot_confusion_matrix(conf_mat, model_name, file_name):
    if model_name is None:
        model_name = ''
    else:
        model_name += ': '

    p = figure(plot_width=330, plot_height=300, title='{}Confusion Matrix'.format(model_name),
               x_axis_label='True Class', y_axis_label='Predicted Class')

    mapper = LinearColorMapper(palette='Greys256', low=conf_mat.min(), high=conf_mat.max())
    source = ColumnDataSource(dict(true_class=[0, 1, 0, 1], predicted_class=[0, 0, 1, 1], n_samples=conf_mat.flatten()))

    p.rect(x='true_class', y='predicted_class', fill_color={'field': 'n_samples', 'transform': mapper},
           width=1, height=1, alpha=0.6, line_color='white', line_width=1.5, source=source)

    for (x, y) in itertools.product([0, 1], [0, 1]):
        n_samples = str(conf_mat[x, y])
        x_offset = -3.5 - len(n_samples)
        text_color = 'black' if (conf_mat[x, y] - conf_mat.min()) / (conf_mat.max() - conf_mat.min()) > 0.5 else 'white'

        label = Label(x=x, y=y, x_offset=x_offset, text=n_samples, text_baseline='middle', text_color=text_color,
                      text_font_size='10px', text_font_style='bold')
        p.add_layout(label)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.ticker = [0, 1]
    p.xaxis.major_label_overrides = {0: 'False', 1: 'True'}
    p.yaxis.ticker = [0, 1]
    p.yaxis.major_label_overrides = {0: 'False', 1: 'True'}
    p.title.align = 'center'
    show(p)

    if file_name is not None:
        p.output_backend = 'svg'
        _ = export_svgs(p, filename=file_name)


def plot_histogram_by_class(score_false, score_true, bins=30, model_name=None, file_name=None):
    if not isinstance(bins, Iterable):
        bins = [bins, bins]
    if model_name is None:
        model_name = ''
    else:
        model_name += ': '
        
    p = figure(plot_width=600, plot_height=400,
               title='{}Reconstruction Error Distribution'.format(model_name),
               x_axis_label='Reconstruction Error', y_axis_label='# Samples')

    source = ColumnDataSource(data=get_histogram(score_false, bins=bins[0]))
    p.quad(bottom=0.0, top='percent', left='left', right='right', 
           fill_alpha='alpha', fill_color='crimson', line_color=None,
           hover_fill_alpha=1.0, hover_fill_color='tan', legend_label='Normal Signals', source=source)

    source = ColumnDataSource(data=get_histogram(score_true, bins=bins[1]))
    p.quad(bottom=0.0, top='percent', left='left', right='right', 
           fill_alpha='alpha', fill_color='indigo', line_color=None,
           hover_fill_alpha=1.0, hover_fill_color='tan', legend_label='Abnormal Signals', source=source)

    p.yaxis.formatter = NumeralTickFormatter(format='0 %')
    p.y_range.start = 0.0
    p.legend.label_text_font_size = '8pt'
    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'
    p.title.align = 'center'
    p.title.text_font_size = '12pt'

    p.add_tools(HoverTool(tooltips=[('interval', '@interval'), ('count', '@count'), ('percent', '@percent')]))
    show(p)
    
    if file_name is not None:
        p.output_backend = 'svg'
        _ = export_svgs(p, filename=file_name)


def plot_loss_per_epoch(history, model_name=None, file_name=None):
    if model_name is None:
        model_name = ''
    else:
        model_name += ': '
        
    p = figure(plot_width=600, plot_height=400, title='{}Loss per Epoch'.format(model_name),
               x_axis_label='# Epochs', y_axis_label='Loss')

    source = ColumnDataSource(data=dict(index=range(len(history.history['loss'])),
                                        loss=history.history['loss'], val_loss=history.history['val_loss']))
    _ = p.line('index', 'loss', color='black', line_dash='dotted', legend_label='Training Loss', source=source)
    _ = p.line('index', 'val_loss', color='coral', line_width=1.5, legend_label='Validation Loss', source=source)

    p.xgrid.grid_line_color = None
    p.legend.label_text_font_size = '8pt'
    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'
    p.title.align = 'center'
    p.title.text_font_size = '12pt'

    p.add_tools(HoverTool(tooltips=[('epoch', '@index'), ('training loss', '@loss'), ('validation loss', '@val_loss')]))
    show(p)
    
    if file_name is not None:
        p.output_backend = 'svg'
        _ = export_svgs(p, filename=file_name)


def plot_pr_curve(pr_curve, auprc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ''
    else:
        model_name += ': '

    p = figure(plot_width=600, plot_height=400, title='{}Precision - Recall Curve'.format(model_name),
               x_axis_label='Recall', y_axis_label='Precision')

    source = dict(zip(['recall', 'precision', 'thr'], pr_curve))
    source['lower_band'] = np.repeat(0.0, source['recall'].shape[0])
    source = ColumnDataSource(source)

    _ = p.line(source=source, x='recall', y='precision', color='coral', line_width=1.0,
               legend_label='AUPRC: {:.2%}'.format(auprc))
    band = Band(base='recall', lower='lower_band', upper='precision', level='underlay', fill_color='coral',
                fill_alpha=0.2, source=source)
    p.add_layout(band)

    p.xgrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format='0%')
    p.yaxis.formatter = NumeralTickFormatter(format='0%')
    p.legend.label_text_font_size = '8pt'
    p.legend.location = 'top_right'
    p.title.align = 'center'
    p.title.text_font_size = '12pt'

    show(p)

    if file_name is not None:
        p.output_backend = 'svg'
        _ = export_svgs(p, filename=file_name)


def plot_roc_curve(roc_curve, auroc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ''
    else:
        model_name += ': '

    p = figure(plot_width=600, plot_height=400, title='{}ROC Curve'.format(model_name),
               x_axis_label='False Positive Rate', y_axis_label='True Positive Rate')

    source = ColumnDataSource(dict(zip(['fpr', 'tpr', 'thr'], roc_curve)))

    _ = p.line('fpr', 'tpr', color='coral', line_width=1.5,
               legend_label='AUROC: {:.2%}'.format(auroc), source=source)
    _ = p.line('fpr', 'fpr', color='black', line_dash='dashed', source=source)

    p.xgrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format='0%')
    p.yaxis.formatter = NumeralTickFormatter(format='0%')
    p.legend.label_text_font_size = '8pt'
    p.legend.location = 'bottom_right'
    p.title.align = 'center'
    p.title.text_font_size = '12pt'

    show(p)

    if file_name is not None:
        p.output_backend = 'svg'
        _ = export_svgs(p, filename=file_name)
