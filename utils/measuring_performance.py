import itertools
import numpy as np
from bokeh.io import export_svgs
from bokeh.plotting import figure, show
from bokeh.models import (
    Band,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    NumeralTickFormatter,
)
from bokeh.models.annotations import Label
from collections.abc import Iterable


def get_prediction(score, threshold=0.5):
    return np.where(score >= threshold, 1, 0)


def get_histogram(score, bins=30):
    hist, edges = np.histogram(score, bins=bins)
    percent = list(map(lambda x: x, hist / hist.sum()))
    alpha = hist / hist.sum() + 0.5 * (1.0 - np.max(hist) / hist.sum())

    histogram = dict(
        count=hist, percent=percent, left=edges[:-1], right=edges[1:], alpha=alpha
    )
    histogram["interval"] = [
        f"{left:.2f} to {right:.2f}"
        for left, right in zip(histogram["left"], histogram["right"])
    ]
    return histogram


def plot_confusion_matrix(conf_mat, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plot = figure(
        plot_width=330,
        plot_height=300,
        title=f"{model_name}Confusion Matrix",
        x_axis_label="True Class",
        y_axis_label="Predicted Class",
    )

    mapper = LinearColorMapper(
        palette="Greys256", low=conf_mat.min(), high=conf_mat.max()
    )
    source = ColumnDataSource(
        dict(
            true_class=[0, 1, 0, 1],
            predicted_class=[0, 0, 1, 1],
            n_samples=conf_mat.flatten(),
        )
    )

    plot.rect(
        x="true_class",
        y="predicted_class",
        fill_color={"field": "n_samples", "transform": mapper},
        width=1,
        height=1,
        alpha=0.6,
        line_color="white",
        line_width=1.5,
        source=source,
    )

    for (x_value, y_value) in itertools.product([0, 1], [0, 1]):
        n_samples = str(conf_mat[x_value, y_value])
        x_offset = -3.5 - len(n_samples)
        text_color = (
            "black"
            if (conf_mat[x_value, y_value] - conf_mat.min())
            / (conf_mat.max() - conf_mat.min())
            > 0.5
            else "white"
        )

        label = Label(
            x=x_value,
            y=y_value,
            x_offset=x_offset,
            text=n_samples,
            text_baseline="middle",
            text_color=text_color,
            text_font_size="10px",
            text_font_style="bold",
        )
        plot.add_layout(label)

    plot.grid.grid_line_color = None
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.axis.minor_tick_line_color = None
    plot.xaxis.ticker = [0, 1]
    plot.xaxis.major_label_overrides = {0: "False", 1: "True"}
    plot.yaxis.ticker = [0, 1]
    plot.yaxis.major_label_overrides = {0: "False", 1: "True"}
    plot.title.align = "center"
    show(plot)

    if file_name is not None:
        plot.output_backend = "svg"
        _ = export_svgs(plot, filename=file_name)


def plot_histogram_by_class(
    score_false, score_true, bins=30, model_name=None, file_name=None
):
    if not isinstance(bins, Iterable):
        bins = [bins, bins]
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plot = figure(
        plot_width=600,
        plot_height=400,
        title=f"{model_name}Reconstruction Error Distribution",
        x_axis_label="Reconstruction Error",
        y_axis_label="# Samples",
    )

    source = ColumnDataSource(data=get_histogram(score_false, bins=bins[0]))
    plot.quad(
        bottom=0.0,
        top="percent",
        left="left",
        right="right",
        fill_alpha="alpha",
        fill_color="crimson",
        line_color=None,
        hover_fill_alpha=1.0,
        hover_fill_color="tan",
        legend_label="Normal Signals",
        source=source,
    )

    source = ColumnDataSource(data=get_histogram(score_true, bins=bins[1]))
    plot.quad(
        bottom=0.0,
        top="percent",
        left="left",
        right="right",
        fill_alpha="alpha",
        fill_color="indigo",
        line_color=None,
        hover_fill_alpha=1.0,
        hover_fill_color="tan",
        legend_label="Abnormal Signals",
        source=source,
    )

    plot.yaxis.formatter = NumeralTickFormatter(format="0 %")
    plot.y_range.start = 0.0
    plot.legend.label_text_font_size = "8pt"
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    plot.title.align = "center"
    plot.title.text_font_size = "12pt"

    plot.add_tools(
        HoverTool(
            tooltips=[
                ("interval", "@interval"),
                ("count", "@count"),
                ("percent", "@percent"),
            ]
        )
    )
    show(plot)

    if file_name is not None:
        plot.output_backend = "svg"
        _ = export_svgs(plot, filename=file_name)


def plot_loss_per_epoch(history, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plot = figure(
        plot_width=600,
        plot_height=400,
        title=f"{model_name}Loss per Epoch",
        x_axis_label="# Epochs",
        y_axis_label="Loss",
    )

    source = ColumnDataSource(
        data=dict(
            index=range(len(history.history["loss"])),
            loss=history.history["loss"],
            val_loss=history.history["val_loss"],
        )
    )
    _ = plot.line(
        x="index",
        y="loss",
        color="black",
        line_dash="dotted",
        legend_label="Training Loss",
        source=source,
    )
    _ = plot.line(
        x="index",
        y="val_loss",
        color="coral",
        line_width=1.5,
        legend_label="Validation Loss",
        source=source,
    )

    plot.xgrid.grid_line_color = None
    plot.legend.label_text_font_size = "8pt"
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    plot.title.align = "center"
    plot.title.text_font_size = "12pt"

    plot.add_tools(
        HoverTool(
            tooltips=[
                ("epoch", "@index"),
                ("training loss", "@loss"),
                ("validation loss", "@val_loss"),
            ]
        )
    )
    show(plot)

    if file_name is not None:
        plot.output_backend = "svg"
        _ = export_svgs(plot, filename=file_name)


def plot_pr_curve(pr_curve, auprc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plot = figure(
        plot_width=600,
        plot_height=400,
        title=f"{model_name}Precision - Recall Curve",
        x_axis_label="Recall",
        y_axis_label="Precision",
    )

    source = dict(zip(["recall", "precision", "thr"], pr_curve))
    source["lower_band"] = np.repeat(0.0, source["recall"].shape[0])
    source = ColumnDataSource(source)

    _ = plot.line(
        x="recall",
        y="precision",
        color="coral",
        line_width=1.0,
        legend_label=f"AUPRC: {auprc:.2%}",
        source=source,
    )
    band = Band(
        base="recall",
        lower="lower_band",
        upper="precision",
        level="underlay",
        fill_color="coral",
        fill_alpha=0.2,
        source=source,
    )
    plot.add_layout(band)

    plot.xgrid.grid_line_color = None
    plot.xaxis.formatter = NumeralTickFormatter(format="0%")
    plot.yaxis.formatter = NumeralTickFormatter(format="0%")
    plot.legend.label_text_font_size = "8pt"
    plot.legend.location = "top_right"
    plot.title.align = "center"
    plot.title.text_font_size = "12pt"

    show(plot)

    if file_name is not None:
        plot.output_backend = "svg"
        _ = export_svgs(plot, filename=file_name)


def plot_roc_curve(roc_curve, auroc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plot = figure(
        plot_width=600,
        plot_height=400,
        title=f"{model_name}ROC Curve",
        x_axis_label="False Positive Rate",
        y_axis_label="True Positive Rate",
    )

    source = ColumnDataSource(dict(zip(["fpr", "tpr", "thr"], roc_curve)))

    _ = plot.line(
        x="fpr",
        y="tpr",
        color="coral",
        line_width=1.5,
        legend_label=f"AUROC: {auroc:.2%}",
        source=source,
    )
    _ = plot.line(x="fpr", y="fpr", color="black", line_dash="dashed", source=source)

    plot.xgrid.grid_line_color = None
    plot.xaxis.formatter = NumeralTickFormatter(format="0%")
    plot.yaxis.formatter = NumeralTickFormatter(format="0%")
    plot.legend.label_text_font_size = "8pt"
    plot.legend.location = "bottom_right"
    plot.title.align = "center"
    plot.title.text_font_size = "12pt"

    show(plot)

    if file_name is not None:
        plot.output_backend = "svg"
        _ = export_svgs(plot, filename=file_name)
