import matplotlib.pyplot as plt
from functools import partial
import numpy as np
#TODO: work on 3d functionality

class InteractivePlotter:

    def __init__(self):
        """
        this class implements an interface for interactive plotting, where information about datapoints is
        shown when the user either clicks on / hovers over them with the mouse
        """
        pass

    @staticmethod
    def _text_wrapper(text, wrap_len=None):
        wrapped_text = ""
        wrapped_text+=text[0]
        if wrap_len:
            for i in range(1, len(text)):
                if i%wrap_len == 0:
                    wrapped_text += "\n"
                wrapped_text += text[i]
            return wrapped_text
        return text

    @staticmethod
    def _hover_callback(event, annotation=None, axis=None, scatter=None, figure=None, labels=None,
                        text_wrap=None):
        vis = annotation.get_visible()
        if event.inaxes == axis:
            cont, ind = scatter.contains(event)
            if cont:
                pos = scatter.get_offsets()[ind["ind"][0]]
                annotation.xy = pos
                text = InteractivePlotter._text_wrapper("".join(labels[ind["ind"][0]]), text_wrap)
                annotation.set_text(text)
                annotation.set_visible(True)
                figure.canvas.draw_idle()
            else:
                if vis:
                    annotation.set_visible(False)
                    figure.canvas.draw_idle()

    @staticmethod
    def _click_callback(event, annotation=None, figure=None, labels=None,
                        text_wrap=None):
        """
        This function defines the behaviour of the plotter in the "click" mode
        """
        annotation.set_visible(False)
        ind = event.ind
        xy = event.artist.get_offsets()[ind][0].tolist()
        annotation.xy = xy
        text = InteractivePlotter._text_wrapper("".join(labels[ind.item()]), text_wrap)
        annotation.set_text(text)
        annotation.set_visible(True)
        figure.canvas.draw_idle()

    @staticmethod
    def plot(x_data, y_data, labels, mode="click", text_wrap=None, plot_args={}):
        #TODO: add some additional styling options for the text display
        """
        This function implements the main functionality of the plotter
        params:
            plot_args provides optional arguments that can be passed to the
            scatter function to customize the style

        """
        fig, ax = plt.subplots()
        sc = ax.scatter(x_data, y_data, **plot_args, picker=2)

        annot = ax.annotate(" ", xy=(0, 0), xytext=(10, 10),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"))
        annot.set_visible(False)

        if mode == "click":
            fig.canvas.callbacks.connect('pick_event', partial(InteractivePlotter._click_callback,
                                                               annotation=annot, figure=fig, labels=labels,
                                                               text_wrap=text_wrap))
        elif mode == "hover":
            fig.canvas.mpl_connect("motion_notify_event", partial(InteractivePlotter._hover_callback,
                                                                  annotation=annot, axis=ax, scatter=sc, figure=fig,
                                                                  labels=labels, text_wrap=text_wrap))
        else:
            raise(Exception("please make sure the mode parameter is either 'click' or 'hover'"))
        return {"fig": fig, "axis": ax, "plot": sc}


# X = [1, 2, 3]
# Y = [4, 5, 6]
# labels = ["point 1", "point 2", "point 3"]
# InteractivePlotter.plot(X, Y, labels)
# plt.show()
