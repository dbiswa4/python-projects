import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *

'''
http://pbpython.com/visualization-tools-1.html
'''

def pandas_plot(data):
    # Now, setup our display to use nicer defaults and create a bar plot:

    #pd.options.display.matplotlib.pyplot.style.use = 'default'
    pd.options.display.mpl_style = 'default'
    data_plot = data.plot(kind="bar",x=data["detail"],
                          title="MN Capital Budget - 2014",
                          legend=False)

    #How the hell it knows to put amount in y axis?

    fig = data_plot.get_figure()
    fig.savefig("2014-mn-capital-budget.png")

def seaborn_plot(data):
    '''
    Seaborn is a visualization library based on matplotlib. It seeks to make default data visualizations much more visually appealing.
    It also has the goal of making more complicated plots simpler to create. It does integrate well with pandas.
    '''

    sns.set_style("darkgrid")
    #title is not a valid option in this method. Use matplotlib to add title
    bar_plot = sns.barplot(x=data["detail"], y=data["amount"],
                           palette="muted",
                           order=data["detail"].tolist())

    '''
    I had to use matplotlib to rotate the x axis titles so I could actually read them. Visually, the display looks nice
    '''
    plt.xticks(rotation=90)
    plt.title("MN Capital Budget - 2014");
    plt.show()

    fig = bar_plot.get_figure()
    fig.savefig("2014-mn-capital-budget-seaborn.png")


def ggplot_plot(data):
    p = ggplot(data, aes(x="detail", y="amount")) + \
        geom_bar(stat="bar", labels=data["detail"].tolist()) + \
        ggtitle("MN Capital Budget - 2014") + \
        xlab("Spending Detail") + \
        ylab("Amount") + scale_y_continuous(labels='millions') + \
        theme(axis_text_x=element_text(angle=90))
    print p
    #ggsave(p, "mn-budget-capital-ggplot.png")


if __name__ == '__main__':
    print 'Execution started'
    budget = pd.read_csv("mn-budget-detail-2014.csv")
    budget = budget.sort_values(by='amount', ascending=False)[:10]
    print budget

    #pandas_plot(budget);
    #seaborn_plot(budget);
    ggplot_plot(budget);
