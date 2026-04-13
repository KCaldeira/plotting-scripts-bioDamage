
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():

    #### Figure 1 and 2, column 1
    from simple_scripts.Fig1_2_col1 import plot_figure1_2_col1
    plot_figure1_2_col1('main')
    plot_figure1_2_col1('SI')

    #### Figure 1 and 2, column 2
    from simple_scripts.Fig1_2_col2 import plot_figure1_2_col2
    plot_figure1_2_col2('main')
    plot_figure1_2_col2('SI')

    #### Figure 1 and 2, column 3, candidate all countries sorted by TAS 
    from simple_scripts.Fig1_2_col3 import plot_figure1_2_col3
    plot_figure1_2_col3('main')
    plot_figure1_2_col3('SI')

    #### Figure 3 
    from simple_scripts.Fig3_plot_scatter import fig3_plot_scatter
    fig3_plot_scatter()
    from simple_scripts.Fig3_plot_violin import fig3_plot_violin
    fig3_plot_violin()
    from simple_scripts.Fig3_plot_map import fig3_plot_map
    fig3_plot_map()

if __name__ == "__main__":
    main() 