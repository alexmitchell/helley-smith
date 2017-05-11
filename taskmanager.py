#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mplmarker
import pandas as pd
import scipy.stats as scstats
#import probability_plots as pbp

import utilities as ut
from utilities import printer

VERBOSE = True
RELOAD_DATA = False

class TaskManager:
    # Data location
    data_path="./data/"
    pickle_names = {'hs':'helly_smith', 'lt':'light_table'}
    hs_max_size = 32 #mm
    lt_size_classes = 0.5, 0.71, 1, 1.4, 2, 2.8, 4, 5.6, 8, 11.2, 16, 22, 32, 45
    window_duration = 60 * 5 # 5min windows in seconds
    window_tolerance = 0.15 # max percent of window that can be skipped when doing rolling calculations
    experiment_start_time = 2870
    hs_sample_times = (np.array([2890,2950,3010,3070]) - 2870) * 60 # Feeling too lazy to parse this from the HS data labels

    def __init__(self):
        self.grapher = ut.Grapher()
        self.loader = ut.DataLoader(Project.data_path)

        names = Project.pickle_names
        if RELOAD_DATA or not self.loader.is_pickled(names.values()):
            printer("Pickles do not exist (or forced reloading). Loading excel files...")
            self.load_peak_data()
        else:
            printer(" Pickles present! Unpacking pickles...")
            self.hs_data = self.loader.load_pickle(names['hs'])
            self.lt_data = self.loader.load_pickle(names['lt'])
            printer(" Pickles unpacked!")

        self.do_science()


    def load_peak_data(self):
        loader = ut.DataLoader(data_path = Project.data_path)
        printer("Loading data...")

        self.load_helly_smith(loader)
        self.load_light_table(loader)
        printer(" Done loading files!")

        prepickles = {Project.pickle_names['hs'] : self.hs_data,
                      Project.pickle_names['lt'] : self.lt_data}
        self.loader.produce_pickles(prepickles)

    def load_helly_smith(self, loader):
        ## Helly Smith Data
        printer(" Loading Helly Smith data...")
        hs_kwargs = {
                'sheetname'  : 'Sheet1',
                'header'     : 0,
                'skiprows'   : 1,
                'index_col'  : [2, 3],
                'parse_cols' : 19,
                'na_values'  : 'ALL PAINT', 
                }
        hs_data = loader.load_xlsx('helly_smith_data.xlsx', hs_kwargs)

        # Clean data
        printer("  Cleaning Helly Smith data...")
        
        # Sort data so indexing works
        hs_data.sort_index(axis=0, inplace=True)

        printer("   Dropping unnecessary columns...")
        hs_data.drop(['Date', 'ID', 32], axis=1, inplace=True)
        
        # Select rows 't2890', 't2950', 't3010', 't3070' for my analysis
        printer("   Selecting times between 2870 to 3110...")
        hs_data = hs_data.loc[pd.IndexSlice['t2890':'t3070',:],:]

        printer("   Reformatting labels...")
        index_labels = hs_data.index.values
        hs_data.index = pd.MultiIndex.from_tuples(index_labels)

        self.hs_data = hs_data
        printer("  Done with Helley Smith data!")

    def load_light_table(self, loader):
        ## Light table data
        printer(" Loading light table data...")

        # Load files
        printer("  Loading light table 2870-2990 data...")
        lt_kwargs = {
                'sheetname'  : 'Qs_t2870 LT',
                'skiprows'   : 3,
                'header'     : 0,
                'skip_footer': 4,
                'index_col'  : 0,
                'parse_cols' : 44,
                'na_values'  : 'ALL PAINT', 
                }
        lt_2870 = loader.load_xlsx('Qs_t2870-2990.xlsx', lt_kwargs)

        printer("  Loading light table 2990-3110 data...")
        lt_kwargs['sheetname'] = 'Qs_t2990-3110'
        lt_2990 = loader.load_xlsx('Qs_t2990-3110.xlsx', lt_kwargs)

        # Clean data
        printer("  Cleaning light table data...")
        for ltd, name in zip([lt_2870, lt_2990],
                            ['2870-2990', '2990-3110']):
            # Sort data so indexing works
            ltd.sort_index(axis=0, inplace=True)
            #ltd.sort_index(axis=1, inplace=True)

            printer("   Dropping unnecessary columns ({})...".format(name))
            drop_list = ['time sec', 'missing ratio', 'vel', 'sd vel',
                    'number vel'] + ["cs_{}".format(gs) for gs in [
                    "Total", "0.5", "0.71", "1", "1.4", "2", "2.8", "4", "5.6",
                    "8", "11.2", "16", "22", "32", "45"]]
            ltd.drop(drop_list, axis=1, inplace=True)

            printer("   Reformatting labels...")
            ltd.rename(columns={'Total':'Total (g)'}, inplace=True)

        # Drop the first row of lt_2990.
        lt_2990.drop(0, axis=0, inplace=True)

        printer("  Combining data into one structure...")

        times_2870 = lt_2870.index.values
        times_2990 = lt_2990.index.values + np.max(times_2870)

        lt_2990.index = pd.Index(times_2990)

        lt_combined = pd.concat([lt_2870, lt_2990])

        self.lt_data = lt_combined
        printer("  Done with light table data!")


    def do_science(self):
        # Things to do:
        #  Generate 5 minute windows of average LT data
        #  compare HS to moving LT window
        #  | start at HS time? or calc estimated lag time?
        #  | match both distribution and total mass
        #  pick "best fit" times
        #  generate graphs for human review
        #  repeat for each HS time step

        #self.plot_lt_totals()

        # Set up HS mather
        hs_distribution = self.hs_data.T.iloc[::-1] # flip order
        self.hs_mather = ut.DistributionMather(hs_distribution, Project.hs_max_size)

        #  Average the 5 HS data for each time step
        self.hs_mather.calc_overall_sampler()
        #self.plot_HS_averages()

        # Prepare rolling LT data
        self.gen_lt_windows()
        #self.plot_windowed_lt()

        # Compare HS to traveling window
        lt_windows = self.windowed_lt.T
        hs_time_sums = self.hs_mather.time_sums
        compare = ut.DistributionMather.compare_distributions
        max_size = max(Project.hs_max_size, np.amax(Project.lt_size_classes))
        
        self.similarity = compare(lt_windows, hs_time_sums, 0, max_size)
        
        # Remove values that occur before the sampling time
        hs_sample_times_str = self.similarity.columns.values
        hs_sample_times = Project.hs_sample_times
        for str, time in zip(hs_sample_times_str, hs_sample_times):
            self.similarity.loc[:time,str] = np.NAN

        self.plot_similarity()

        #plt.show()

    def gen_lt_windows(self):
        # Generate 5 minute LT windows
        #
        # Can skip up to tolerance blank rows
        n_sec = Project.window_duration
        tolerance =  int(n_sec * Project.window_tolerance)
        lt_data = self.lt_data.loc[1:, Project.lt_size_classes]

        self.window_roll =  lt_data.rolling(window=n_sec, min_periods=n_sec-tolerance)
        self.windowed_lt = self.window_roll.sum().loc[n_sec:]


    def plot_HS_averages(self):
        # Plot the averaged HS distributions to see what they looks like
        hsm = self.hs_mather
        title = "Summed Helley-Smith sampler distributions"
        self.grapher.pd_plot_cumsum(hsm.time_cumsums,
                xticks=hsm.class_geometric_means, title=title, show=True)

    def plot_windowed_lt(self):
        title = 'Light table mass fluxes after a five-minute rolling sum'
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True) 
        
        windowed = self.windowed_lt
        gs = windowed.columns.values
        self.grapher.pd_plot(windowed.loc[:,gs[0:7]], title=title, axis=ax1)
        self.grapher.reset_line_format_iter()
        self.grapher.pd_plot(windowed.loc[:,gs[7:14]], axis=ax2)
        for ax in ax1, ax2,:
            # Split into two plots for clarity
            ax.set_ylim((5,12000))
            ax.set_yscale("log", nonposy='clip')
            ax.set_ylabel("Mass Flux (g / 5-minutes)")
            ax.set_xlim((0,16000))
            ax.grid(which='major', axis='x')
            ax.legend(loc='center right')
        ax1.set_title("Light table mass flux for each grain size class")
        ax2.set_xlabel("Time (s)")
        fig.subplots_adjust(hspace=.01)

        plt.show()

    # Functions to write (in this file or in utility file)
    # average_distributions(*distributions)
    # compare_distributions(dist_1, dist_2)

    def plot_lt_totals(self):
        n_sec = Project.window_duration
        tolerance =  int(n_sec * Project.window_tolerance)

        lt_data = self.lt_data.loc[:,'Total (g)']
        window =  lt_data.rolling(window=n_sec, min_periods=n_sec-tolerance)
        mean = window.mean().loc[n_sec:]

        fig = plt.figure()
        axis = plt.gca()
        lt_data.plot(ax=axis, color='r')
        mean.plot(ax=axis, color='b')
        #plt.show()

    def plot_similarity(self):
        title = 'Difference index for Helley-Smith and light table distributions'
        fig, axes = plt.subplots(5, sharex=True) 
        flux_ax  = axes[0]
        sim_axes = axes[1:]

        sim_data = self.similarity
        times = sim_data.index.values
        hs_sample_times_str = sim_data.columns.values
        hs_sample_times = Project.hs_sample_times

        # Plot total flux for convenience
        n_sec = Project.window_duration
        tolerance =  int(n_sec * Project.window_tolerance)
        lt_total = self.lt_data[['Total (g)', 'D50']]
        all_roll =  lt_total.rolling(window=n_sec,
                min_periods=n_sec-tolerance)

        total_roll = all_roll.sum()['Total (g)'] / 1000
        total_roll.plot(ax=flux_ax, linestyle='solid', color='r')
        flux_ax.tick_params('y', colors='r')
        #flux_ax.semilogy(nonposy='clip')
        flux_ax.set_ylabel('Total flux\n(kg / 5-minutes)', color='r')
        flux_min, flux_max = flux_ax.get_ylim()
        flux_ax.vlines(hs_sample_times, flux_min, flux_max, linestyle='dashed', label='Sampling time')

        # Plot D50 for convenience
        D50_ax = flux_ax.twinx()
        smoothed_D50 = all_roll.mean()['D50']
        smoothed_D50.plot(ax=D50_ax, linestyle='solid', color='b')
        D50_ax.set_ylabel('Smoothed D50 (mm)', color='b')
        D50_ax.tick_params('y', colors='b')

        # Plot the similarity values
        ymin, ymax = 0.2, 1.8
        for ax, hsst_str, hsst in zip(sim_axes, hs_sample_times_str, hs_sample_times):
            sim_data[hsst_str].plot(ax=ax, color='k')
            ax.vlines(hsst, ymin, ymax, linestyle='dashed', label='Sampling time')
            ax.set_ylim((ymin, ymax))
            ax.set_ylabel("{}\nDifference index".format(hsst_str))
            ax.set_yticks(ax.get_yticks()[::2])
            ax.grid(which='major', axis='y')

        fig.suptitle("Difference index for each Helley-Smith sample time")
        fig.subplots_adjust(hspace=.2)
        tmin = 0
        tmax = np.amax(times)
        axes[-1].set_xlim((tmin, tmax))
        axes[-1].set_xlabel("Time (s)")
        
        # Plot time in minutes at top
        min_ax = flux_ax.twiny()
        min_start = Project.experiment_start_time
        tmin_min = tmin/60 + min_start
        tmax_min = tmax/60 + min_start
        min_ax.set_xlim((tmin_min, tmax_min))
        min_ax.set_xlabel("Experiment Time (min)")
        

        plt.show()



if __name__ == "__main__":
    manager = TaskManager()
