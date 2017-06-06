#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import geoutilities.ut_basic as gut_b
import geoutilities.ut_data_loading as gut_dl
import geoutilities.ut_distributions as gut_dist
import geoutilities.ut_taskmanager as gut_tm
import geoutilities.ut_graphing as gut_gr

from constants import Constants

class LTFile:
    def __init__(self, name, filename, sheetname, time_range):
        self.name = name
        self.filename = filename
        self.sheetname = sheetname
        self.time_range = time_range


class TaskManager (gut_tm.TaskManager):

    lt_file2 = 'Qs_t2990-3110.xlsx'

    lt_2870_meta = LTFile(
            name = 't2870',
            filename = 'Qs_t2870-2990.xlsx',
            sheetname = 'Qs_t2870 LT',
            time_range = '2870-2990')

    lt_2990_meta = LTFile(
            name = 't2870',
            filename = 'Qs_t2990-3110.xlsx',
            sheetname = 'Qs_t2990-3110',
            time_range = '2990-3110')

    def __init__(self):
        self.grapher = gut_gr.Grapher()
        self.plots_pending = False
        self.all_plots = False

        gut_b.VERBOSE = Constants.verbose

        options = [
                ('--plot-all', self.plot_all, 'Plot distributions.'),
                ('--c-distr', self.compare_gsd, 'Perform grain size distribution comparisons.'),
                ]
        gut_tm.TaskManager.__init__(self, options)

        if self.plots_pending:
            self.grapher.show_plots()

    def load_data(self, reload=False):
        self.loader = gut_dl.DataLoader(Constants.data_path)

        names = Constants.pickle_names
        if Constants.reload_data or reload or not self.loader.is_pickled(names.values()):
            gut_b.printer("Pickles do not exist (or forced reloading). Loading excel files...")
            self.load_peak_data()
        else:
            gut_b.printer(" Pickles present! Unpacking pickles...")
            self.hs_data = self.loader.load_pickle(names['hs'])
            self.lt_data = self.loader.load_pickle(names['lt'])
            gut_b.printer(" Pickles unpacked!")

    def load_peak_data(self):
        loader = gut_dl.DataLoader(data_path = Constants.data_path)
        gut_b.printer("Loading data...")

        self.load_helly_smith(loader)
        self.load_light_table(loader)
        gut_b.printer(" Done loading files!")

        prepickles = {Constants.pickle_names['hs'] : self.hs_data,
                      Constants.pickle_names['lt'] : self.lt_data}
        self.loader.produce_pickles(prepickles)

    def load_helly_smith(self, loader):
        ## Helly Smith Data
        gut_b.printer(" Loading Helly Smith data...")
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
        gut_b.printer("  Cleaning Helly Smith data...")
        
        # Sort data so indexing works
        hs_data.sort_index(axis=0, inplace=True)

        gut_b.printer("   Dropping unnecessary columns...")
        hs_data.drop(['Date', 'ID', 32], axis=1, inplace=True)
        
        # Select rows 't2890', 't2950', 't3010', 't3070' for my analysis
        gut_b.printer("   Selecting times between 2870 to 3110...")
        hs_data = hs_data.loc[pd.IndexSlice['t2890':'t3070',:],:]

        gut_b.printer("   Reformatting labels...")
        index_labels = hs_data.index.values
        hs_data.index = pd.MultiIndex.from_tuples(index_labels)

        hs_data= hs_data.T.iloc[::-1] # flip order
        self.hs_data = hs_data
        gut_b.printer("  Done with Helley Smith data!")

    def load_light_table(self, loader):
        ## Light table data
        # To prepare the light table files for this program:
        # - -
        printer = gut_b.printer

        printer(" Loading light table data...")

        # Load files
        lt_kwargs = {
                'sheetname'  : None,
                'skiprows'   : 2,
                'header'     : 0,
                'skip_footer': 4,
                'index_col'  : 0,
                'parse_cols' : 44,
                'na_values'  : 'ALL PAINT', 
                }

        files = [TaskManager.lt_2870_meta,
                 TaskManager.lt_2990_meta,
                ]

        lt_partials = []
        first = True
        last_index_max = 0
        for meta in files:
            name = meta.name
            filename = meta.filename
            sheetname = meta.sheetname
            time_range = meta.time_range

            printer("  Loading light table {} data...".format(time_range))
            lt_kwargs['sheetname'] = sheetname
            lt_partial = loader.load_xlsx(filename, lt_kwargs)

            # Clean data
            printer("  Cleaning light table {} data...".format(time_range))
            
            # Sort data so indexing works
            lt_partial.sort_index(axis=0, inplace=True)
            #lt_partial.sort_index(axis=1, inplace=True)

            printer("   Dropping unnecessary columns ({})...".format(name))
            # Note that with pandas.load_xlsx, repeated column names get a 
            # ".{x}" added to the end, where x is the number of repetitions. So 
            # if the excel file has two colums named "0.71", then the first 
            # column is "0.71" and the second one is "0.71.1" and third would 
            # be "0.71.2" and so on.
            drop_list = ['time sec', 'missing ratio', 'vel', 'sd vel',
                    'number vel', 'count stones'] + ["{}.1".format(gs) for gs in [
                    "0.5", "0.71", "1", "1.4", "2", "2.8", "4", "5.6",
                    "8", "11.2", "16", "22", "32", "45"]]
            lt_partial.drop(drop_list, axis=1, inplace=True)

            printer("   Reformatting labels...")
            lt_partial.rename(columns={'Bedload transport':'Total (g)'}, inplace=True)
            print(lt_partial.columns)

            if first:
                first = False
            else:
                # Drop the first row of data if not the first dataset; to 
                # prevent overlapping rows.
                lt_partial.drop(0, axis=0, inplace=True)

            printer("  Resetting index values...")
            partial_times = lt_partial.index.values + last_index_max
            last_index_max = np.max(partial_times)
            lt_partial.index = pd.Index(partial_times)

            # Save the partial data to a list
            lt_partials.append(lt_partial)

        printer("  Combining data into one structure...")
        lt_combined = pd.concat(lt_partials)

        self.lt_data = lt_combined
        printer("  Done with light table data!")


    def plot_all(self):
        self.all_plots = True

    def compare_gsd(self):
        # Things to do:
        #  Generate 5 minute windows of average LT data
        #  compare HS to moving LT window
        #  | start at HS time? or calc estimated lag time?
        #  | match both distribution and total mass
        #  pick "best fit" times
        #  generate graphs for human review
        #  repeat for each HS time step
        #
        #  Note: plots will not be shown unless --plot-all option given

        self.plot_lt_totals()

        # Set up HS mather
        self.hs_mather = gut_dist.PDDistributions(self.hs_data, Constants.hs_max_size)

        #  Average the 5 HS data for each time step
        time_sums, time_cumsums = self.calc_overall_sampler()
        self.plot_HS_averages(time_cumsums)

        # Prepare rolling LT data
        self.gen_lt_windows()
        self.plot_windowed_lt()

        # Compare HS to traveling window
        lt_windows = self.windowed_lt.T
        hs_time_sums = time_sums
        compare = gut_dist.PDDistributions.compare_distributions
        max_size = max(Constants.hs_max_size, np.amax(Constants.lt_size_classes))
        
        self.difference = compare(lt_windows, hs_time_sums, 0, max_size)
        
        # Remove values that occur before the sampling time
        hs_sample_times_str = self.difference.columns.values
        hs_sample_times = Constants.hs_sample_times
        for str, time in zip(hs_sample_times_str, hs_sample_times):
            self.difference.loc[:time,str] = np.NAN

        self.plot_difference(force=True)


    def gen_lt_windows(self):
        # Generate 5 minute LT windows
        #
        # Can skip up to tolerance blank rows
        n_sec = Constants.window_duration
        tolerance =  int(n_sec * Constants.window_tolerance)
        lt_data = self.lt_data.loc[1:, Constants.lt_size_classes]

        self.window_roll =  lt_data.rolling(window=n_sec, min_periods=n_sec-tolerance)
        self.windowed_lt = self.window_roll.sum().loc[n_sec:]

    def calc_overall_sampler(self):
        # Calculate the overall distribution from the selected columns
        # 
        # Sum the values in each grain size class then calc new cumsum 
        # distributions
        #
        # Raw values are summed b/c normalized distributions would need 
        # weights, which are based on the raw values anyway.
        
        data = self.hs_mather.data
        cumsum = self.hs_mather.cumsum
        sizes = data.index
        times = cumsum.columns.levels[0]
        time_sums = pd.DataFrame(index=sizes, columns=times).fillna(0)
        
        # Sum the masses from different samplers at each timestep
        # Gets rid of the MultiIndex too.
        for time in times:
            # Pick the data subset
            slicer_key = (slice(None)),(time, slice(1,5))
            data_slice = data.loc[slicer_key]
            
            # sum up the masses in each size class
            time_sums[time] = data_slice.sum(axis=1)

        # calculate the new normalized cumsum
        time_cumsums = self.hs_mather.calc_normalized_cumsum(data=time_sums)

        return time_sums, time_cumsums


    def plot_check(self, force=False):
        if self.all_plots or force:
            self.plots_pending = True
            return True
        else:
            return False

    def plot_HS_averages(self, time_cumsums, force=False):
        # Plot the averaged HS distributions to see what they looks like
        if not self.plot_check(force): return

        hsm = self.hs_mather
        title = "Summed Helley-Smith sampler distributions"
        self.grapher.pd_plot_cumsum(time_cumsums,
                xticks=hsm.class_geometric_means, title=title)

    def plot_windowed_lt(self, force=False):
        if not self.plot_check(force): return
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

    def plot_lt_totals(self, force=False):
        if not self.plot_check(force): return
        n_sec = Constants.window_duration
        tolerance =  int(n_sec * Constants.window_tolerance)

        lt_data = self.lt_data.loc[:,'Total (g)']
        window =  lt_data.rolling(window=n_sec, min_periods=n_sec-tolerance)
        mean = window.mean().loc[n_sec:]

        fig = plt.figure()
        axis = plt.gca()
        lt_data.plot(ax=axis, color='r')
        mean.plot(ax=axis, color='b')

    def plot_difference(self, force=False):
        if not self.plot_check(force): return
        title = 'Difference index for Helley-Smith and light table distributions'
        fig, axes = plt.subplots(5, sharex=True) 
        flux_ax  = axes[0]
        diff_axes = axes[1:]

        diff_data = self.difference
        times = diff_data.index.values
        hs_sample_times_str = diff_data.columns.values
        hs_sample_times = Constants.hs_sample_times

        # Plot total flux for convenience
        n_sec = Constants.window_duration
        tolerance =  int(n_sec * Constants.window_tolerance)
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

        # Plot the difference values
        ymin, ymax = 0.2, 1.8
        for ax, hsst_str, hsst in zip(diff_axes, hs_sample_times_str, hs_sample_times):
            diff_data[hsst_str].plot(ax=ax, color='k')
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
        min_start = Constants.experiment_start_time
        tmin_min = tmin/60 + min_start
        tmax_min = tmax/60 + min_start
        min_ax.set_xlim((tmin_min, tmax_min))
        min_ax.set_xlabel("Experiment Time (min)")
        



if __name__ == "__main__":
    manager = TaskManager()
