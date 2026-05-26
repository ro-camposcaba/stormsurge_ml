Below brief explanations of the performance scripts are provided:

Performance evaluation by location:

- performance_traintestTG_v1.ipynb: evaluates a selected run from emulator using the testing output, obtaining performance metrics and density scatter plot.

- process_error_metrics_all_emulators.ipynb: process all the test outputs, of all the runs, of all the emulators launched. As a results, an excel file is obtained, which sheet names correspond to the names provided to each folder of each emulator launched. Each sheet contains the performance metrics of the emulators for each run.

- process_error_metrics_peaks99_all_emulators.ipynb: same as "process_error_metrics_all_emulators.ipynb", but applied to the surge peaks above the 99th percentile (or any other percentile desired).

- process_peaks_bootstrap_2026.ipynb: applies the bootstrap analysis for surge peaks above the 99th percentile (or any other percentile desired) of all the runs for a specific emulator, with mean metric difference relative to the numerical model (SHYFEM-MPI). On its current version, it works for all the runs of a single emulator.


Averaged performance across locations:

- process_average_metrics_all_emulators.ipynb: loads the excel files obtained with "process_error_metrics_peaks99_all_emulators.ipynb" at each location and average them for all the runs of each emulator.

- process_average_peaks99_metrics_all_emulators.ipynb: same as "process_average_metrics_all_emulators.ipynb", but for the surge peaks above 99th percentile analysis.

- plot_violinplots.ipynb: script to generate the violin plots for the full time-series analysis. It loads the excel file obtained with "process_average_metrics_all_emulators.ipynb" and generates the plot. In this studye, to make it work correctly, the excel file sheets must have the following order: MLR-MSE, MLR-MADc, MLP-MSE, MLP-MADc, RNN-MSE, RNN-MADc, RNNh-MSE, RNNh-MADc, LSTM-MSE, LSTM-MADc, LSTMh-MSE, LSTMh-MADc.

- plot_violinplots_surgepeaks.ipynb: same as "plot_violinplots.ipynb" but for the excel file obtained for the surge peaks analysis.

- plot_BootstrapAveraged_v2.ipynb: it loads the bootstrap analysis results for a single emulator at both locations, averaged them, and plot the diagram.


 