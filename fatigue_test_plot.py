
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy.linalg
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import sys
from os.path import basename, splitext
import argparse
from contextlib import contextmanager, closing
import re
import datetime

def split_key(key):
    """Split a name into a prefix, separator, and numeric part"""
    prefix_and_separator = key.rstrip('0123456789') #chop off the numbers
    prefix = prefix_and_separator.rstrip('_- ') # chop off the underscore/dash/space
    return prefix, prefix_and_separator[len(prefix):], key[len(prefix_and_separator):]

def key_prefix(key):
    """return just the prefix part of a key (see ``split_key``)"""
    return split_key(key)[0]

def key_number(key):
    """return just the appended number part of a key (see ``split_key```)"""
    return split_key(key)[2]

def key_prefixes(group):
    """Determine the unique prefixes in an HDF5 group.

    I often put lots of items into a data group that have the same
    name, with a number afterwards.  This function figures out what
    the unique names are in a given dataset, and how many there are
    of each.  If there's an underscore or dash or space before the number, that's 
    removed too.
    """
    prefixes = {}
    for k in group.keys():
        prefix = key_prefix(k)
        if prefix in prefixes:
            prefixes[prefix] = prefixes[prefix] + 1 # keep track of how many we have
        else:
            prefixes[prefix] = 1
    return prefixes

def numeric_items(group, prefix):
    """Return items from a group starting with a prefix, in numeric order.

    Return all the items in ``group`` that start with ``prefix`` and end
    with a number, optionally separated by dash/underscore/space characters.
    """
    keys = [k for k in group.keys() if key_prefix(k) == prefix]
    keys.sort(key=lambda k: int(key_number(k)))
    return [group[k] for k in keys]

def timestamp_to_datetime(item):
    """retrieve the string-formatted timestamp from an item and convert to a datetime"""
    ts_string = item.attrs['timestamp']
    if not isinstance(ts_string, str):
        ts_string = ts_string.decode("utf-8")
    return datetime.datetime.fromisoformat(ts_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse data from the fatigue test experiment.")
    parser.add_argument("filename", default="fatigue_tests.hdf5", nargs="?", help="Filename of the data file you are plotting")
    parser.add_argument("group_path", default="<latest>", nargs="?", help="The path of the data group you want to plot (defaults to the most recent).")
    parser.add_argument("--output_file", help="Filename for the output HDF5 file, defaults to the input filename plus _<group_path>_summary.h5.")
    parser.add_argument("--no_summary", action="store_true", help="Disable generating the summary file (HDF5 file with images stripped out)")
    args = parser.parse_args()

    print ("Loading data from {}...".format(args.filename))
    df = h5py.File(args.filename, mode = "r")

    print("Data groups in this file:")
    for k, v in df.items():
        print("{}: ".format(k), end='')
        for prefix, count in key_prefixes(v).items():
            print("{} {}s, ".format(count, prefix), end="")
        print("")
    try:
        assert args.group_path != "<latest>"
        data_group = df[args.group_path] # should be fatigue_test%03d
    except:
        print("Picking latest group, no group specified")
        data_group = list(df.values())[-1]
    print("\nUsing group {}".format(data_group.name))

    print("In this group, we have:")
    print(key_prefixes(data_group))

    camera_datasets = numeric_items(data_group, "data_cam")
    stage_datasets = numeric_items(data_group, "data_stage")
    time_datasets = numeric_items(data_group, "data_time")

    N = len(camera_datasets)

    print("\nThe test was started at {} and finished at {}".format(
        camera_datasets[0].attrs['timestamp'], 
        camera_datasets[-1].attrs['timestamp']))

    first_move = camera_datasets[0]
    m = first_move.shape[0] # the number of data points in each move
    print("There are {} points in each dataset".format(m))
    
    print("Combining datasets...", end="")
    cam_coords = np.empty((N*m, 2))
    stage_coords = np.empty_like(cam_coords)
    times = np.empty(N*m)
    #start_datetime = timestamp_to_datetime(camera_datasets[0])
    
    for i, (data_cam, data_stage, data_time) in enumerate(
                zip(camera_datasets, stage_datasets, time_datasets)):
        data_cam = data_group['data_cam{:05d}'.format(i)]
        data_stage = data_group['data_stage{:05d}'.format(i)]
        x = np.arange(m, dtype=np.int) + i*m
        rr = slice(np.min(x), np.max(x)+1)
        cam_coords[rr, :] = data_cam[:,:2]
        stage_coords[rr, :] = data_stage[:,:2]
        times[rr] = data_time[:]
    times -= times[0] #start the times from zero
    print("done")

    cycle = m * 4 # at the moment, we go forward, stop, back, stop, so 
    # the cycle is actually 4 times as long as m.

    print("Plotting datasets...", end="")
    f, ax = plt.subplots(2,1, sharex=True)
    f2, ax2 = plt.subplots(1,1)
    f3, ax3 = plt.subplots(2,1, sharex=True)
    for j in range(2):
        for k in range(cycle):
            ax[j].plot(times[k::cycle], cam_coords[k::cycle,j])
        ax3[j].plot(times[:2*cycle], cam_coords[:2*cycle,j], "o-")
        ax[j].set_ylabel("{} position/pixels".format(["X","Y"][j]))
        ax3[j].set_ylabel("{} position/pixels".format(["X","Y"][j]))
    ax[1].set_xlabel("time/seconds")
    ax3[1].set_xlabel("time/seconds")
    f3.suptitle("First 2 cycles")
    f.suptitle("Full test")
    f2.suptitle("XY plot")
    for j in range(cycle):
        ax2.plot(cam_coords[j::cycle,0], cam_coords[j::cycle,1])
    ax2.set_aspect(1)
    print("done")
    
    # Save these plots as a PDF and PNG files
    if "_summary" in args.filename:
        plot_fname = splitext(args.filename)[0] + ".pdf"
    else:
        plot_fname = splitext(args.filename)[0] + "_" + basename(data_group.name)
    print("Saving plots as {}".format(plot_fname))
    with PdfPages(plot_fname + ".pdf") as pdf:
        pdf.savefig(f3)
        pdf.savefig(f)
        pdf.savefig(f2)
    for k, v in [("start", f3), ("full_timeseries", f), ("scatter", f2)]:
        v.savefig(plot_fname + "_" + k + ".png")
   
    # Generate a "summary" HDF5 file without the images embedded every 100 moves, and containing only one group.
    if not args.no_summary:
        def basename(f):
            return f.split('/')[-1]
        if args.output_file is None:
            output_fname = splitext(args.filename)[0] + "_" + basename(data_group.name) + "_summary.h5"
        else:
            output_fname = args.output_file
        print("Saving just this dataset, with no images, to {}".format(output_fname))
        with closing(h5py.File(output_fname, mode="w")) as outfile:
            g = outfile.create_group(data_group.name)
            # copy the group attributes
            for k, v in data_group.attrs.items():
                g.attrs[k] = v
            for name, dset in data_group.items():
                if name.startswith("data_"):
                    g[name] = np.array(dset)
                    for k, v in dset.attrs.items():
                        g[name].attrs[k] = v
            g['template_image'] = np.array(data_group['template_image'])
    
    plt.show()
