#!/usr/bin/env python
'''
Copyright The Jackson Laboratory, 2022
author: J. Peterson

This scripts generates lists of random frames.  This program was used
for testing.

'''
import argparse
import random

def main():
    '''
    This simple script generates a random list of frames for exclusion from each slice of a scan.
    '''
    start_frame = 1
    end_frame = 46
    frame_count = 1
    slice_count = 17
    random_seed = -1
    output_file = ""
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("-s", "--start_frame", type=int, default=start_frame,
        help=f"index of first frame ({start_frame})")
    cl_parser.add_argument("-e", "--end_frame", type=int, default=end_frame,
        help=f"index of last frame({end_frame})")
    cl_parser.add_argument("-n", "--slice_count", type=int, default=slice_count,
        help=f"number of slices({slice_count})")
    cl_parser.add_argument("-f", "--frame_count", type=int, default=frame_count,
        help=f"number of random frames ({frame_count})")
    cl_parser.add_argument("-r", "--random_seed", type=int, default=random_seed,
        help=f"Seed for random number generator when non-negative ({random_seed})")
    cl_parser.add_argument("-o", "--output_file", default=output_file,
        help="output file (optional)")
    args = cl_parser.parse_args()
    start_frame = args.start_frame
    end_frame = args.end_frame
    slice_count = args.slice_count
    frame_count = args.frame_count
    random_seed = args.random_seed
    output_file = args.output_file

    print(f"\nstart_frame={start_frame}")
    print(f"end_frame={end_frame}")
    print(f"slice_count={slice_count}")
    print(f"frame_count={frame_count}")
    print(f"random_seed={random_seed}")
    print(f"output_file={output_file}\n")
    if frame_count <= 0:
        print("\nNumber of random frames must be at least 1.\n")
        return

    if random_seed >= 0:
        random.seed(random_seed)

    frame_list = list(range(start_frame, end_frame + 1))

    frames_str = "z-index, excluded_frames\n"
    for slice_idx in range(slice_count):
        frames_str += f"{slice_idx}, "
        new_frame_list = frame_list.copy()
        while len(new_frame_list) > frame_count:
            idx = random.randint(0, len(new_frame_list)-1)
            del new_frame_list[idx]
        csv_line = ""
        for i in new_frame_list:
            csv_line += f"{i:2}, "
            frames_str += f"{i}, "
        csv_line = csv_line[:-2]
        frames_str = frames_str[:-2]
        frames_str += "\n"
        print(f"Z{slice_idx:02}: {csv_line}")
    print("")

    if output_file:
        try:
            with open(output_file, 'w', encoding='latin-1') as file:
                file.write(frames_str)
            print(f"\nframes written to: {args.output_file}\n")
        except OSError as error:
            print(f"\nException - frames file ({args.output_file})could not be written: {error}\n")

if __name__== "__main__":
    main()
