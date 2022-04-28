#!/usr/bin/python

##########
#
# Parallel execution
# Minhyuk Sung
# Jan. 2016
#
##########

import functools
import getpass
import multiprocessing as mp
import subprocess
import time


def exec_cmd(name, cmd):
    #print(cmd)
    #print('Job [' + name + '] Started.')
    subprocess.call(cmd, shell=True)
    print('Job [' + name + '] Finished.')


def log_result(result):
    g_results.append(result)
    num_completed_jobs = len(g_results)
    num_remaining_jobs = g_num_total_jobs - num_completed_jobs
    elapsed_time = time.time() - g_starting_time

    msg = 'Batch: ' + g_batch_name + '. '
    msg = msg + 'Elapsed time: ' + \
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
    msg = msg + 'Waiting for ' + str(num_remaining_jobs) + \
            ' / ' + str(g_num_total_jobs) + ' jobs...'
    print(msg)


def run_parallel(batch_name, cmd_list):
    global g_results
    global g_starting_time
    global g_num_total_jobs
    global g_batch_name

    g_results = []
    g_starting_time = time.time()
    g_num_total_jobs = len(cmd_list)
    g_batch_name = batch_name
    num_processors = int(0.8 * mp.cpu_count())
    if g_num_total_jobs == 0: return;

    msg = 'Batch (' + g_batch_name + ') Starting...'
    print(msg)
    print(' - Job count = ' + str(g_num_total_jobs))
    print(' - Processor count = ' + str(num_processors))

    pool = mp.Pool(processes=num_processors)
    for args in cmd_list:
        pool.apply_async(func=exec_cmd, args=args, callback=log_result)
    pool.close()
    pool.join()

    msg = 'Batch (' + g_batch_name + ') Completed.'
    msg = msg + ' Current time: ' + \
            time.strftime("%H:%M:%S", time.localtime(time.time())) + '. '
    print(msg)

