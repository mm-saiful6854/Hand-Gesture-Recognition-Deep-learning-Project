# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:31:56 2019

@author: cse.repon
"""

import logging

import threading

import time


def thread_function(name,repon):

    logging.info("Thread %s: starting", name)

    time.sleep(2)

    logging.info("Thread %s: finishing", name)


if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"

    logging.basicConfig(format=format, level=logging.INFO,

                        datefmt="%H:%M:%S")


    logging.info("Main    : before creating thread")

    x = threading.Thread(target=thread_function, args=("repon","fafaf"))

    logging.info("Main    : before running thread")

    x.start()

    logging.info("Main    : wait for the thread to finish")

    # x.join()

    logging.info("Main    : all done")