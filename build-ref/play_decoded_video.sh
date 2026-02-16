#!/bin/bash

mplayer -demuxer rawvideo -rawvideo w=352:h=288 ../build-ref/output.yuv
