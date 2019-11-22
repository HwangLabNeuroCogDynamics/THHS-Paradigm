#!/usr/bin/env python
# -*- coding: utf-8 -*-


## April 25, 2019: changing to eliminate one branch for our lesion patient 
## March 06, 2019: adaption for EEG
## March 04, 2019: change cue (shape) time from 1 second to 500ms

"""
Authors: Marco Pipoly, Dillan Cellier, Kai Hwang.
University of Iowa, 
Iowa City, IA
Hwang Lab, Dpt. of Psychological and Brain Sciences
As of Feb. 15, 2019:

Office:225 SLP
Office Phone:319-467-0610
Fax Number:319-335-0191
Lab:104 SLP

Lab Contact:
Web - https://kaihwang.github.io/
Email - kai-hwang@uiowa.edu

This version of the Psychopy script is designed to mimic and model functions
written from the psychopy builder

However, this script has been heavily edited with python code non-specific to the builder.
Take this into account when reading through this as script design here does not mirror builder
script layout. 
"""

from __future__ import absolute_import, division, print_function
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock, info
#from psychopy.event import globalKeys
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
from psychopy.hardware.emulator import launchScan # Add and customize this www.psychopy.org/api/hardware/emulator.html
from psychopy.info import RunTimeInfo

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
from random import choice as randomchoice
import os  # handy system and path functions
import sys  # to get file system encoding
import glob # this pulls files in directories to create Dictionaries/Str
import pandas as pd #Facilitates data structure and analysis tools. prepend 'np.'
#import matplotlib.pyplot as plt #This will be relevant for later functions
import pyglet as pyg
import copy #Incorporated for randomization of stimuli
import csv #for export purposes and analysis
import serial

#-----------Notes on Script Initiation----------
#This script calls on the directory it is housed in 
#as well as the folder containing the correct outputs
#Ensure this is addressed appropriatley before running the script

#-----------Change Directory------
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))#.decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'Task_THHS'  # from the Builder filename that created this script
expInfo = {'block': '001', 'participant': '', 'Stay_R':'0.5','IDS_R':'1.0','refresh_rate':60} 
# A note on this: blocks are added each run of the script
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName) # Gui grabs the dictionary to create a pre-experiment info deposit
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

#----------------setup windows and display objects--------
##### Setup the display Window
win = visual.Window(
    size=(1280, 800), fullscr=True, screen=0,
    allowGUI=False, allowStencil=False, units='deg',
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

#------------Initialize Variables------
#### Timing and trial keeping variables
#Welcome_Time_On_Screen = 1  # in seconds.
#Instruction_Time = 5 #
real_Total_trials = 83 #97 # Adjust number as indicated for experiment
Total_trials=real_Total_trials-3
Stay_probability=float(expInfo['Stay_R'])
Switch_probability =1-float(expInfo['Stay_R'])
Extra_dimension_switch_probability = 1-float(expInfo['IDS_R'])
Intra_dimension_switch_probability = float(expInfo['IDS_R'])
if Extra_dimension_switch_probability + Intra_dimension_switch_probability > 1: # making sure the IDS/EDS ratio make sense
    print('IDS and EDS ratio add up to more than 1')
    core.quit()
if Switch_probability >= .9: # same w switch ratio
    print('Switch Prob is No Bueno (equal to or above 90%)')
    core.quit()
Cue_time = .5
Pic_time = 3
#ITIs = [1, 1.1, 1.2, 1.3, 1.4, 1.5] # need to jitter this for fMRI, no is just uniform random
Initial_wait_time = 3
Number_of_initial_stays = 3

#### Set up the order of switch versus stay trials,
# integer code for trial type
Intra_dimension_switch = 11  #IDS, intra dimensional/tree switch
Extra_dimension_switch = 33  #EDS, shifting out of decision tree
Stay_trial = 99

#### setup number of trials for each trial type (EDS, IDS, stay)
Total_switch_trials = round(Total_trials * Switch_probability)
Number_of_EDS = round(Total_switch_trials * Extra_dimension_switch_probability)
Number_of_IDS = round(Total_switch_trials * Intra_dimension_switch_probability)
Number_of_stay = round(Total_trials * Stay_probability)

print(Number_of_EDS)
print(Number_of_IDS)
print(Number_of_stay)
#core.quit()
# ###setup switch and stay trials order, represented by integer code above
Trial_order = np.concatenate(( np.repeat(Stay_trial, Number_of_stay), np.repeat(Intra_dimension_switch, Number_of_IDS), np.repeat(Extra_dimension_switch, Number_of_EDS)))
Trial_order_int = np.random.permutation(np.random.permutation(Trial_order)) #randomly permute the trial order
Trial_order_int = np.concatenate((np.repeat(Stay_trial, Number_of_initial_stays), Trial_order_int))  # always add 3 "Stay trials at the beginig"

#### now change the integers to string ("switch", "IDS", "EDS") so it is more readable
Trial_order = Trial_order_int.astype('object') 	#convert integer representation of trial order into text for readability
Trial_order[Trial_order == 99] = 'Stay'
Trial_order[Trial_order == 11] = 'IDS'
Trial_order[Trial_order == 33] = 'EDS'

# random sequence of face or scene picture presentation
Pic_order = np.random.permutation(np.random.randint(1,3,len(Trial_order)).astype('object'))
Pic_order[Pic_order == 1] = 'Face'
Pic_order[Pic_order == 2] = 'Scene'

# SHape Sizing
vis_deg_circ=8#15 # too big!
vis_deg_poly=13#20 #good for MRI?

# Add multiple shutdown keys "at once".
for key in ['q', 'escape']:
    event.globalKeys.add(key, func=core.quit)


#### Dependent on what variation of the experiment is running

ITIs =[3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6] # averages to around 4 seconds?
EEGflag=1
yes_key='1'
no_key='0'
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
thisDir_save=_thisDir
thisDir_save=thisDir_save.split('/')
thisDir_save='/'.join(thisDir_save[:-1])
filename = thisDir_save + u'/ThalHi_data/eeg_data/behavioral_data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['block'],expName, expInfo['date'])
#Above to be saved in Thalamege *change later January 21 2019
vis_deg_circ=10
vis_deg_poly=15
refresh_rate=expInfo['refresh_rate']

#EEGflag=0
if EEGflag:
    port=serial.Serial('COM4',baudrate=115200)
    port.close()
    startSaveflag=bytes([254])
    stopSaveflag=bytes([255])
    
    #fill=['Donut','Filled']
    #shape=['Polygon','Circle']
    #color=['red','blue']
    
    #cue_n_bytes=99
    #cuetrigDict={}
    #for f in fill:
      #  for s in shape:
       #     for c in color:
          #      cue_n_bytes=cue_n_bytes+2
           #     cuetrigDict[f+'_'+s+'_'+c+'_trig']=cue_n_bytes
    cuetrigDict={'IDS_trig':101,'EDS_trig':103,'Stay_trig':105}
    face_probe_trig=151
    scene_probe_trig=153
    subNonResp_trig=155
    subResp_trig=157
    ITI_trig=159
    endofBlock_trig=161
    


endExpNow = False  # flag for 'escape' or other condition => quit the exp

#This will be for CSV initiation
def makeCSV(filename, thistrialDict, trial_num):
    with open(filename + '.csv', mode='w') as our_data:
         ExpHead=thistrialDict[trial_num].keys()
         writer=csv.DictWriter(our_data,fieldnames=ExpHead)
         writer.writeheader()
         for n in range(trial_num+1):
            writer.writerow(thistrialDict[n])

def wait_here(t,to_draw):
    interval=1/refresh_rate
    num_frames=int(t/interval)
    for n in range(num_frames): #change back to num_frames
        to_draw.draw()
        win.flip()

#Response Keys dictionary
corrAns = {"yes":yes_key, "no":no_key} #This is not in use for this script version



# Make sure to adjust window and monitor parameters per computer used
#not doing this will affect stimulus presentation among other things

win.recordFrameIntervals=True
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so default 60 Hz


#### Welcome screen
# Initialize components for Routine "Welcome"
#WelcomeClock = core.Clock()
Welc = visual.TextStim(win=win, name='Welc',
    text=u'Welcome!', units='norm',
    font=u'Arial',
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0);


##### Instructions screen
# Initialize components for Routine "Instructions"
#InstructionsClock = core.Clock()
Directions = visual.TextStim(win=win, name='Directions',
    text=u'You are now about to begin the task. \n\nGet Ready \n\nPress Any Key to Continue',
    font=u'Arial', alignVert='center', units='norm',
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0);


##### Central fixations
# Initialize components for Routine "Fixation"
#FixationClock = core.Clock()
Fix_Cue = visual.TextStim(win=win, name='Fix_Cue',
    text=u'+', units='norm',
    font=u'Arial',
    pos=(0, 0), height=0.3, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0);

#### For Plug in below
Wait_for_Scanner = visual.TextStim(win=win, name='Wait_for_Scanner',
    text=u'Waiting for MRI to initiate task',
    alignVert='center', units='norm',
    pos=(0,0), height=0.09, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=0.0);
##### Load Face and Scene pictures into stim objects, organized into a dict
#Dictionaries and the corresponding file paths
direc = os.getcwd()+'/localizer_stim/' #_thisDir #'/Users/mpipoly/Desktop/Psychopy/localizer_stim/' #always setup path on the fly in case you switch computers
ext = 'scenes/*.jpg' #file delimiter
faces_ext = 'faces/*.jpg'
faces_list = glob.glob(direc + faces_ext)
scenes_list = glob.glob(direc + ext)
#print(faces_list)
#print(scenes_list)
Img_Scene = {}
Img_Faces = {}
Img_path = {}

# randomly select pics from list, only load same number of pics as number of trials to save memory
for i,f in enumerate(np.random.randint(low=0,high=len(faces_list),size=len(Trial_order))):

    if Pic_order[i] == 'Face':
        Img_Faces[i] = visual.ImageStim(win=win, image=faces_list[f])
        Img_path[i] = faces_list[f]
    if Pic_order[i] =='Scene':
        Img_Scene[i] = visual.ImageStim(win=win, image=scenes_list[f])
        Img_path[i] = scenes_list[f]


##### Setup Cue stim objects
#Overlapping vertices for below
#red = [1.0,-1,-1]
#blue = [0, 0, 255]
#color = 'green'
#scale = 0.7
#donutVert = [[(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)],[(-.15,-.15),(-.15,.15),(.15,.15),(.15,-.15)]]
#simpleVert= [(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)]
size=(.75,.85)
#filled circle 
circle_filled_r = visual.ImageStim(
    win=win, image=os.getcwd()+'/cues/fill_r_circle.png',name='circle_filled_r', units='norm', 
    pos=(0,0), size=size) #, size=[vis_deg_circ/1.75,vis_deg_circ/1.75

circle_filled_b = visual.ImageStim(
    win=win, image=os.getcwd()+'/cues/fill_b_circle.png',name='circle_filled_b', units='norm', 
    pos=(0,0), size=size) #, size=[vis_deg_circ/1.75,vis_deg_circ/1.75]

#donut circle
circle_donut_r = visual.ImageStim(
    win=win,  image=os.getcwd()+'/cues/donut_r_circle.png',name='circle_donut_r', units='norm', pos=(0,0), 
    interpolate=True,size=size) #size=[vis_deg_circ/1.75,vis_deg_circ/1.75]

circle_donut_b = visual.ImageStim(
    win=win,  image=os.getcwd()+'/cues/donut_b_circle.png',name='circle_donut_b', units='norm', pos=(0,0), 
    interpolate=True,size=size ) 
    #size=[vis_deg_circ/1.75,vis_deg_circ/1.75]

#filled square
polygon_filled_r = visual.ImageStim(
    win=win, image=os.getcwd()+'/cues/fill_r_square.png',name='polygon_filled_r', units='norm', pos=(0,0),size=size) #size=[vis_deg_poly,vis_deg_poly],

polygon_filled_b = visual.ImageStim(
    win=win, image=os.getcwd()+'/cues/fill_b_square.png',name='polygon_filled_b', units='norm', pos=(0,0),size=size) #size=[vis_deg_poly,vis_deg_poly],

#donut square
polygon_donut_b = visual.ImageStim(
    win=win,image=os.getcwd()+'/cues/donut_b_square.png', name='polygon_donut_b', units='norm',  pos=(0,0), interpolate=True,size=size) #size=[vis_deg_poly,vis_deg_poly],

polygon_donut_r = visual.ImageStim(
    win=win,image=os.getcwd()+'/cues/donut_r_square.png', name='polygon_donut_r', units='norm',  pos=(0,0), interpolate=True,size=size) #size=[vis_deg_poly,vis_deg_poly],

#----------setup trial ordrs (switch and stay trials)---------

#### hiearchical cue tree
#
#			  filled                      donut         # texture dermine 2nd feature of interest
#			/	     \	                /       \
#	   polygon       circle           red       blue
#      /      \      /     \         /   \     /    \
#     red    blue   red    blue     po   ci   po    ci  # po = polygon, ci = circle
#      |       |     |       |      |    |    |     |
#     fpr     fpb   fcr     fcb    dpr  dcr  dpb   dcb
#      |       |     |       |      |    |    |     |
#     face    face  scene   scene   f    f    s     s   # f = face task, s = scene task
#
#
# Total 8 cues, a short hand will be a three letter code
# First letter: texture, f= filled, d=donut, if cue is filled, focus on shape, if donut, focus on color
# Second letter: shape of cue, p = polygon, c = circle. If texture is filled and shape is pologon, do the face task. If texture is filled and
# shape is circle, do the scene task.
# Third letter: color of cue, r = red, b = blue. If texture is donut and color is red, do the face task. If texture is dont and color is blue, do the
# scene task
#
# for ease of tracking, we will give each individual cue an integer code
#
# 1: fpr  = filled polygon with red    ~ is pic a face?
# 2: fpb  = filled polygon with blue   ~ is pic a face?
# 3: fcr  = filled circle with red     ~ is pic a scene?
# 4: fcb  = filled circle with blue    ~ is pic a scene?
# 5: dpr  = donut polygon with red    ~ is pic a face ?
# 6: dcr  = donut circile with red    ~ is pic a face?
# 7: dpb  = donut polygon with blue   ~ is pic a scene?
# 8: dcb  = donut circle with blue    ~ is pic a scene?

#["Empty"]*len(Trial_order)
#polygon_filled=[]
#circle_filled=[]
#polygon_donut=[]
#circle_donut=[]

##### create a attribute dictionary saving all cue types
# note, copying the shapestim and visual.circle object using copy.copy() turns out to be critical, otherwise can't change colors on the fly
Cue_types = {'fpr': {'cue':'fpr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled_r) },
             'fpb': {'cue':'fpb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled_b) },
             'fcr': {'cue':'fcr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled_r) },
             'fcb': {'cue':'fcb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled_b) }}
             #'dpr': {'cue':'dpr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_donut_r) },
             #'dcr': {'cue':'dcr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Face', 'cue_stim': copy.copy(circle_donut_r) },
             #'dpb': {'cue':'dpb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Scene', 'cue_stim': copy.copy(polygon_donut_b) },
             #'dcb': {'cue':'dcb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_donut_b) }}

#select a random first trial

random_first = randomchoice(list(Cue_types))

#### Now create a massive trial dictionary. In this dictionary will save all the attributes for each trial, by its order.
#### The "key" to this dict will be the trial number, in each key will be anohter nested dictionary saving all the
#### attributes of each trial, the cue type, color, shape, texture, the cue stimuli , and the picture stim.
Trial_dict = {}
for i in range(len(Trial_order)):

    if i == 0 :   #set first trial, randomly select
        Trial_dict[i] = Cue_types[random_first]
    else:
        if Trial_order[i] == 'Stay':  # stay
            if Trial_dict[i-1]['cue'] in ['fpr', 'fpb']:
                Trial_dict[i] = Cue_types[randomchoice(['fpr', 'fpb'])].copy() # having .copy() is extremenly important, otherwise it will be
            elif Trial_dict[i-1]['cue'] in ['fcr', 'fcb']:# a link to original object and not "switch"
                Trial_dict[i] = Cue_types[randomchoice(['fcr', 'fcb'])].copy()
            elif Trial_dict[i-1]['cue'] in ['dpr', 'dcr']:
                Trial_dict[i] = Cue_types[randomchoice(['dpr', 'dcr'])].copy()
            elif Trial_dict[i-1]['cue'] in ['dpb', 'dcb']:
                Trial_dict[i] = Cue_types[randomchoice(['dpb', 'dcb'])].copy()
            else:
                print('something wrong with stay trial sequence')
                core.quit()

        elif Trial_order[i] == 'IDS':  #intra shift
            if Trial_dict[i-1]['cue'] in ['fpr', 'fpb']:
                Trial_dict[i] = Cue_types[randomchoice(['fcr', 'fcb'])].copy()
            elif Trial_dict[i-1]['cue'] in ['fcr', 'fcb']:
                Trial_dict[i] = Cue_types[randomchoice(['fpr', 'fpb'])].copy()
            elif Trial_dict[i-1]['cue'] in ['dpr', 'dcr']:
                Trial_dict[i] = Cue_types[randomchoice(['dpb', 'dcb'])].copy()
            elif Trial_dict[i-1]['cue'] in ['dpb', 'dcb']:
                Trial_dict[i] = Cue_types[randomchoice(['dpr', 'dcr'])].copy()
            else:
                print('something wrong with intra dimensional shift trial sequence')
                core.quit()

        elif Trial_order[i] == 'EDS': # out of tree shift
            if Trial_dict[i-1]['cue'] in ['fpr', 'fpb', 'fcr', 'fcb']:
                Trial_dict[i] = Cue_types[randomchoice(['dpr', 'dcr', 'dpb', 'dcb'])].copy()
            elif Trial_dict[i-1]['cue'] in ['dpr', 'dcr', 'dpb', 'dcb']:
                Trial_dict[i] = Cue_types[randomchoice(['fpr', 'fpb', 'fcr', 'fcb'])].copy()
            else:
                print('something wrong with extra dimensional shift trial sequence')
                core.quit()
        else:
            print('something wrong with trial type sequence generation')
            core.quit()

    # record the trial type for each trial
    Trial_dict[i]['Trial_type'] = Trial_order[i]

#    #set colors of the cue stim # donot circle need to set line color but not fill
#    if Trial_dict[i]['cue'] == 'dcr':
#        Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
#        #Trial_dict[i]['cue_stim'].lineWidth=200 
#    elif Trial_dict[i]['cue'] == 'dcb':
#        Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
#    else: # every othre cue set both line and fill color
#        Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
#        Trial_dict[i]['cue_stim'].setFillColor(Trial_dict[i]['Color'])

    # random place scene or face picture
    Trial_dict[i]['pic'] = Pic_order[i]
    if Trial_dict[i]['pic'] == 'Face':
        Trial_dict[i]['pic_stim'] = Img_Faces[i]
    if Trial_dict[i]['pic'] == 'Scene':
        Trial_dict[i]['pic_stim'] = Img_Scene[i]
    Trial_dict[i]['img_path'] = Img_path[i]


#### Old CLocks

Welc.draw()
win.flip()
event.waitKeys(maxWait=3)

Directions.draw()
win.flip()
event.waitKeys()


##### TTL Pulse trigger
if EEGflag:
    port.open()
    #win.callonFlip(pport.setData,delay1trig)
    port.write(startSaveflag)
    #port.close()

#### Setting up a global clock to track initiation of experiment to end
#Time_Since_Run = core.MonotonicClock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
RT_clock=core.Clock()
##### 2 seconds Intial fixation
Fix_Cue.draw()
win.flip()
core.wait(Initial_wait_time)

##### Start trials

for trial_num in range(len(Trial_dict)): #range(len(Trial_dict))
    
    if Trial_dict[trial_num]['cue']=='dcr' or Trial_dict[trial_num]['cue']=='dpr':
        if Trial_dict[trial_num]['pic']=='Face':
            corr_resp=yes_key
        else:
            corr_resp=no_key
    if Trial_dict[trial_num]['cue']=='fpr' or Trial_dict[trial_num]['cue']=='fpb':
        if Trial_dict[trial_num]['pic']=='Face':
            corr_resp=yes_key
        else:
            corr_resp=no_key
    if Trial_dict[trial_num]['cue']=='fcr' or Trial_dict[trial_num]['cue']=='fcb':
        if Trial_dict[trial_num]['pic']=='Scene':
            corr_resp=yes_key
        else:
            corr_resp=no_key
    if Trial_dict[trial_num]['cue']=='dpb' or Trial_dict[trial_num]['cue']=='dcb':
        if Trial_dict[trial_num]['pic']=='Scene':
            corr_resp=yes_key
        else:
            corr_resp=no_key
    
    if EEGflag:
        #cue_trig_name=Trial_dict[trial_num]['Texture']+'_'+Trial_dict[trial_num]['Shape']+'_'+Trial_dict[trial_num]['Color']+'_trig'
        cue_trig_name=Trial_dict[trial_num]['Trial_type']+'_trig'
        cue_trig=cuetrigDict[cue_trig_name]
    
    if EEGflag:
        win.callOnFlip(port.write,bytes([cue_trig]))
    
    # draw the cue
    wait_here(Cue_time,to_draw=Trial_dict[trial_num]['cue_stim'])
    #core.wait(Cue_time)
    
    # draw the face or scene picture
    if EEGflag:
        if Trial_dict[trial_num]['pic']=='Scene':
            stim_trig=scene_probe_trig
        else:
            stim_trig=face_probe_trig
    
    if EEGflag:
        win.callOnFlip(port.write,bytes([stim_trig]))
    
    #Trial_dict[trial_num]['pic_stim'].draw()
    
    Trial_dict[trial_num]['pic_stim'].autoDraw=True
    event.clearEvents()
    max_win_count=int(2/(1/refresh_rate))
    win_count=0
    subRespo=None
    RT_clock.reset()
    while not subRespo:
        win.flip()
        subRespo=event.getKeys(timeStamped=RT_clock, keyList=[yes_key,no_key])
        win_count=win_count+1
        if win_count==max_win_count:
            break
    
    Trial_dict[trial_num]['pic_stim'].autoDraw=False
    #subRespo = event.waitKeys(maxWait=2.5, timeStamped=RT_clock, keyList=['0','1'])
    #print(subRespo)
    #print(subRespo[0])
    #print(subRespo[0][0])
    #print(subRespo[0][1])
    
    print(subRespo)
    if not subRespo:
        if EEGflag:
            port.write(bytes([subNonResp_trig]))
            respTrigname='NoResp'
            resptrig=subNonResp_trig
        trial_Corr=-1
        rt='none'
        subKEY='none'
    elif subRespo[0][0]==corr_resp:
        if EEGflag:
            port.write(bytes([subResp_trig]))
            respTrigname='resp'
            resptrig=subResp_trig
        trial_Corr=1
        rt=subRespo[0][1]
        subKEY=subRespo[0][0]
    elif subRespo[0][0]!=corr_resp:
        if EEGflag:
            port.write(bytes([subResp_trig]))
            respTrigname='resp'
            resptrig=subResp_trig
        trial_Corr=0
        rt=subRespo[0][1]
        subKEY=subRespo[0][0]
    else:
        print('Something Wrong with subRespo.waitKeys')
        core.quit()
    
    Trial_dict[trial_num]['trial_Corr']=trial_Corr
    Trial_dict[trial_num]['rt']=rt
    Trial_dict[trial_num]['What_Is_CorrResp']=corr_resp
    Trial_dict[trial_num]['Subject_Respo']=subKEY
    Trial_dict[trial_num]['trial_n']=trial_num
    Trial_dict[trial_num]['block']=expInfo['block']
    Trial_dict[trial_num]['sub']=expInfo['participant']
    if EEGflag:
        Trial_dict[trial_num]['trigs']={cue_trig_name:cue_trig,Trial_dict[trial_num]['pic']:stim_trig,respTrigname:resptrig}
    #core.wait(Pic_time)
    #print(Trial_dict[trial_num]['img_path'])
    if EEGflag:
        win.callOnFlip(port.write,bytes([ITI_trig]))

    # draw the ITI fixation
    #Fix_Cue.draw()
    wait_here(randomchoice(ITIs),to_draw=Fix_Cue)
     # randomly pick one from the range of ITIs
    makeCSV(filename=filename,thistrialDict=Trial_dict,trial_num=trial_num)

