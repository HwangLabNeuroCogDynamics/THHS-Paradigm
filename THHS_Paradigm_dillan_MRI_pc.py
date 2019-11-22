#!/usr/bin/env python
# -*- coding: utf-8 -*-

## march 04, 2019: change cue (shape) time from 1 second to 500ms
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
from psychopy import  gui, visual, core, data, event,clock
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
expInfo = {'block': '001', 'participant': '', 'sex': '', 'MRI/Behavior? (M/B)':'', 'Stay_R':'0.5','IDS_R':'0.5'} 
# A note on this: blocks are added each run of the script
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName) # Gui grabs the dictionary to create a pre-experiment info deposit
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

#----------------setup windows and display objects--------
##### Setup the display Window
win = visual.Window(
    size=(1440,900), fullscr=True, screen=0,
    allowGUI=False, allowStencil=False, units='deg',
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

#### Dependent on what variation of the experiment is running
if expInfo['MRI/Behavior? (M/B)']=='M':
    ITIpath='H:/Generate_ITIs/ThalHiITIs/'
    ITI_rand_file=np.random.choice(os.listdir(ITIpath))
    ITI_rand_file=open(ITIpath+ITI_rand_file,'r').readlines()
    ITI_list=[]
    
    for i in ITI_rand_file:
        t=i.split('\n')
        t=t[0]
        ITI_list.append(float(t))
    
    def make_ITI(trial_n):
        #ITI=np.random.choice([1,2,3,4,5,6,7,8,9,10],1,p=[(.7/9),(.7/9),(.7/9),.3,(.7/9),(.7/9),(.7/9),(.7/9),(.7/9),(.7/9)])[0] # averages to around 4 seconds?
        ITI=ITI_list[trial_n]
        return ITI
    real_Total_trials = 48+3 #  number of first stay trials 
    MRIflag=1
    yes_key='1'
    no_key='2'
    thisDir_save=_thisDir
    thisDir_save=thisDir_save.split('/')
    thisDir_save='/'.join(thisDir_save[:-1])
    filename = thisDir_save + u'/ThalHi_data/MRI_data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['block'],expName, expInfo['date'])
        # SHape Sizing
    vis_deg_circ=15 # too big! Original is 15
    vis_deg_poly=20 #good for MRI?
    Cue_time = .5
    Pic_time = 2.5
    refresh_rate=60

elif expInfo['MRI/Behavior? (M/B)']=='B':
    def make_ITI(trial_n):
        ITI=np.random.choice([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        return ITI
    
    real_Total_trials = 83 #97 # Adjust number as indicated for experiment
    MRIflag=0
    yes_key='1'
    no_key='0'
    # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    thisDir_save=_thisDir
    thisDir_save=thisDir_save.split('/')
    thisDir_save='/'.join(thisDir_save[:-1])
    filename = thisDir_save + u'/ThalHi_data/behav_data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['block'],expName, expInfo['date'])
    #Above to be saved in Thalamege *change later January 21 2019
    vis_deg_circ=10
    vis_deg_poly=15
    Cue_time = .5
    Pic_time = 2.5
    refresh_rate=60
else:
    print('no ITIs provided')
    core.quit()


#------------Initialize Variables------
#### Timing and trial keeping variables
#Welcome_Time_On_Screen = 1  # in seconds.
#Instruction_Time = 5 #

#ITIs = [1, 1.1, 1.2, 1.3, 1.4, 1.5] # need to jitter this for fMRI, no is just uniform random
Initial_wait_time = 3
Number_of_initial_stays = 3

Total_trials=real_Total_trials-Number_of_initial_stays
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


#### Set up the order of switch versus stay trials,
# integer code for trial type
Intra_dimension_switch = 11  #IDS, intra dimensional/tree switch
Extra_dimension_switch = 33  #EDS, shifting out of decision tree
Stay_trial = 99

#### setup number of trials for each trial type (EDS, IDS, stay)
Total_switch_trials = round((Total_trials) * Switch_probability)
Number_of_EDS = round(Total_switch_trials * Extra_dimension_switch_probability)
Number_of_IDS = round(Total_switch_trials * Intra_dimension_switch_probability)
Number_of_stay = round((Total_trials) * Stay_probability)

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


# Not using experiment handler anymore
# An ExperimentHandler isn't essential but helps with data saving
# thisExp = data.ExperimentHandler(name=expName, version='',
#     extraInfo=expInfo, runtimeInfo=None,
#     originPath='/Users/mpipoly/Desktop/Psychopy/Task_Kai.py',
#     savePickle=False, saveWideText=True,
#     dataFileName=filename)

# save a log file for detail verbose info
#logFile = logging.LogFile(filename+'.log', level=logging.EXP)
#logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

#
endExpNow = False  # flag for 'escape' or other condition => quit the exp

#This will be for CSV initiation
def makeCSV(filename, thistrialDict, trial_num):
    with open(filename + '.csv', mode='w') as our_data:
         ExpHead=thistrialDict[trial_num].keys()
         writer=csv.DictWriter(our_data,fieldnames=ExpHead)
         writer.writeheader()
         for n in range(trial_num+1):
            writer.writerow(thistrialDict[n])
#Response Keys dictionary
#corrAns = {"yes":1, "no":0} #This is not in use for this script version


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
    languageStyle='LTR',
    depth=0.0);


##### Instructions screen
# Initialize components for Routine "Instructions"
#InstructionsClock = core.Clock()
Directions = visual.TextStim(win=win, name='Directions',
    text=u'You are now about to begin the task. \n\nGet Ready \n\nPress Any Key to Continue',
    font=u'Arial', alignVert='center', units='norm',
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=0.0);


##### Central fixations
# Initialize components for Routine "Fixation"
#FixationClock = core.Clock()
Fix_Cue = visual.TextStim(win=win, name='Fix_Cue',
    text=u'+', units='norm',
    font=u'Arial',
    pos=(0, 0), height=0.3, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=0.0);

#### For Plug in below
Wait_for_Scanner = visual.TextStim(win=win, name='Wait_for_Scanner',
    text=u'Waiting for MRI to initiate task',
    alignVert='center', units='norm',
    pos=(0,0), height=0.09, wrapWidth=None, ori=0,
    color=u'white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
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
             'fcb': {'cue':'fcb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled_b) },
             'dpr': {'cue':'dpr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_donut_r) },
             'dcr': {'cue':'dcr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Face', 'cue_stim': copy.copy(circle_donut_r) },
             'dpb': {'cue':'dpb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Scene', 'cue_stim': copy.copy(polygon_donut_b) },
             'dcb': {'cue':'dcb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_donut_b) }}

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
#globalClock = core.Clock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
#routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine
#
# ------Prepare to start Routine "Welcome"-------
#t = 0
#WelcomeClock.reset()  # clock
#frameN = -1
#continueRoutine = True
#routineTimer.add(Welcome_Time_On_Screen)
# update component parameters for each repeat
# keep track of which components have finished
#WelcomeComponents = [Welc]
#
#
# no idea what this is...??
#for thisComponent in WelcomeComponents:
#    if hasattr(thisComponent, 'status'):
#        thisComponent.status = NOT_STARTED
#
# -------Start Routine "Welcome"-------
#while continueRoutine and routineTimer.getTime() > 0:
#    # get current time
#    t = WelcomeClock.getTime()
#    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
#    # update/draw components on each frame
#
#    # *Welc* updates
#    if t >= 0.0 and Welc.status == NOT_STARTED:
#        # keep track of start time/frame for later
#        Welc.tStart = t
#        Welc.frameNStart = frameN  # exact frame index
Welc.draw()
win.flip()
event.waitKeys(maxWait=3)
#    frameRemains = 0.0 + 1.0- win.monitorFramePeriod * 0.75  # most of one frame period left
#    if Welc.status == STARTED and t >= frameRemains:
#        Welc.setAutoDraw(False)
#
#    # check if all components have finished
#    if not continueRoutine:  # a component has requested a forced-end of Routine
#        break
#    continueRoutine = False  # will revert to True if at least one component still running
#    for thisComponent in WelcomeComponents:
#        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
#            continueRoutine = True
#            break  # at least one component has not yet finished
#
#    # check for quit (the Esc key)
#    if endExpNow or event.getKeys(keyList=["escape"]):
#        core.quit()
#
#    # refresh the screen
#    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
#        win.flip()
#
# -------Ending Routine "Welcome"-------
#for thisComponent in WelcomeComponents:
#    if hasattr(thisComponent, "setAutoDraw"):
#        thisComponent.setAutoDraw(False)
#
# ------Prepare to start Routine "Instructions"-------
#t = 0
#InstructionsClock.reset()  # clock
#frameN = -1
#continueRoutine = True
#routineTimer.add(Instruction_Time)
# update component parameters for each repeat
# keep track of which components have finished
#InstructionsComponents = [Directions]
#for thisComponent in InstructionsComponents:
#    if hasattr(thisComponent, 'status'):
#        thisComponent.status = NOT_STARTED
#
# -------Start Routine "Instructions"-------
#while continueRoutine and routineTimer.getTime() > 0:
#    # get current time
#    t = InstructionsClock.getTime()
#    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
#    # update/draw components on each frame
#
#    # *Directions* updates
#    if t >= 3.0 and Directions.status == NOT_STARTED:
#        # keep track of start time/frame for later
#        Directions.tStart = t
#        Directions.frameNStart = frameN  # exact frame index
Directions.draw()
win.flip()
event.waitKeys()
#    frameRemains = 3.0 + 5.0- win.monitorFramePeriod * 0.75  # most of one frame period left
#    if Directions.status == STARTED and t >= frameRemains:
#        Directions.setAutoDraw(False)
#
#    # check if all components have finished
#    if not continueRoutine:  # a component has requested a forced-end of Routine
#        break
#    continueRoutine = False  # will revert to True if at least one component still running
#    for thisComponent in InstructionsComponents:
#        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
#            continueRoutine = True
#            break  # at least one component has not yet finished
#
#    # check for quit (the Esc key)
#    if endExpNow or event.getKeys(keyList=["escape"]):
#        core.quit()
#
#    # refresh the screen
#    #if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
#
#
# -------Ending Routine "Instructions"-------
#for thisComponent in InstructionsComponents:
#    if hasattr(thisComponent, "setAutoDraw"):
#        thisComponent.setAutoDraw(False)


# -------Ending Routine "Instructions"-------


#------------ Start Trial presentation sequence

##### TTL Pulse trigger
if MRIflag:
    Wait_for_Scanner.draw()
    win.flip()
    event.waitKeys(keyList=['lshift','z'])
    core.wait(6)

##### Triggers for serial port ***

#### Setting up a global clock to track initiation of experiment to end
Time_Since_Run = core.MonotonicClock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
RT_clock=core.Clock()
##### 2 seconds Intial fixation
Fix_Cue.draw()
win.flip()
core.wait(Initial_wait_time)

##### Start trials

for trial_num in range(len(Trial_dict)): #range(len(Trial_dict))
    #print('\n')
    print(trial_num)
    # draw the cue
    Trial_dict[trial_num]['cue_stim'].draw()
    Cue_Prez_T=Time_Since_Run.getTime()
    win.flip()
    core.wait(Cue_time)

    # this is for debug
    #print(Trial_dict[trial_num]['Trial_type'] )
    #print(Trial_dict[trial_num]['cue'])
    #print(Trial_dict[trial_num]['Color'])

    # draw the face or scene picture
    event.clearEvents()
    Trial_dict[trial_num]['pic_stim'].autoDraw=True
    Photo_Prez=Time_Since_Run.getTime()
    
    subResps=[]
    max_win=int(Pic_time/(1/refresh_rate))
    win_count=0
    RT_clock.reset()
    while win_count !=max_win:
        win.flip()
        win_count+=1
        subRespo=event.getKeys(timeStamped=RT_clock, keyList=[no_key,yes_key])
        if subRespo:
            subResps.append(subRespo)
    #core.wait(Pic_time)
    #subRespo = event.getKeys(timeStamped=RT_clock, keyList=['0','1'])
    Trial_dict[trial_num]['pic_stim'].autoDraw=False
    subRespo_T=Time_Since_Run.getTime()
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

    print(subResps)
    #print(subResps[0])
    #print(subRespo[0][0])
    #print(subRespo[0][1])
    
    if not subResps:
        trial_Corr=-1
        rt='none'
        subKEY='none'
    else:
        subRespo=subResps[0]
        if subRespo[0][0]==corr_resp:
            trial_Corr=1
            rt=subRespo[0][1]
            subKEY=subRespo[0][0]
        elif subRespo[0][0]!=corr_resp:
            trial_Corr=0
            rt=subRespo[0][1]
            subKEY=subRespo[0][0]
        else:
            print('Something Wrong with subRespo.waitKeys')
            core.quit()
    
    #print(subKEY)
    Trial_dict[trial_num]['Time_Since_Run_subRespo']=subRespo_T
    Trial_dict[trial_num]['Time_Since_Run_Photo_Prez']=Photo_Prez
    Trial_dict[trial_num]['Time_Since_Run_Cue_Prez']=Cue_Prez_T
    Trial_dict[trial_num]['trial_Corr']=trial_Corr
    Trial_dict[trial_num]['rt']=rt
    Trial_dict[trial_num]['What_Is_CorrResp']=corr_resp
    Trial_dict[trial_num]['Subject_Respo']=subKEY
    Trial_dict[trial_num]['trial_n']=trial_num
    Trial_dict[trial_num]['block']=expInfo['block']
    Trial_dict[trial_num]['sub']=expInfo['participant']
    Trial_dict[trial_num]['Stay_R']=Stay_probability
    Trial_dict[trial_num]['IDS_R']=Intra_dimension_switch_probability
    #core.wait(Pic_time)
    #print(Trial_dict[trial_num]['img_path'])

    # draw the ITI fixation
    Fix_Cue.draw()
    if trial_num<Number_of_initial_stays: 
        ITI=1.5
    else:
        ITI=make_ITI(trial_num-Number_of_initial_stays)
    #print(ITI)
    win.flip()
    core.wait(ITI) # randomly pick one from the range of ITIs
    makeCSV(filename=filename,thistrialDict=Trial_dict,trial_num=trial_num)

##### Add a finish screen so subjects know they are done with a task block

###### Not using trial handler
# # set up handler to look after randomisation of conditions etc
# trials = data.TrialHandler(nReps=2, method='random',
#     extraInfo=expInfo, originPath=-1, #if needed, add dataTypes=['rt']
#     trialList=(stimList, 0:30), # add 'Kai_testing.xlsx' for excel method
#     seed=None, name='trials')
# thisExp.addLoop(trials)  # add the loop to the experiment
# thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
# if thisTrial != None:
#     for paramName in thisTrial:
#         exec('{} = thisTrial[paramName]'.format(paramName))

# t=0
# CueClock.reset() #reset
# frameN = -1
# continueRoutine = True


# for thisTrial in trialHandler:
#     Fix_Cue.setAutoDraw = True
#     win.flip()
#     core.wait(secs=0.5, hogCPUperiod=0.1)
#     Fix_Cue.setAutoDraw=False
#     win.flip()
#     core.wait(secs=0.5, hogCPUperiod=0.1)

#     if thisTrial['stim']=='face' and Fix_Cue.autoDraw == False:
#         thisFace=np.random.choice(Img_Faces.keys(),1)
#         thisFace=Img_Faces[thisFace]
#         thisStim=thisFace
#         thisReactionTime=..... + float(thisTrial['face'])/2.0

#     elif thisTrial['stim']=='place':
#         thisPlace=np.random.choice(Img_Scene.keys(),1)
#         thisPlace=Img_Scene[thisPlace]
#         thisStim=thisPlace

#         thisReactionTime=..... + float(thisTrial['place'])/2.0

#     if thisTrial['color']=='red':
#         thisStim.setColor==red
#     elif thisTrial['color']=='blue'
#         thisStim.setColor==blue

#     if thisTrial['stim']==''


# if hasattr(thisStim, 'status'):
#     thisStim.status = NOT_STARTED
