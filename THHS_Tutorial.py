#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


#-----------Change Directory------
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))#.decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

#Below the tutorial provides a gui to flag for screen adjustments depending on location.
#Example: running the tutorial in lab space for behavioral only will require different
#monitor setings than in the mock scanner bay
dlg=gui.Dlg(title="THHS Tutorial")
#dlg.addText('Experiment Version')
dlg.addField('MRI or behavior? (m/b):','')
MorB = dlg.show()  # show dialog and wait for OK or Cancel

if dlg.OK:  # or if ok_data is not None
    print(MorB)
else:
    print('user cancelled')
    core.quit()

if MorB[0]=='m':
    behavFlag=0
    MRIflag=1
    yes_key='1'
    no_key='2'
elif MorB[0]=='b':
    behavFlag=1
    MRIflag=0
    yes_key='1'
    no_key='0'

vis_deg_circ=7#3.5
vis_deg_poly=10#6
circ_ratio=vis_deg_poly/vis_deg_circ
#----------------setup windows and display objects--------

#Here is where you would manage the window screen setting changing the:
#Screen Distance, Size (in pixels), and screen width
##### Setup the display Window
win = visual.Window(
    size=(1280, 800), fullscr=False, screen=0,
    allowGUI=False, allowStencil=False, units='deg',
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

win.recordFrameIntervals=True
# store frame rate of monitor if we can measure it

# Add multiple shutdown keys "at once".
for key in ['q', 'escape']:
    event.globalKeys.add(key, func=core.quit)


#### Setting up a global clock to track initiation of experiment to end
globalClock = core.MonotonicClock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock

n_trials=18
#This function is called to initiate a 20 trial sample of the actual experiment
#When called in runs through a variant of code similar to the actual paradigm
#This should run at the end of the tutorial
def pracBlocks(n_trials=n_trials):

    Total_trials = n_trials
    Switch_probability = 0.5
    Extra_dimension_switch_probability = 0.7
    Intra_dimension_switch_probability = 0.3
    Cue_time = 1
    Pic_time = 3
    ITIs = [1, 1.1, 1.2, 1.3, 1.4, 1.5] # need to jitter this for fMRI, no is just uniform random
    Initial_wait_time = 2
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
    Number_of_stay = round(Total_trials * Switch_probability)
    
    print(Number_of_EDS)
    print(Number_of_IDS)
    print(Number_of_stay)
    
    
    
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
    for i,f in enumerate(np.random.randint(low=0,high=len(faces_list),size=len(Trial_order))) :

        if Pic_order[i] == 'Face':
            Img_Faces[i] = visual.ImageStim(win=win, image=faces_list[f])
            Img_path[i] = faces_list[f]
        if Pic_order[i] =='Scene':
            Img_Scene[i] = visual.ImageStim(win=win, image=scenes_list[f])
            Img_path[i] = scenes_list[f]

    Cue_types = {'fpr': {'cue':'fpr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled) },
                 'fpb': {'cue':'fpb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled) },
                 'fcr': {'cue':'fcr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled) },
                 'fcb': {'cue':'fcb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled) },
                 'dpr': {'cue':'dpr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_donut) },
                 'dcr': {'cue':'dcr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Face', 'cue_stim': copy.copy(circle_donut) },
                 'dpb': {'cue':'dpb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Scene', 'cue_stim': copy.copy(polygon_donut) },
                 'dcb': {'cue':'dcb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_donut) }}
    random_first = randomchoice(list(Cue_types))
    print('\n\n trial order: \n')
    print(len(Trial_order))
    Trial_dict = {}
    for i in range(len(Trial_order)):
        if i == 0 :   #set first trial, randomly select
            Trial_dict[i] = Cue_types[random_first]
        else:
            #print('looking at trial 2')
            if Trial_order[i] == 'Stay':  # stay
                if Trial_dict[i-1]['cue'] in ['fpr', 'fpb']:
                    Trial_dict[i] = Cue_types[randomchoice(['fpr', 'fpb'])].copy() # having .copy() is extremenly important, otherwise it will be
                elif Trial_dict[i-1]['cue'] in ['fcr', 'fcb']: # a link to original object and not "switch"
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

        #set colors of the cue stim # donot circle need to set line color but not fill
        if Trial_dict[i]['cue'] == 'dcr':
            Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
        elif Trial_dict[i]['cue'] == 'dcb':
            Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
        else: # every othre cue set both line and fill color
            Trial_dict[i]['cue_stim'].setLineColor(Trial_dict[i]['Color'])
            Trial_dict[i]['cue_stim'].setFillColor(Trial_dict[i]['Color'])

        # random place scene or face picture
        Trial_dict[i]['pic'] = Pic_order[i]
        if Trial_dict[i]['pic'] == 'Face':
            Trial_dict[i]['pic_stim'] = Img_Faces[i]
        if Trial_dict[i]['pic'] == 'Scene':
            Trial_dict[i]['pic_stim'] = Img_Scene[i]
        Trial_dict[i]['img_path'] = Img_path[i]

    acc=[]
    for trial_num in range(len(Trial_dict)): #range(len(Trial_dict))
        # draw the cue
        ITI=np.random.choice(ITIs,1)[0]
        if trial_num==0:
            win.flip()
            get_ready=visual.TextStim(win=win, name='DemoInstruct',
            text=u'Get ready',alignVert='center', units='norm',
            pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
            color=u'white', colorSpace='rgb', opacity=1,
            languageStyle='LTR',
            depth=0.0);
            get_ready.draw()
            win.flip()
            core.wait(2)

        Trial_dict[trial_num]['cue_stim'].draw()
        win.flip()
        core.wait(Cue_time)
        print(Trial_dict[trial_num])
        # this is for debug
        print(Trial_dict[trial_num]['Trial_type'] )
        print(Trial_dict[trial_num]['cue'])
        print(Trial_dict[trial_num]['Color'])

        # draw the face or scene picture
        Trial_dict[trial_num]['pic_stim'].draw()
        win.flip()
        subRespo = event.waitKeys(maxWait=Pic_time, timeStamped=True, keyList=[yes_key,no_key])
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

        print(subRespo)

        if subRespo==None:
            trial_Corr=0
            #rt=-1
            #subResp='NR'
        elif subRespo[0][0]==corr_resp:
            trial_Corr=1
            #rt=subRespo[0][1]
            #subResp=subRespo[0][0]
        else:
            trial_Corr=0
            #rt=subRespo[0][1]
            #subResp=[0][0]
        win.flip()
        #event.waitKeys()

        acc.append(trial_Corr)
        #Trial_dict[trial_num]['rt']=rt
        #Trial_dict[trial_num]['What_Is_CorrResp']=corr_resp
        #Trial_dict[trial_num]['Subject_Respo']=subResp
        core.wait(ITI)
        #return;
    print('\n')
    print(acc)
    print(np.sum(acc))
    acc_feedback=visual.TextStim(win=win, name='accFeedback',
                    text=u'Your accuracy was '+str((np.sum(acc)/len(Trial_order))*100)+' percent. Would you like to try again?', font=u'Arial',
                    alignVert='center', units='norm',pos=(0, 0), height=0.09, wrapWidth=None, ori=0,color=u'white', colorSpace='rgb', opacity=1,
                    languageStyle='LTR', depth=0.0);
    Proportion = str(np.sum(acc)/n_trials*100)
    print(Proportion)

    acc_feedback.draw()
    win.flip()
    anotherPrac=event.waitKeys(keyList=[yes_key,no_key])

    if anotherPrac[0]==yes_key:
        pracBlocks()


if behavFlag:
    
    #### Welcome screen
    # Initialize components for Routine "Welcome"
    WelcomeClock = core.Clock()
    Welc = visual.TextStim(win=win, name='Welc',
        text=u'Welcome!', units='norm',
        font=u'Arial',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ##### Instructions screen
    # Initialize components for Routine "Instructions"
    InstructionsClock = core.Clock()
    Directions = visual.TextStim(win=win, name='Directions',
        text=u'During this task you will be shown a series of objects followed by photos of faces or scenes. While the photo is present on screen\nyou will be expected to answer a yes/no question about each photo. \n\nEach specific yes/no question will be dependent on the object presented before the photo. In order for the correct response to be recorded, note that YES is farthest left, and NO is farthest right. \n\nPress YES to continue.',
        font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ###### Demo Instructions and Special Figures
    DemoInstruct = visual.TextStim(win=win, name='DemoInstruct',
        text=u'Now we will show you a demo of the task.\nPlease listen carefully to the experimenter as they guide you through the task.\n\n Press NO to continue.', font=u'Arial',
        alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    KeyConstant = visual.TextStim(win=win, name='KeyConstant',
        text=u'Press Any Key to Continue',
        alignVert='center', units='norm',
        pos=(-.5, .9), height=0.06, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ShapeMatters = visual.TextStim(win=win, name='ShapeMatters',
        text=u'If FILLED, then SHAPE matters!',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0, #height =0.09
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Arrow = visual.TextStim(win=win, name='Arrow',
        text=u'←',
        alignVert='center', units='norm',
        pos=(.8, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Equals = visual.TextStim(win=win, name='Equals',
        text=u'=',
        alignVert='center', units='norm',
        pos=(0, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Face_txt = visual.TextStim(win=win, name='Face_txt',
        text=u'FACE',
        alignVert='center', units='norm',
        pos=(.5, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Scene_txt = visual.TextStim(win=win, name='Scene_txt',
        text=u'SCENE',
        alignVert='center', units='norm',
        pos=(.55, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Question_Mrk = visual.TextStim(win=win, name='Question_Mrk',
        text=u'?',
        alignVert='center', units='norm',
        pos=(.5, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Donut_Stim = visual.TextStim(win=win, name='Donut_Stim',
        text=u'Now, what if the object is no longer filled?',
        alignVert='center', units='norm',
        pos=(0, .35), height=0.15, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Instruct_new = visual.TextStim(win=win, name='Instruct_new',
        text=u'Before, FILLED = SHAPE, now EMPTY = COLOR',
        alignVert='center', units='norm',
        pos=(0, -.25), height=0.15, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ColorMatters = visual.TextStim(win=win, name='ColorMatters',
        text=u'If EMPTY, then COLOR matters!',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Quiz_txt=visual.TextStim(win=win, name='quiz',
        text=u'If you see this object, what is the correct response to this picture?',
        alignVert='center', units='norm',
        pos=(0, .75), height=0.075, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Keep_demo=visual.TextStim(win=win, name='keepDemo',
        text=u'Do you feel ready to move on to a practice?',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Feedback_txt=[visual.TextStim(win=win, name='Correct!',
        text=u'CORRECT!',font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0),visual.TextStim(win=win, name='Incorrect',
        text=u'Oops, try another',font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0)]
    
    ##### Central fixations
    # Initialize components for Routine "Fixation"
    FixationClock = core.Clock()
    Fix_Cue = visual.TextStim(win=win, name='Fix_Cue',
        text=u'+', units='norm',
        font=u'Arial',
        pos=(0, 0), height=0.3, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ##### Load Face and Scene pictures into stim objects, organized into a dict
    #Dictionaries and the corresponding file paths
    direc = os.getcwd()+'/localizer_stim/' #_thisDir #'/Users/mpipoly/Desktop/Psychopy/localizer_stim/' #always setup path on the fly in case you switch computers
    
    
    #### For Slow intro
    Face_Stim = visual.ImageStim(
        win=win, image=direc + 'faces/1.jpg', size=(7,7),
        units=None, pos=(6,0))
    Scene_Stim = visual.ImageStim(
        win=win, image=direc + 'scenes/2.jpg',size=(7,7),
        units=None, pos=(6,0))
    
    
    ##### Setup Cue stim objects
    #Overlapping vertices for below
    #red = [1.0,-1,-1]
    #blue = [0, 0, 255]
    #color = 'green'
    scale = 0.7
    donutVert = [[(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)],[(-.15,-.15),(-.15,.15),(.15,.15),(.15,-.15)]]
    simpleVert= [(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)]
    
    #filled circle # setting colors as None now, will change them when setting up trial sequence
    circle_filled = visual.Circle(
        win=win, name='circle_filled', units='deg', radius=0.5,
        pos=(0,0), size=[vis_deg_circ,vis_deg_circ], color=None, fillColor=None,
        fillColorSpace='rgb255')
    
    #donut circle
    circle_donut = visual.Circle(
        win=win, name='circle_donut', units='deg', radius=0.5,
        fillColor=None, lineWidth=25, pos=(0,0),
        size=[vis_deg_circ,vis_deg_circ], lineColorSpace='rgb255', lineColor=None,
        interpolate=True)
    
    polygon_filled = visual.ShapeStim(
        win=win, name='polygon_filled', units='deg', vertices=simpleVert,
        fillColor=None, fillColorSpace='rgb255',
        lineWidth=0, size=[vis_deg_poly,vis_deg_poly], pos=(0,0))
    
    polygon_donut = visual.ShapeStim(
        win=win, name='polygon_donut', units='deg', vertices=donutVert,
        fillColor=None, fillColorSpace='rgb255', lineWidth=3,
        size=[vis_deg_poly,vis_deg_poly], pos=(0,0))

elif MRIflag:
    
    #### Welcome screen
    # Initialize components for Routine "Welcome"
    WelcomeClock = core.Clock()
    Welc = visual.TextStim(win=win, name='Welc',
        text=u'Welcome!', units='norm',
        font=u'Arial',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ##### Instructions screen
    # Initialize components for Routine "Instructions"
    InstructionsClock = core.Clock()
    Directions = visual.TextStim(win=win, name='Directions',
        text=u'During this task you will be shown a series of objects followed by photos of faces or scenes. While the photo is present on screen\nyou will be expected to answer a yes/no question about each photo. \n\nEach specific yes/no question will be dependent on the object presented before the photo. In order for the correct response to be recorded, note that YES is farthest left, and NO is farthest right. \n\nPress YES to continue.',
        font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ###### Demo Instructions and Special Figures
    DemoInstruct = visual.TextStim(win=win, name='DemoInstruct',
        text=u'Now we will show you a demo of the task.\nPlease listen carefully to the experimenter as they guide you through the task.\n\n Press NO to continue.', font=u'Arial',
        alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    KeyConstant = visual.TextStim(win=win, name='KeyConstant',
        text=u'Press Any Key to Continue',
        alignVert='center', units='norm',
        pos=(-.5, .9), height=0.06, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ShapeMatters = visual.TextStim(win=win, name='ShapeMatters',
        text=u'If FILLED, then SHAPE matters!',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0, #height =0.09
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Arrow = visual.TextStim(win=win, name='Arrow',
        text=u'←',
        alignVert='center', units='norm',
        pos=(.8, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Equals = visual.TextStim(win=win, name='Equals',
        text=u'=',
        alignVert='center', units='norm',
        pos=(0, 0), height=0.4, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Face_txt = visual.TextStim(win=win, name='Face_txt',
        text=u'FACE',
        alignVert='center', units='norm',
        pos=(.6, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Scene_txt = visual.TextStim(win=win, name='Scene_txt',
        text=u'SCENE',
        alignVert='center', units='norm',
        pos=(.5, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Question_Mrk = visual.TextStim(win=win, name='Question_Mrk',
        text=u'?',
        alignVert='center', units='norm',
        pos=(.5, 0), height=0.2, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Donut_Stim = visual.TextStim(win=win, name='Donut_Stim',
        text=u'Now, what if the object is no longer filled?',
        alignVert='center', units='norm',
        pos=(0, .35), height=0.15, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Instruct_new = visual.TextStim(win=win, name='Instruct_new',
        text=u'Before, FILLED = SHAPE, now EMPTY = COLOR',
        alignVert='center', units='norm',
        pos=(0, -.25), height=0.15, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ColorMatters = visual.TextStim(win=win, name='ColorMatters',
        text=u'If EMPTY, then COLOR matters!',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Quiz_txt=visual.TextStim(win=win, name='quiz',
        text=u'If you see this object, what is the correct response to this picture?',
        alignVert='center', units='norm',
        pos=(0, .75), height=0.075, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Keep_demo=visual.TextStim(win=win, name='keepDemo',
        text=u'Do you feel ready to move on to a practice?',
        alignVert='center', units='norm',
        pos=(0, .5), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    Feedback_txt=[visual.TextStim(win=win, name='Correct!',
        text=u'CORRECT!',font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0),visual.TextStim(win=win, name='Incorrect',
        text=u'Oops, try another',font=u'Arial', alignVert='center', units='norm',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0)]
    
    ##### Central fixations
    # Initialize components for Routine "Fixation"
    FixationClock = core.Clock()
    Fix_Cue = visual.TextStim(win=win, name='Fix_Cue',
        text=u'+', units='norm',
        font=u'Arial',
        pos=(0, 0), height=0.3, wrapWidth=None, ori=0,
        color=u'white', colorSpace='rgb', opacity=1,
        languageStyle='LTR',
        depth=0.0);
    
    ##### Load Face and Scene pictures into stim objects, organized into a dict
    #Dictionaries and the corresponding file paths
    direc = os.getcwd()+'/localizer_stim/' #_thisDir #'/Users/mpipoly/Desktop/Psychopy/localizer_stim/' #always setup path on the fly in case you switch computers
    
    
    #### For Slow intro
    Face_Stim = visual.ImageStim(
        win=win, image=direc + 'faces/1.jpg',
        units=None, pos=(4,0))
    Scene_Stim = visual.ImageStim(
        win=win, image=direc + 'scenes/2.jpg',
        units=None, pos=(4,0))
    
    
    ##### Setup Cue stim objects
    #Overlapping vertices for below
    #red = [1.0,-1,-1]
    #blue = [0, 0, 255]
    #color = 'green'
    scale = 0.7
    donutVert = [[(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)],[(-.15,-.15),(-.15,.15),(.15,.15),(.15,-.15)]]
    simpleVert= [(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)]
    
    #filled circle # setting colors as None now, will change them when setting up trial sequence
    circle_filled = visual.Circle(
        win=win, name='circle_filled', units='deg', radius=0.5,
        pos=(0,0), size=[vis_deg_circ,vis_deg_circ], color=None, fillColor=None,
        fillColorSpace='rgb255')
    
    #donut circle
    circle_donut = visual.Circle(
        win=win, name='circle_donut', units='deg', radius=0.5,
        fillColor=None, lineWidth=25, pos=(0,0),
        size=[vis_deg_circ,vis_deg_circ], lineColorSpace='rgb255', lineColor=None,
        interpolate=True)
    
    polygon_filled = visual.ShapeStim(
        win=win, name='polygon_filled', units='deg', vertices=simpleVert,
        fillColor=None, fillColorSpace='rgb255',
        lineWidth=0, size=[vis_deg_poly,vis_deg_poly], pos=(0,0))
    
    polygon_donut = visual.ShapeStim(
        win=win, name='polygon_donut', units='deg', vertices=donutVert,
        fillColor=None, fillColorSpace='rgb255', lineWidth=3,
        size=[vis_deg_poly,vis_deg_poly], pos=(0,0))




###### Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

#### First slide to appear, welcome to the participaint

Welc.draw()
win.flip()
event.waitKeys(maxWait=4)

#### First set of instructions to test whether participant understand what each button corresponds 2

Directions.draw()
win.flip()
event.waitKeys(keyList=[yes_key]) #Key list needs to be changed for fMRI to match button box output

DemoInstruct.draw()
win.flip()
event.waitKeys(keyList=[no_key]) #Key list needs to be changed for fMRI to match button box output

#### Begining of SLOW tutorial... step by step to ensure hierarchical structure is digested

DEMOAGAIN=1 # CHANGE TO 1
while DEMOAGAIN:


#### Here we reinforce the first interdimensional shift rules
    KeyConstant.setAutoDraw(True)
    ShapeMatters.pos=(0,0.5)
    ShapeMatters.draw()
    blue_sq=copy.copy(polygon_filled) # Filled Blue Square
    blue_sq.setColor('blue')
    blue_sq.pos=(-0.5*vis_deg_poly,0) # -4,0
    blue_sq.size=(vis_deg_poly+2,vis_deg_poly+2)
    red_sq=copy.copy(polygon_filled) # Filled Red Square
    red_sq.setColor('red')
    red_sq.pos=(0.5*vis_deg_poly,0) #2,0
    red_sq.size=(vis_deg_poly+2,vis_deg_poly+2)
    #Arrow.height=(.2)
    Arrow.draw()
    red_sq.draw()
    blue_sq.draw()
    win.flip()
    event.waitKeys()

#### Subject should grasp now that color is not important when the shape is filled!
#### Filled Squares equal face photo, REGARDLESS of COLOR

    Face_txt.pos=(.5,0)
    Face_txt.draw()
    Equals.pos=(0,0)
    Equals.draw()
    blue_sq.draw() #Filled Blue Square
    win.flip()
    event.waitKeys()

#### This scheme is being reinforced by presenting a subsequent blue
#### stimulus that happens to have a different Shape. This change in Shape
#### indicates that an interdimensional shift has occurred between Square and Circle
    blue_circ=copy.copy(circle_filled)
    blue_circ.setColor('blue')
    blue_circ.pos=(-0.5*circ_ratio*vis_deg_circ,0)
    blue_circ.draw() # Filled Blue circle
    Equals.draw()
    Scene_txt.pos=(.6,0)
    Scene_txt.draw()
    win.flip()
    event.waitKeys()

#### Subject should loosely understand that FILLED shape Interdimensional shifts are:
#### Filled Polygon (square) = Face, Filled Circle = Scene
    red_circ=copy.copy(circle_filled)
    red_sq.pos=(-.5*vis_deg_poly,.30*vis_deg_poly) #(-3.5,2)
    red_circ.pos=(-.5*circ_ratio*vis_deg_circ,-.30*circ_ratio*vis_deg_circ)#(-3.5,-2)
    red_circ.setColor('red')
    ShapeMatters.pos=(0,5.5)
    Equals_1=copy.copy(Equals)
    Equals_2=copy.copy(Equals)
    Equals_1.pos=(.2,.35)
    Equals_2.pos=(.2,-.35)
    red_sq.draw()
    red_circ.draw()
    ShapeMatters.draw()
    Equals_1.draw()
    Equals_2.draw()
    win.flip()
    event.waitKeys()

#### If the rule has been somewhat understood, here is a slow review where participant
#### should be probed by RA to see how they interpret these stimuli
    red_sq.pos=(-.5*vis_deg_poly,.30*vis_deg_poly)#(-3.5,2)
    red_circ.pos=(-.5*circ_ratio*vis_deg_circ,-.30*circ_ratio*vis_deg_circ)#(-3.5,-2)
    #ShapeMatters.pos=(0,5.5)
    #Equals_1.pos=(.2,.35)
    #Equals_2.pos=(.2,-.35)
    Question_Mrk.pos=(.5,0)
    red_sq.draw()
    red_circ.draw()
    ShapeMatters.draw()
    Equals_1.draw()
    Equals_2.draw()
    Question_Mrk.draw()
    win.flip()
    event.waitKeys()

#### initiate ? as participant thinks, press button again to
#### reveal first answer
    #red_sq.pos=(-3.5,2)
    #red_circ.pos=(-3.5,-2)
    #ShapeMatters.pos=(0,5.5)
    #Equals_1.pos=(.25,.35)
    #Equals_2.pos=(.25,-.35)
    Face_txt.pos=(.5,.35)
    Face_txt.draw()
    red_sq.draw()
    red_circ.draw()
    ShapeMatters.draw()
    Equals_1.draw()
    Equals_2.draw()
    win.flip()
    event.waitKeys()

#### Does the participant now know what the second answer will be?
    red_sq.pos=(-.5*vis_deg_poly,.30*vis_deg_poly)#(-3.5,2)
    red_circ.pos=(-.5*circ_ratio*vis_deg_circ,-.30*circ_ratio*vis_deg_circ)
    #ShapeMatters.pos=(0,5.5)
    #Equals_1.pos=(.25,.35)
    #Equals_2.pos=(.25,-.35)
    Scene_txt.pos=(.55,-.35)
    Scene_txt.draw()
    red_sq.draw()
    red_circ.draw()
    ShapeMatters.draw()
    Equals_1.draw()
    Equals_2.draw()
    win.flip()
    event.waitKeys()

#### Now we move into discussing the rules for the extra dimensional switch
#### This means that we are no longer concerned with SHAPE, but COLOR
    Donut_Stim.draw()
    win.flip()
    event.waitKeys()

    Donut_Stim.draw()
    Instruct_new.draw()
    win.flip()
    event.waitKeys()

#### Participant is now informed of how this secondary stimulus looks
#### Donots that are empty are indicative of ignoring shape and focusing on color
    ColorMatters.draw()
    circ_B_do=copy.copy(circle_donut)
    circ_B_do.lineColor=('blue')
    circ_B_do.pos=(-.5*circ_ratio*vis_deg_circ,0)#(-5,0)
    circ_B_do.size=(vis_deg_circ-1,vis_deg_circ-1)
    circ_R_do=copy.copy(circle_donut)
    circ_R_do.lineColor=('red')
    circ_R_do.pos=(.5*circ_ratio*vis_deg_circ,0)#(3,0)
    circ_R_do.size=(vis_deg_circ-1,vis_deg_circ-1)
    Arrow.height=(.2)
    Arrow.draw()
    circ_B_do.draw()
    circ_R_do.draw()
    win.flip()
    event.waitKeys()

#### COLOR no SHAPE predicts whether the subsequent photo presented
#### will be a face or a scene
    circ_R_do.lineColor=('red')
    circ_R_do.pos=(-.5*circ_ratio*vis_deg_circ,0)
    circ_R_do.size=(vis_deg_circ-1,vis_deg_circ-1)
    Arrow.height=(.2)
    Face_txt.pos=(.35,0)
    Equals.draw()
    Face_txt.draw()
    Arrow.draw()
    circ_R_do.draw()
    win.flip()
    event.waitKeys()

#### The above is an example of this and the below reinforces shape
#### no longer matters
    circ_B_do.lineColor=('blue')
    circ_B_do.pos=(-.5*circ_ratio*vis_deg_circ,0)#(-7,0)
    circ_B_do.size=(vis_deg_circ-1,vis_deg_circ-1)
    Equals.draw()
    Scene_txt.pos=(.35,0)
    Scene_txt.draw()
    circ_B_do.draw()
    win.flip()
    event.waitKeys()

    #ColorMatters.draw()
    #Question_Mrk.pos=(0,-2)
    #Question_Mrk.draw()
    #win.flip()
    #event.waitKeys()
    #
    #circ_B_do.lineColor=('red')
    #circ_B_do.pos=(-7,0)
    #circ_B_do.size=(6,6)
    #Arrow.height=(.2)
    #Question_Mrk.pos=(0,2)
    #Equals.draw()
    #Question_Mrk.draw()
    #Arrow.draw()
    #circ_B_do.draw()
    #win.flip()
    #event.waitKeys()
    #
    #circ_B_do.lineColor=('red')
    #circ_B_do.pos=(-7,0)
    #circ_B_do.size=(6,6)
    #Arrow.height=(.2)
    #Face_txt.pos=(.25,0)
    #Equals.draw()
    #Face_txt.draw()
    #Arrow.draw()
    #circ_B_do.draw()
    #win.flip()
    #event.waitKeys()

#### To index whether the participant understands this, we review
#### Can participant predict (ignoring the shape) what the answer
#### to red and blue is?
    sq_B_do=copy.copy(red_sq)
    sq_B_do.fillColor=None
    sq_B_do.lineWidth=25
    sq_B_do.lineColor=('blue')
    sq_B_do.pos=(-.25*circ_ratio*vis_deg_circ,-0.35*circ_ratio*vis_deg_circ )#(-2,-4)
    sq_B_do.size=(vis_deg_poly,vis_deg_poly)
    sq_R_do=copy.copy(sq_B_do)
    sq_R_do.lineColor=('red')
    sq_R_do.pos=(-.25*vis_deg_poly,0.35*vis_deg_poly)#(-2,4)
    Arrow.height=(.2)
    #Equals_1.pos=(.25,.4)
    #Equals_2.pos=(.25,-.4)
    Equals_1.draw()
    #Question_Mrk.pos=(.5,0)
    Question_Mrk.draw()
    Equals_2.draw()
    #Face_txt.pos=(.5,.4)
    #Scene_txt.pos=(.7,-.4)
    #Scene_txt.draw()
    #Face_txt.draw()
    #Arrow.draw()
    sq_B_do.draw() # Empty Blue Square
    sq_R_do.draw() # Empty Red Square
    win.flip()
    event.waitKeys()


    Equals_1.draw()
    Equals_2.draw()
    Face_txt.pos=(.5,.4)
    Scene_txt.pos=(.6,-.4)
    Scene_txt.draw()
    Face_txt.draw()
    #Arrow.draw()
    sq_B_do.draw() # Empty Blue Square
    sq_R_do.draw() # Empty Red Square
    win.flip()
    event.waitKeys()

    Quiz_txt.draw()
    red_sq.pos=(-0.7*vis_deg_poly,0)#(-4,0) #red FILLED square
    red_sq.draw()
    Equals.draw()
    Face_Stim.draw()
    win.flip()
    resp1=event.waitKeys()
    if resp1[0]==yes_key:
        Feedback_txt[0].draw()
    else:
        Feedback_txt[1].draw()
    win.flip()
    event.waitKeys()

    Quiz_txt.draw()
    circ_R_do.pos=(-0.7*circ_ratio*vis_deg_circ,0) #red EMPTY circle
    circ_R_do.draw()
    Equals.pos=(0,0)
    Equals.draw()
    Question_Mrk.pos=(0,.15)
    Question_Mrk.draw()
    Scene_Stim.draw()
    win.flip()
    resp2=event.waitKeys()
    print(resp2)
    if resp2[0]==no_key:
        Feedback_txt[0].draw()
    else:
        Feedback_txt[1].draw()
    win.flip()
    event.waitKeys()

    Quiz_txt.draw()
    blue_circ.pos=(-0.7*circ_ratio*vis_deg_circ,0)#(-10,0) #blue FILLED circle
    blue_circ.draw()
    Equals.draw()
    Question_Mrk.draw()
    Scene_Stim.draw()
    win.flip()
    resp2=event.waitKeys()
    print(resp2)
    if resp2[0]==yes_key:
        Feedback_txt[0].draw()
    else:
        Feedback_txt[1].draw()
    win.flip()
    event.waitKeys()


    Quiz_txt.draw()
    red_circ.pos=(-0.7*circ_ratio*vis_deg_circ,0)#(-10,0) #red FILLED circle
    red_circ.draw()
    Equals.draw()
    Question_Mrk.draw()
    Scene_Stim.draw() #calls on scene dictionary index
    win.flip()
    resp2=event.waitKeys()
    print(resp2)
    if resp2[0]==yes_key:
        Feedback_txt[0].draw()
    else:
        Feedback_txt[1].draw()
    win.flip()
    event.waitKeys()

    Quiz_txt.draw()
    sq_B_do.pos=(-0.7*vis_deg_poly,0)#(-10,0) #blue EMPTY square
    sq_B_do.draw()
    Equals.draw()
    Question_Mrk.draw()
    Face_Stim.draw() # calls on face dictionary index
    win.flip()
    resp2=event.waitKeys()
    print(resp2)
    if resp2[0]==no_key:
        Feedback_txt[0].draw()
    else:
        Feedback_txt[1].draw()
    win.flip()
    event.waitKeys()

    Keep_demo.draw()
    win.flip()
    sub_resp=event.waitKeys(keyList=[no_key,yes_key])

#Here an option os presented to start over or move on
    if sub_resp[0]==yes_key:
        DEMOAGAIN=0
    elif sub_resp[0]==no_key:
        DEMOAGAIN=1

#### Dynamic Practice Initiation
KeyConstant.setAutoDraw(False)

pracBlocks()
