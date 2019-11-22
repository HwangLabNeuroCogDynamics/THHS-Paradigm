## Practice ThalHi for David, to be run with the tutorial video ##


from __future__ import absolute_import, division, print_function
import psychopy.visual
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
import imageio

#----------------setup windows and display objects--------
##### Setup the display Window
win = visual.Window(
    size=(1280, 800), fullscr=True, screen=0,
    allowGUI=False, allowStencil=False, units='deg',
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

#direc = os.path.dirname(os.path.abspath(__file__))
#print(direc)
#imageio.plugins.ffmpeg.download()
#tutorial_video=visual.MovieStim3(win=win,filename=os.getcwd()+'\ThalHi_tutorial_vid.mp4',pos=(0,0),volume=0.0)

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


yes_key='1'
no_key='0'

n_trials=18
#This function is called to initiate a 20 trial sample of the actual experiment
#When called in runs through a variant of code similar to the actual paradigm
#This should run at the end of the tutorial
def pracBlocks(n_trials=n_trials):
    
#    while tutorial_video.status != visual.FINISHED:
#        tutorial_video.draw()
#        win.flip()
#        if event.getKeys():
#            breaks
    
    #tutorial_video.draw()
    #tutorial_video.play()
    #win.flip()
    #core.wait(5)

    Total_trials = n_trials
    Switch_probability = 0.5
    Extra_dimension_switch_probability = 0.7
    Intra_dimension_switch_probability = 0.3
    Cue_time = 2#1
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
    direc = os.getcwd()+'\localizer_stim/' #_thisDir #'/Users/mpipoly/Desktop/Psychopy/localizer_stim/' #always setup path on the fly in case you switch computers
    ext = 'scenes\*.jpg' #file delimiter
    faces_ext = 'faces\*.jpg'
    faces_list = glob.glob(direc + faces_ext)
    scenes_list = glob.glob(direc + ext)
    print(faces_list)
    print(scenes_list)
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

    Cue_types = {'fpr': {'cue':'fpr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled_r) },
                 'fpb': {'cue':'fpb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_filled_b) },
                 'fcr': {'cue':'fcr', 'Color': 'red', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled_r) },
                 'fcb': {'cue':'fcb', 'Color': 'blue', 'Texture': 'Filled', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_filled_b) },
                 'dpr': {'cue':'dpr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Face', 'cue_stim': copy.copy(polygon_donut_r) },
                 'dcr': {'cue':'dcr', 'Color': 'red', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Face', 'cue_stim': copy.copy(circle_donut_r) },
                 'dpb': {'cue':'dpb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Polygon', 'Task': 'Scene', 'cue_stim': copy.copy(polygon_donut_b) },
                 'dcb': {'cue':'dcb', 'Color': 'blue', 'Texture': 'Donut', 'Shape': 'Circle', 'Task': 'Scene', 'cue_stim': copy.copy(circle_donut_b) }}
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
        subRespo = event.waitKeys(maxWait=Pic_time, timeStamped=True, keyList=[yes_key,no_key,'escape'])
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
        elif subRespo[0][0]==corr_resp:
            trial_Corr=1
        elif subRespo[0][0]=='escape':
            core.quit()
        else:
            trial_Corr=0
        win.flip()
        #event.waitKeys()

        acc.append(trial_Corr)
        core.wait(ITI)
        #return;
    print('\n')
    print(acc)
    print(np.sum(acc))
    acc_feedback=visual.TextStim(win=win, name='accFeedback',
                    text=u'Your accuracy was '+str(int((np.sum(acc)/len(Trial_order))*100))+' percent. Would you like to try again? Press 1 or 0', font=u'Arial',
                    alignVert='center', units='norm',pos=(0, 0), height=0.09, wrapWidth=None, ori=0,color=u'white', colorSpace='rgb', opacity=1,
                     depth=0.0);
    Proportion = str(np.sum(acc)/n_trials*100)
    print(Proportion)

    acc_feedback.draw()
    win.flip()
    anotherPrac=event.waitKeys(keyList=[yes_key,no_key])

    if anotherPrac[0]==yes_key:
        pracBlocks()

pracBlocks()