# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:25:10 2019

@author: cse.repon
"""
import os
import tkinter as tk
import sys
if "tkinter" not in sys.modules:
    import tkinter as tk
root= tk.Tk()
root.resizable(False, False)
canvas1 = tk.Canvas(root, width = 840, height = 490)
root.wm_title("Hand Gesture Control")
canvas1.pack()

root.iconbitmap(os.path.join("resources\HandGesture.ico"))
background_image=tk.PhotoImage(file = (os.path.join("resources\handBackground.png"))) 
background_imageInit=tk.PhotoImage(file = (os.path.join("resources\handBackgroundInit.png"))) 
background_label = tk.Label(root, image=background_imageInit)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#labelBack = tk.PhotoImage(file = (os.path.join("resources\ButtonStyle.png"))) 
initText = tk.PhotoImage(file = (os.path.join("resources\initText.png"))) 
photo = tk.PhotoImage(file = (os.path.join("resources\ConfigButton.png"))) 
photo2 = tk.PhotoImage(file = (os.path.join("resources\previewButton.png"))) 
photo3 = tk.PhotoImage(file = (os.path.join("resources\RestartButton.png"))) 

isVisible = False
isVisiblePreview = False

subRate = tk.Scale(root, label="Threshold",from_=5, to=15, orient=tk.HORIZONTAL,width=20)
dynEnvButton = tk.Checkbutton( root, text = "Dynamic Environment",width=20)
handColorButton = tk.Checkbutton( root, text = "Use Color Prediction",width=20)
handColorTrackButton = tk.Checkbutton( root, text = "Use Color Tracking",width=20)
preview1 = tk.Label(root, text= 'preview1', fg='green', font=('helvetica', 25, 'bold'))
preview2 = tk.Label(root, text= 'preview2', fg='green', font=('helvetica', 25, 'bold'))

subRate.set(8)

dynId = canvas1.create_window(110, 300, window=dynEnvButton, state='hidden')
handId = canvas1.create_window(110, 336, window=handColorButton, state='hidden')
handTrackId = canvas1.create_window(110, 372, window=handColorTrackButton, state='hidden')
subRateId = canvas1.create_window(69, 431, window=subRate, state='hidden')
preview1Id = canvas1.create_window(360, 375, window=preview1, state='hidden')
preview2Id = canvas1.create_window(660, 375, window=preview2, state='hidden')



def showHide (): 
    global isVisible
    if(isVisible == False):
        canvas1.itemconfigure(dynId, state='normal')
        canvas1.itemconfigure(handId, state='normal')
        canvas1.itemconfigure(handTrackId, state='normal')
        canvas1.itemconfigure(subRateId, state='normal')
        isVisible = True
    else:
        canvas1.itemconfigure(dynId, state='hidden')
        canvas1.itemconfigure(handId, state='hidden')
        canvas1.itemconfigure(handTrackId, state='hidden')
        canvas1.itemconfigure(subRateId, state='hidden')
        isVisible = False
        
def preview (): 
    global isVisiblePreview
    if(isVisiblePreview == False):
        canvas1.itemconfigure(preview1Id, state='normal')
        canvas1.itemconfigure(preview2Id, state='normal')
        isVisiblePreview = True
    else:
        canvas1.itemconfigure(preview1Id, state='hidden')
        canvas1.itemconfigure(preview2Id, state='hidden')
        isVisiblePreview = False
        
gestureLabel = tk.Label(root, text= 'Detecting Gesture...', fg='green', font=('helvetica', 25, 'bold'))
button1 = tk.Button(command=showHide, image = photo,relief=tk.FLAT, bg='#222',activebackground='#fff',highlightcolor='#222')
button2 = tk.Button(command=preview, image = photo2,relief=tk.FLAT, bg='#222',activebackground='#fff',highlightcolor='#222')
button3 = tk.Button(image = photo3,relief=tk.FLAT, bg='#222',activebackground='#fff',highlightcolor='#222')


canvas1.create_window(100, 250, window=button1)
canvas1.create_window(300, 250, window=button2)
canvas1.create_window(300, 50, window=button3)
canvas1.create_window(600, 50, window=gestureLabel)

def update():
    #root.mainloop()
    root.update_idletasks()
    root.update()
  
#while True:
    #update()
    