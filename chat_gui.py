from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.recycleview import RecycleView
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window

import threading

import main
import speech_recognition as sr
from pygame import mixer
import pyttsx3
from gtts import gTTS
import os
import time
        
class AppLayout(GridLayout):
    def __init__(self,**kwargs):
        super(AppLayout, self).__init__(**kwargs)
        '''if(Window.keycode==40):'''
        Window.bind(on_key_down=self.run_)
    def run_(self, instance,keyboard, keycode, text, modifiers):
        if self.av.focus and keycode==40:
            self.send_msg(self.av.text)
    i=0
    def string_adj(self,val):
        msg_len=len(val)
        if(msg_len>6):
            x=(msg_len-6)
            return "  "*x+val
        elif(msg_len<6):
            x=6-msg_len
            return val+"  "*x
        else:
            return 'Can\'t process your request'
    def listen(self):
        talk = pyttsx3.init()
        r = sr.Recognizer()
        r.pause_threshold = 0.7
        r.energy_threshold = 400
        with sr.Microphone() as source:
            try:
                audio = r.listen(source,timeout=4)
                msg = str(r.recognize_google(audio))
                self.av.text=msg
                talk.runAndWait()
            except sr.UnknownValueError:
                print('Google Speech Recognition could not understand audio')

            except sr.RequestError as e:
                print('Could not request results from Google Speech Recognition Service')
    def speak(self):
        talk = pyttsx3.init()
        voices=talk.getProperty('voices')
        talk.setProperty('voices', voices[0].id)
        talk.say(self.response1)
        talk.runAndWait()
        self.av.focus=True
    def send_msg(self, val):
        msg=val
        self.response1=main.response(msg)
        self.rv.data.insert(self.i,{'text':self.string_adj('User: '+msg)})
        self.i=self.i+1
        if self.response1 != None:
            
            self.rv.data.insert(self.i,{'text':self.string_adj('Hera: '+self.response1)})
        else:
            self.response1="Sorry! Can't able to process your request! eeeee :("
            self.rv.data.insert(self.i,{'text':self.string_adj('Hera: '+self.response1)})
        self.av.text=''
        self.i=self.i+1
        #self.speak()
        t1=threading.Thread(target=self.speak)
        t1.start()
        
        
               

class chat(App):
    
    def build(self):
        return AppLayout()
chat().run()
