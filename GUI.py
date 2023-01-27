'''
-----------------------------------------------------CODE PAGE 1 -----------------------------------------------------------------
'''

import tkinter as tk
from time import *
from tkinter import *
from PIL import Image, ImageTk
from livetesting import printsentiment
import warnings
warnings.filterwarnings("ignore")

window = tk.Tk(className=' Speech Emotion Recognition App')
window.geometry("500x700")
window.configure(bg='#6959CD')
window.iconbitmap("PICTURE\\Logo.ico")



# Open a New Image
image = Image.open("PICTURE\\ondeSonore.jpg")
# Resize Image using resize function
resized_image = image.resize((150, 150), Image.ANTIALIAS)
# Convert the image into PhotoImage
img = ImageTk.PhotoImage(resized_image)

s = Image.open('PICTURE\\Start.gif')
start = s.resize((120, 50))
useStart = ImageTk.PhotoImage(start)

micro = Image.open("PICTURE\\micro.jpg")
resized_micro = micro.resize((30, 30))
mcr = ImageTk.PhotoImage(resized_micro)

def changepage():
    global pagenum, window
    for widget in window.winfo_children():
        widget.pack_forget()
        widget.place_forget()
    if pagenum == 1:
        page2(window)
        pagenum = 2
    else:
        page1(window)
        pagenum = 1

def pagehistory():
    for widget in window.winfo_children():
        widget.pack_forget()
        widget.place_forget()
    page3(window)

def topage2():
    for widget in window.winfo_children():
        widget.pack_forget()
        widget.place_forget()
    page2(window)


logo = Label(window, image=img, width=150, height=150, bg='#6959CD')
greeting = tk.Label(window, text="Hey, \n \n Welcome to the Speech Emotion Recognition Application!",
                    font=("Times New Roman", 16), bg='#6959CD',fg='white')
message = '''
             Speech Emotion Recognition, abbreviated as SER, 
           is the act of attempting to recognize human emotion 
                      and affective states from speech.

          This is capitalizing on the fact that voice often reflects
                 underlying emotion through tone and pitch.
                    Click on START to try on your voice!

    '''
text_box = Text(
        window,
        font=("Times New Roman", 13),
        bg='#6959CD',
        fg='black',
        height=8,
        width=65,
        relief=SUNKEN,
        bd=0,

)

startbutton = Button(window, image= useStart,bd=0, bg='#00FF7F', command=changepage)
labelmicro = Label(window, image=mcr, bg='#6959CD')

def page1(window):
    logo.pack(pady=10)
    greeting.pack(pady=21, side=TOP)
    text_box.place(x=35,y=280)
    text_box.insert('end', message)
    text_box.config(state='disabled')
    startbutton.place(x=180,y=500)
    labelmicro.place(x=225,y=555)


'''
--------------------------------------------CODE PAGE 2--------------------------------------------------------------------------
'''
wi = 200
he = 55
canvas_lg = Canvas(window, width=wi, height=he, highlightthickness=0,bg='#6959CD')


def change_img(canvas_logo,num):

      tab_img = [f"WAVE\\frame-01.gif", f"WAVE\\frame-02.gif", f"WAVE\\frame-03.gif", f"WAVE\\frame-04.gif", f"WAVE\\frame-05.gif", f"WAVE\\frame-06.gif", f"WAVE\\frame-07.gif", f"WAVE\\frame-08.gif", f"WAVE\\frame-09.gif", f"WAVE\\frame-10.gif"]  # images gif en 90x90
      for i in tab_img:
          photo = PhotoImage(file=i)
          canvas_logo.create_image(wi / 2, he / 2, image=photo)
          canvas_logo.itemconfigure('image', image=photo)
          canvas_logo.update()
          sleep(1/14)

      window.after(1000, change_img(canvas_logo,num+1))

def myCanvas():
    photo = PhotoImage(file=f"WAVE\\frame-01.gif")
    canvas_lg.create_image(wi /2 , he /2, image=photo)
    #canvas_lg.place(x=180, y=15)
    change_img(canvas_lg,8)
    canvas_lg.destroy()


prev = Image.open("PICTURE\\prev.png")
resized_prev = prev.resize((40, 40), Image.ANTIALIAS)
prev_image = ImageTk.PhotoImage(resized_prev)

his = Image.open("PICTURE\\history.png")
resized_his = his.resize((40, 40), Image.ANTIALIAS)
his_image = ImageTk.PhotoImage(resized_his)

prevbutton = tk.Button(window, image=prev_image, bg='#6959CD', command=changepage)
historybutton = tk.Button(window, image=his_image, bg='#6959CD', command=pagehistory)

label1 = Label(window, text='Press for Speak', bg='#6959CD',
               fg='orange', font=('Verdana', 15))

var = StringVar()
labelemotion = tk.Label(window, font="bold", fg="white", bg="#6959CD", textvariable=var, relief="groove", width='15',
                        height='5',bd=0)

# Open a New Image
image_micro = Image.open("PICTURE\\micro.png")
# Resize Image using resize function
resized_image_micro = image_micro.resize((150, 150), Image.ANTIALIAS)
# Convert the image into PhotoImage
img2 = ImageTk.PhotoImage(resized_image_micro)

labelemoticon = tk.Label(window, bg='#6959CD')


def page2(window):
    prevbutton.place(x=0, y=0)
    label1.place(x=180, y=40)
    labelemotion.place(x=175, y=320)
    var.set("Mood Analysis:")
    labelemoticon.place(x=178, y=480)
    buttonaudio.place(x=180, y=80)
    labelemoticon.configure(image='')
    historybutton.place(x=453, y=0)
    canvas_lg.place(x=158, y=245)



def fonctionaudio():

    result = printsentiment()
    var.set("Mood Analysis:\n" + result)

    if result == 'Angry':
        # Open a New Image
        image_angry = Image.open("PICTURE\\angry.png")
        resized_image = image_angry.resize((150, 150), Image.ANTIALIAS)
        stgImg = ImageTk.PhotoImage(resized_image)
        labelemoticon.configure(image=stgImg)
        labelemoticon.image = stgImg
        addAngry()

    if result == 'Happy':
        # Open a New Image
        image_happy = Image.open("PICTURE\\happy.png")
        resized_image = image_happy.resize((150, 150), Image.ANTIALIAS)
        stgImg = ImageTk.PhotoImage(resized_image)
        labelemoticon.configure(image=stgImg)
        labelemoticon.image = stgImg
        addHappy()

    if result == 'Surprised':
        # Open a New Image
        image_surprised = Image.open("PICTURE\\surprised.png")
        resized_image = image_surprised.resize((150, 150), Image.ANTIALIAS)
        stgImg = ImageTk.PhotoImage(resized_image)
        labelemoticon.configure(image=stgImg)
        labelemoticon.image = stgImg
        addSurprised()

    if result == 'Sad':
        # Open a New Image
        image_sad = Image.open("PICTURE\\sad.png")
        resized_image = image_sad.resize((150, 150), Image.ANTIALIAS)
        stgImg = ImageTk.PhotoImage(resized_image)
        labelemoticon.configure(image=stgImg)
        labelemoticon.image = stgImg
        addSad()


buttonaudio = tk.Button(window, image=img2, bg='#6959CD', command=lambda:[fonctionaudio(),myCanvas()])



'''
--------------------------------------------CODE PAGE 3--------------------------------------------------------------------------
'''


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sqlite3

prevbutton2 = tk.Button(window, image=prev_image, bg='#6959CD', command=topage2)
labelmood = Label(window, text='MOOD HISTORY', font='Helvetica 18 bold',fg='black',bg='#c4ffd2')

connexion = sqlite3.connect('database.db')
curseur = connexion.cursor() #Récupération d'un curseur

def create_Table_Emotions():
    # Create score table
    curseur.execute("""

    CREATE TABLE IF NOT EXISTS emotions(
    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
    emotion TEXT,
    valeur INTEGER); 
    """)

    #curseur.execute("""DELETE FROM emotions""")

    donnees = [
        ("angry", 0),
        ("happy", 0),
        ("surprised", 0),
        ("sad", 0),
    ]

    # Insert donnees
    curseur.executemany('''INSERT INTO emotions (emotion, valeur) VALUES (?, ?)''', donnees)
    # Validation
    connexion.commit()




def addAngry():
    sql = "UPDATE emotions SET valeur=valeur+1 WHERE emotion ='angry' "
    curseur.execute(sql)
    connexion.commit()


def getnumAngry():
    curseur.execute("SELECT  MAX(valeur) FROM emotions WHERE emotion ='angry'")
    mytuple=curseur.fetchone()
    my_num_angry=mytuple[0]
    return my_num_angry


def addHappy():
    sql = "UPDATE emotions SET valeur=valeur+1 WHERE emotion ='happy' "
    curseur.execute(sql)
    connexion.commit()
    #print_emotions()

def getnumHappy():
    curseur.execute("SELECT  MAX(valeur) FROM emotions WHERE emotion ='happy'")
    mytuple=curseur.fetchone()
    my_num_happy=mytuple[0]
    return my_num_happy

def addSurprised():
    sql = "UPDATE emotions SET valeur=valeur+1 WHERE emotion ='surprised' "
    curseur.execute(sql)
    connexion.commit()

def getnumSurprised():
    curseur.execute("SELECT  MAX(valeur) FROM emotions WHERE emotion ='surprised'")
    mytuple=curseur.fetchone()
    my_num_surprised=mytuple[0]
    return my_num_surprised


def addSad():
    sql = "UPDATE emotions SET valeur=valeur+1 WHERE emotion ='sad' "
    curseur.execute(sql)
    connexion.commit()

def getnumSad():
    curseur.execute("SELECT  MAX(valeur) FROM emotions WHERE emotion ='sad'")
    mytuple=curseur.fetchone()
    my_num_sad=mytuple[0]
    return my_num_sad





def page3(window):

    Label(window, text= 'Angry count :', font=("Times New Roman", 16),bg='#6959CD').place(x=100,y=610)
    Label(window, text='Happy count :', font=("Times New Roman", 16),bg='#6959CD').place(x=100,y=580)
    Label(window, text='Surprised count :', font=("Times New Roman", 16),bg='#6959CD').place(x=100,y=640)
    Label(window, text='Sad count :', font=("Times New Roman", 16), bg='#6959CD').place(x=100,y=670)

    Label(window, text=str(getnumAngry()) , font=("Times New Roman", 16),bg='#6959CD').place(x=270, y=610)
    Label(window, text=str(getnumHappy()), font=("Times New Roman", 16),bg='#6959CD').place(x=270, y=580)
    Label(window, text=str(getnumSurprised()), font=("Times New Roman", 16),bg='#6959CD').place(x=270, y=640)
    Label(window, text=str(getnumSad()), font=("Times New Roman", 16), bg='#6959CD').place(x=270, y=670)

    prevbutton2.place(x=0, y=0)
    labelmood.place(x=170, y=70)

    frameChartsLT = tk.Frame(window, width=900, height=400)
    frameChartsLT.place(x=0,y=100)

    stockListExp = ['ANGRY', 'HAPPY', 'SURPRISED','SAD']
    stockSplitExp = [getnumAngry(), getnumHappy(), getnumSurprised(),getnumSad()]

    fig = Figure(figsize=(4,4), dpi=120)  # create a figure object
    fig.set_facecolor('#6959CD')
    ax = fig.add_subplot(111)  # add an Axes to the figure
    ax.pie(stockSplitExp, radius=1, labels=stockListExp, autopct='%0.2f%%', shadow=True, )

    chart1 = FigureCanvasTkAgg(fig, frameChartsLT)
    chart1.get_tk_widget().pack()


create_Table_Emotions()

pagenum = 1
page1(window)
window.mainloop()
connexion.close()
