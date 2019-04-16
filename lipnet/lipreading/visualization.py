import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import wave
import numpy as np

def show_video_subtitle(frames, subtitle):
    fig, ax = plt.subplots()
    fig.show()

    text = plt.text(0.5, 0.1, "", 
        ha='center', va='center', transform=ax.transAxes, 
        fontdict={'fontsize': 15, 'color':'white', 'fontweight': 500})
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
        path_effects.Normal()])

    subs = subtitle.split()
    inc = max(len(frames)/(len(subs)+1), 0.01)

    i = 0
    img = None
    for frame in frames:
        sub = " ".join(subs[:int(i/inc)])

        text.set_text(sub)

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)
        fig.canvas.draw()
        i += 1

def show_square(sq,avg_sq):
    plt.figure()
    f, axes = plt.subplots(2, 1)
    axes[0].plot(sq)
    axes[0].set_ylabel('square')
    axes[1].plot(avg_sq)
    axes[1].set_ylabel('disp')

    plt.show()



def show_sq_audio(sq,path):
    spf = wave.open(path,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')

    
    plt.title('Signal ')
    plt.plot(signal)
    plt.plot(sq)
    plt.show()
