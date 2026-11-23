"""Draw the study guide's labeled camera-geometry illustrations."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'graphics'
OUT.mkdir(exist_ok=True)
INK='#203247'; BLUE='#2265b7'; GREEN='#09856e'; ORANGE='#ba5a13'; GRAY='#718096'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,
                     'mathtext.fontset':'dejavusans','axes.edgecolor':INK})

def arrow(ax,a,b,color,label=None,offset=(0,0),size=15):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'<->','color':color,'lw':2,
                                          'shrinkA':0,'shrinkB':0})
    if label:
        ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,
                color=color,fontsize=size,ha='center',va='center',
                bbox={'facecolor':'white','edgecolor':'none','pad':2})

def title(fig,heading,sub):
    fig.text(.055,.95,heading,fontsize=23,weight='bold')
    fig.text(.055,.905,sub,fontsize=12,color=GRAY)

def pinhole():
    fig=plt.figure(figsize=(12,8),facecolor='white')
    title(fig,'Pinhole geometry: locate f, X, Z, and u',
          'Top view in the X–Z plane. The sideways X direction is drawn upward on this page.')
    ax=fig.add_axes([.06,.32,.88,.54]);ax.set_xlim(-1.1,12);ax.set_ylim(-2.15,5.35);ax.axis('off')
    f=2.5; Z=9.6; X=3.8; u=f*X/Z
    ax.add_patch(Polygon([(0,0),(Z,0),(Z,X)],facecolor=GREEN,alpha=.07))
    ax.add_patch(Polygon([(0,0),(f,0),(f,u)],facecolor=BLUE,alpha=.16))
    ax.annotate('',xy=(11,0),xytext=(-.65,0),arrowprops={'arrowstyle':'->','lw':1.4,'color':GRAY})
    ax.text(10.85,-.25,'Optical axis (+Z)',ha='right',color=GRAY)
    ax.plot([f,f],[-.25,4.65],color=BLUE,lw=2.3)
    ax.text(f,4.95,'Virtual image plane\n(in front of the pinhole)',ha='center',fontsize=12,color=BLUE)
    ax.plot([0,Z],[0,X],color=INK,lw=2)
    ax.plot([Z,Z],[0,X],color=GREEN,ls='--',lw=1.3)
    ax.scatter([0],[0],s=95,color=INK,zorder=6)
    ax.text(-.15,.4,'C: pinhole /\ncamera center',ha='right',fontsize=12)
    ax.scatter([Z],[X],s=85,color=GREEN,zorder=6)
    ax.text(Z-.1,X+.55,'Scene point P',ha='center',color=GREEN,weight='bold')
    ax.scatter([f],[u],s=65,color=BLUE,zorder=6)
    ax.annotate('Projected point',xy=(f,u),xytext=(4.5,1.05),fontsize=11,color=BLUE,
                arrowprops={'arrowstyle':'->','color':BLUE})
    arrow(ax,(Z+.55,0),(Z+.55,X),GREEN,'X',(.25,0),19)
    arrow(ax,(f-.22,0),(f-.22,u),BLUE,'u',(-.25,.06),17)
    ax.plot([0,0],[-1.6,0],color=GRAY,lw=.8,ls=':')
    ax.plot([f,f],[-.8,0],color=BLUE,lw=.8,ls=':')
    ax.plot([Z,Z],[-1.6,0],color=ORANGE,lw=.8,ls=':')
    arrow(ax,(0,-.65),(f,-.65),BLUE,'f',(0,-.08),18)
    arrow(ax,(0,-1.45),(Z,-1.45),ORANGE,'Z',(0,-.08),18)
    ax.text(7.4,4.9,'Same ray → similar triangles',ha='center',fontsize=12)
    fig.text(.075,.255,r'$\frac{u}{f}=\frac{X}{Z}\qquad\Rightarrow\qquad u=\frac{fX}{Z}$',fontsize=24)
    fig.text(.57,.255,'The small blue triangle is a scaled copy\nof the large scene triangle.',fontsize=12)
    fig.text(.075,.16,'f  = camera center → image plane',color=BLUE,fontsize=13,weight='bold')
    fig.text(.075,.105,'u = image position, measured from the axis',color=BLUE,fontsize=12)
    fig.text(.57,.16,'X = sideways offset of the scene point',color=GREEN,fontsize=12,weight='bold')
    fig.text(.57,.105,'Z = depth measured along the optical axis',color=ORANGE,fontsize=12,weight='bold')
    fig.text(.075,.035,'The physical sensor sits behind the pinhole and gives an inverted image; the virtual plane makes the coordinates positive.',
             fontsize=10.5,color=GRAY)
    fig.savefig(OUT/'02-pinhole-labeled.png',dpi=180,facecolor='white');plt.close(fig)

def camera(ax,x,y,opening=.18):
    ax.add_patch(Rectangle((x-.43,y-.3),.52,.6,facecolor=INK,edgecolor='none'))
    ax.add_patch(Rectangle((x+.09,y-.2),.16,.4,facecolor=INK,edgecolor='none'))
    ax.add_patch(Circle((x-.17,y),opening,facecolor='white',edgecolor='white'))

def tree(ax,x,y,color,soft=False):
    # Repeated faint outlines communicate defocus; the text gives the exact status.
    for dx,dy,alpha in ([(-.06,0,.08),(.06,0,.08),(0,.06,.08),(0,-.06,.08),(0,0,.25)] if soft else [(0,0,1)]):
        ax.plot([x+dx,x+dx],[y+dy,y+.42+dy],lw=4,color=color,alpha=alpha)
        ax.add_patch(Polygon([(x-.3+dx,y+.22+dy),(x+.3+dx,y+.22+dy),(x+dx,y+1.02+dy)],
                             facecolor=color,edgecolor='none',alpha=alpha))

def dof():
    fig=plt.figure(figsize=(12,8.4),facecolor='white')
    title(fig,'Depth of field (DoF): which distances look sharp?',
          'Keep focal length, focus distance, and the acceptable-sharpness criterion fixed. Change only the aperture.')
    for index,(heading,near,far,large) in enumerate([
        ('Large aperture / low f-number → shallow DoF',4.9,6.9,True),
        ('Small aperture / high f-number → deep DoF',2.6,9.6,False)]):
        bottom=.51 if index==0 else .17
        ax=fig.add_axes([.06,bottom,.88,.29]);ax.set_xlim(0,11);ax.set_ylim(-1.2,2.1);ax.axis('off')
        ax.text(.1,1.78,heading,fontsize=16,weight='bold')
        ax.add_patch(Rectangle((near,-.05),far-near,1.55,facecolor='#dbf2e9',edgecolor=GREEN,lw=1))
        ax.text((near+far)/2,1.3,'Acceptably sharp range',ha='center',fontsize=11,color=GREEN)
        ax.axvline(6.1,ymin=.19,ymax=.82,color=ORANGE,lw=1.5,ls='--')
        camera(ax,.8,.4,.21 if large else .075)
        ax.text(.7,-.12,'Camera',ha='center',fontsize=11)
        for x,label in [(3.5,'Foreground'),(6.1,'Focus distance'),(8.7,'Background')]:
            sharp=near<x<far
            tree(ax,x,.05,GREEN if sharp else GRAY,not sharp)
            ax.text(x,-.16,label,ha='center',va='top',fontsize=10.5)
            ax.text(x,-.49,'SHARP' if sharp else 'OUT OF FOCUS',ha='center',va='top',fontsize=10,
                    color=GREEN if sharp else GRAY,weight='bold')
        arrow(ax,(near,-.94),(far,-.94),GREEN,'DoF',(0,0),14)
        ax.text(10.6,.2,'Farther\naway →',ha='center',fontsize=10.5,color=GRAY)
    fig.text(.07,.075,'Same focus distance in both views',fontsize=13,color=ORANGE,weight='bold')
    fig.text(.52,.075,'Smaller opening → a larger range of sharp depths',fontsize=12,color=GREEN)
    fig.text(.07,.025,'The colored band is a range of scene distances. Its boundaries depend on how much blur you consider acceptable. Schematic, not to scale.',
             fontsize=10,color=GRAY)
    fig.savefig(OUT/'02-depth-of-field.png',dpi=180,facecolor='white');plt.close(fig)

def fov():
    fig=plt.figure(figsize=(12,8.4),facecolor='white')
    title(fig,'Field of view (FoV): how wide a scene fits?',
          'Same camera position and same 36 mm sensor width. Compare a 24 mm lens with a 50 mm lens.')
    for index,(f_mm,subtitle) in enumerate([(24,'Shorter focal length → wider FoV'),(50,'Longer focal length → narrower FoV')]):
        ax=fig.add_axes([.055+index*.48,.21,.43,.62]);ax.set_xlim(-1.2,10.9);ax.set_ylim(-7.8,8.1);ax.axis('off');ax.set_aspect('equal')
        f=f_mm/10;halfwidth=1.8;z=9;halfscene=z*halfwidth/f
        angle=math.degrees(2*math.atan(halfwidth/f))
        ax.text(4.4,7.65,f'{f_mm} mm lens',ha='center',fontsize=18,weight='bold')
        ax.add_patch(Polygon([(0,0),(z,halfscene),(z,-halfscene)],facecolor=BLUE,alpha=.12))
        ax.plot([0,z],[0,halfscene],color=BLUE,lw=1.7)
        ax.plot([0,z],[0,-halfscene],color=BLUE,lw=1.7)
        ax.plot([0,9.5],[0,0],color=GRAY,lw=.8,ls='--')
        ax.plot([f,f],[-halfwidth,halfwidth],color=ORANGE,lw=4)
        ax.text(4.4,6.5,'Sensor width w = 36 mm',ha='center',fontsize=10.5,color=ORANGE)
        ax.scatter([0],[0],s=60,color=INK,zorder=7)
        ax.text(-.15,-.9,'Camera\ncenter',ha='center',fontsize=10)
        ax.add_patch(Arc((0,0),3.4,3.4,theta1=-angle/2,theta2=angle/2,color=BLUE,lw=2))
        ax.text(2.0,.22,f'{angle:.1f}°',fontsize=14,color=BLUE,weight='bold')
        ax.plot([z,z],[-7,7],color=GRAY,lw=1,ls=':')
        for y in [-5.4,-1.8,1.8,5.4]:
            inside=abs(y)<halfscene
            ax.add_patch(Circle((z,y),.22,facecolor=GREEN if inside else '#c7cdd5',edgecolor='none'))
        ax.text(z,7.05,'Same scene',ha='center',fontsize=10)
        arrow(ax,(0,-7.15),(f,-7.15),ORANGE,f'f = {f_mm} mm',(0,.4),12)
        fig.text(.055+index*.48,.175,subtitle,fontsize=13,weight='bold',color=BLUE)
    fig.text(.075,.095,r'$\mathrm{FoV}=2\arctan\!\left(\frac{w}{2f}\right)$',fontsize=23)
    fig.text(.53,.105,'w = sensor width; f = focal length\nGreen dots fall inside the captured angle.',fontsize=12)
    fig.text(.075,.035,'Virtual image planes are drawn in front of the camera, using the same construction as the pinhole diagram. Schematic, not to scale.',
             fontsize=10,color=GRAY)
    fig.savefig(OUT/'02-field-of-view.png',dpi=180,facecolor='white');plt.close(fig)

if __name__=='__main__':
    pinhole();dof();fov()
    print('Created 3 labeled illustrations in',OUT)
