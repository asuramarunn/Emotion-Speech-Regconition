import numpy as np
import librosa, pygame
from lib.preprocessing import decode_emotion, decode_emotion_vi


def atdb(m):
    m = m.copy()
    m[m < 1e-10] = 1e-10
    return 20 * np.log10(m)
    
    

def maurgb(r=255, g=120, b=0):
    dr = []
    x=1
    y=1
    z=1
    for i in range(128):
        if (r>=253 and x==1) or (r<2 and x==-1):
            x=-x
        if (g>=253 and y==1) or (g<2 and y==-1):
            y=-y
        if (b>=253 and z==1) or (b<2 and z==-1):
            z=-z
        r = r+2*x
        g = g+2*y
        b = b+2*z
        
        dr.append((r, g, b))
    return dr


def camxuc(cx):
    be_mat = pygame.Surface((500, 320)) 
    font = pygame.font.SysFont("Calibri", 60)   
    pygame.draw.rect(be_mat, "red", [0, 0, 500, 320], border_radius=50)
    pygame.draw.rect(be_mat, "black", [2, 2, 496, 316], border_radius=50)
    text_cx = decode_emotion(cx)
    text_cx_vi = decode_emotion_vi(cx)
    tcx = font.render(text_cx, True, (255, 226, 244))
    tcxvi = font.render(text_cx_vi, True, (255, 226, 244))
    match cx:
        case 0:
            be_mat.blit(tcx, (160, 90))
            be_mat.blit(tcxvi, (145, 165))
        case 1:
            be_mat.blit(tcx, (175, 90))
            be_mat.blit(tcxvi, (178, 165))
        case 2:
            be_mat.blit(tcx, (210, 90))
            be_mat.blit(tcxvi, (190, 165))
        case 3:
            be_mat.blit(tcx, (190, 90))
            be_mat.blit(tcxvi, (160, 165))
        case 4:
            be_mat.blit(tcx, (210, 90))
            be_mat.blit(tcxvi, (190, 165))
        case 5:
            be_mat.blit(tcx, (165, 90))
            be_mat.blit(tcxvi, (93, 165))
        case 6:
            be_mat.blit(tcx, (155, 90))
            be_mat.blit(tcxvi, (120, 165))
        case _:
            pass
    return be_mat

def thanhtiendo(x):
    be_mat = pygame.Surface((250, 30))    
    pygame.draw.rect(be_mat, "red", [0, 0, 250, 30], border_radius=14)
    pygame.draw.rect(be_mat, "black", [2, 2, 246, 26], border_radius=14)
    pygame.draw.rect(be_mat, "blue", [2, 2, x*246/100, 26], border_radius=14)
    return be_mat

