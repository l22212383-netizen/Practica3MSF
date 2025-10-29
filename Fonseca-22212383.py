"""
Práctica 2: Sistema cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Ivan De Jesus Fonseca Diaz
Número de control: 22212383
Correo institucional: l22212383@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiologicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx', header=None))

# Datos de la simulación
x0, t0, tF, dt, w, h = 0, 0, 10, 1E-3, 10, 5
N = round((tF-t0)/dt) + 1
t = np.linspace(t0, tF, N)
u = np.zeros(N); u[round(1/dt):round(2/dt)] = 1


# Componentes del circuito RLC y funcion de transferencia
def muscle(R,Cs,Cp):
    num = [Cs*R,0.75]
    den = [R*(Cs+Cp),1]
    sys = ctrl.tf(num,den)
    return sys


# Control
R, Cs, Cp = 100, 10e-6, 100e-6
syscontrol = muscle(R,Cs,Cp)
print(f"Control: {syscontrol}")

# Caso
R, Cs, Cp = 10e3, 10e-6, 100e-6
syscaso = muscle(R,Cs,Cp)
print(f"Caso: {syscaso}")


#Respuestas en lazo abierto
clr1 = np.array([230, 39, 39])/255
clr2 = np.array([0, 0, 0])/255
clr3 = np.array([67, 0, 255])/255
clr4 = np.array([22, 97, 14])/255
clr5 = np.array([250, 129, 47])/255
clr6 = np.array([145, 18, 188])/255

_,Pp0 = ctrl.forced_response(syscontrol,t,u,x0)
_,Pp1 = ctrl.forced_response(syscaso,t,u,x0)

fg1 = plt.figure()
plt.plot(t,u,'-', color = clr3, label = 'Fs(t)')
plt.plot(t,Pp0,'-',linewidth=1, color = clr1, label='Fs1(t): Control')
plt.plot(t,Pp1,'-',linewidth=1, color = clr2, label='Fs2(t): Caso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t[s]')
plt.ylabel('Fs(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema musculoesqueletico python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema musculoesqueletico python.pdf',bbox_inches='tight')


def controlador(kP,kI):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    return PI

PI = controlador(0.0219712691125975,41303.6112882169)
X = ctrl.series(PI,syscaso)
sysPI = ctrl.feedback(X,1,sign=-1)


_,Pp2 = ctrl.forced_response(sysPI,t,Pp0,x0)


fg2 = plt.figure()
plt.plot(t,Pp2,':',linewidth=2, color = clr3, label='Fs(t): Tratamiento')
plt.plot(t,Pp0,'-',linewidth=1, color = clr1, label='Fs(t): Control')
plt.plot(t,Pp1,'-',linewidth=1, color = clr2, label='Fs(t): Caso')
plt.plot(t,u,'-', color = clr4, label = 'Fs(t)')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t[s]')
plt.ylabel('Fs(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=5)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema cardiovascular PI python.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema cardiovascular PI python.pdf',bbox_inches='tight')

