#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional -
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Filtros de particulas.

from math import *
from robot import *
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import select
from datetime import datetime
# ******************************************************************************
# Declaración de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

def pinta(secuencia,args):
  # Dibujar una secuencia de puntos
  t = np.array(secuencia).T.tolist()
  plt.plot(t[0],t[1],args)

def mostrar(objetivos,trayectoria,trayectreal,filtro):
  # Mostrar mapa y trayectoria
  plt.ion() # modo interactivo
  plt.clf()
  plt.axis('equal')
  # Fijar los bordes del gráfico
  objT   = np.array(objetivos).T.tolist()
  bordes = [min(objT[0]),max(objT[0]),min(objT[1]),max(objT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar mapa
  for p in filtro:
    dx = cos(p.orientation)*.05
    dy = sin(p.orientation)*.05
    plt.arrow(p.x,p.y,dx,dy,head_width=.05,head_length=.05,color='k')
  pinta(trayectoria,'--g')
  pinta(trayectreal,'-r')
  pinta(objetivos,'-.ob')
  p = hipotesis(filtro)
  dx = cos(p[2])*.05
  dy = sin(p[2])*.05
  plt.arrow(p[0],p[1],dx,dy,head_width=.075,head_length=.075,color='m')
  # Mostrar y comprobar pulsaciones de teclado:
  plt.show()
#  if sys.stdin in select.select([sys.stdin],[],[],.01)[0]:
#    line = sys.stdin.readline()
#  input()
  plt.pause(0.01)
  # plt.close()

def genera_filtro(num_particulas, balizas, real, centro=[2,2], radio=3):
  # Inicialización de un filtro de tamaño 'num_particulas', cuyas partículas
  # imitan a la muestra dada y se distribuyen aleatoriamente sobre un área dada.
  
  # Creamos tantos robots como partículas y establecemos el ruido
  filtro = [robot() for particula in range(num_particulas)]
  for particula in filtro:
    particula.set_noise(0.01,0.01,0.01)
  
  # Colocamos las partículas en una posición aleatoria con el centro y radio. Además se orientan.
  for particula in filtro:
    x = random.uniform(centro[0] - radio, centro[0] + radio)
    y = random.uniform(centro[1] - radio, centro[1] + radio)
    particula.set(x,y, real.orientation + random.gauss(0,0.1))
  
  # El peso de cada particula corresponde a las mediciones (mayor peso = menos error/diferencia entre mediciones)
  for particula in filtro:
    particula.weight = particula.measurement_prob(real.sense(balizas), balizas)
  
  return filtro

def dispersion(filtro):
  # Dispersion espacial del filtro de particulas
  valores_x = []
  valores_y = []
  # Obtener las componentes por separado
  for particula in filtro:
    valores_x.append(particula.x)
    valores_y.append(particula.y)

  # Calcular los mínimos y máximos
  min_x, max_x = min(valores_x), max(valores_x)
  min_y, max_y = min(valores_y), max(valores_y)
  return [min_x, max_x, min_y, max_y]

def peso_medio(filtro):  
  # Sumar todos los pesos
  peso_total = sum(particula.weight for particula in filtro)

  # Dividir cada peso entre el total
  for particula in filtro:
    particula.weight /= peso_total

  return filtro

# ******************************************************************************

random.seed(0)

# Definición del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posición y orientacion)
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (º/s)
FPS       = 10.        # Resolución temporal (fps)
HOLONOMICO = 0         # Robot holonómico
GIROPARADO = 0         # Si tiene que tener vel. lineal 0 para girar
LONGITUD   = .1        # Longitud del robot

N_PARTIC  = 100         # Tamaño del filtro de partículas
N_INICIAL = 2000       # Tamaño inicial del filtro

# Definición de trayectorias:
trayectorias = [
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.4*pi*i),2+2*cos(.4*pi*i)] for i in range(5)],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)],
    [[2+2*sin(1.2*pi*i),2+2*cos(1.2*pi*i)] for i in range(5)],
    [[2*(i+1),4*(1+cos(pi*i))] for i in range(6)],
    [[2+.2*(22-i)*sin(.1*pi*i),2+.2*(22-i)*cos(.1*pi*i)] for i in range(20)],
    [[2+(22-i)/5*sin(.1*pi*i),2+(22-i)/5*cos(.1*pi*i)] for i in range(20)]
    ]

# Definición de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(sys.argv[0]+" <indice entre 0 y "+str(len(trayectorias)-1)+">")
objetivos = trayectorias[int(sys.argv[1])]

# Definición de constantes:
EPSILON = .1                # Umbral de distancia
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

real = robot()
real.set_noise(.01,.01,.01) # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

#inicialización del filtro de partículas y de la trayectoria
##############################################
filtro = genera_filtro(N_INICIAL, objetivos, real, centro=[P_INICIAL[0], P_INICIAL[1]], radio=1)
pose = hipotesis(filtro) # Cogemos la partícula más probable
trayectoria = [pose]    # Iniciamos la trayectoria como la partícula más probable

trayectreal = [real.pose()]


tiempo  = 0.
espacio = 0.
for punto in objetivos:
  while distancia(trayectoria[-1],punto) > EPSILON and len(trayectoria) <= 1000:

    #seleccionar pose (usando la partícula más probable)
    pose = hipotesis(filtro)

    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0
    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:v = 0
      real.move(w,v)
    else:
      real.move_triciclo(w,v,LONGITUD)
 

    # Seleccionar hipótesis de localización y actualizar la trayectoria
    
    # Movemos todas las partículas 
    for particula in filtro:
      if HOLONOMICO:
        if GIROPARADO and abs(w) > .01: v = 0
        particula.move(w, v)
      else:
        particula.move_triciclo(w, v, LONGITUD)
        
    # Recalcular pesos
    for part in filtro:
      part.weight = part.measurement_prob(real.sense(objetivos), objetivos)
    filtro = peso_medio(filtro)
	
    trayectoria.append(pose)
    trayectreal.append(real.pose())
    mostrar(objetivos,trayectoria,trayectreal,filtro)

    # remuestreo
    filtro = resample(filtro, N_PARTIC)

    espacio += v
    tiempo  += 1

if len(trayectoria) > 1000:
  print ("<< ! >> Puede que no se haya alcanzado la posicion final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s" )
print ("Error medio de la trayectoria: "+str(round(sum(\
    [distancia(trayectoria[i],trayectreal[i])\
    for i in range(len(trayectoria))])/tiempo,3))+"m" )
input()

